import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import random
import builtins
import os
import sys
from augmentation import *
from model import *
from dataloader import Kinetics_Data, DataAllocate
import argparse
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--seq', default=16, type=int)
    parser.add_argument('--sample', default=2, type=int)
    parser.add_argument('--rate', default=3.0, type=float)
    parser.add_argument('--train_batch', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--img_dim', default=112, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--pos', default=6, type=int)
    parser.add_argument('--neg', default=201, type=int)
    parser.add_argument('--cluster', default=1000, type=int)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--csv_file', default='kinetics.csv', type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--split', default=20, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--path', default='', type=str)
    parser.add_argument('--mode', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr_decay', default=70, type=int)
    parser.add_argument('--thres', default=0.05, type=float)
    # parallel configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # for torch.distributed.launch
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    return args
    
def feat_contrast(feat, feat_full, criterion):
    """Contrastive learning of high-level feature representation, instance discrimination"""
    b, n, c = feat_full.shape
    replica_id = dist.get_rank()
    ngpus = torch.cuda.device_count()
    assert feat_full.shape[0] // feat.shape[0] == ngpus
    n_pos = n
    feat = F.normalize(feat, dim=-1)
    feat_full = F.normalize(feat_full, dim=-1)
    similarity = torch.mm(feat_full[:, 0], feat_full[:, 1].transpose(0, 1))
    similarity = torch.softmax(similarity/0.07, dim=1)
    num_neg = (b-1) * n_pos
    pools = []
    for i in range(b//ngpus):
        pools.append(torch.cat([feat_full[:i+b//ngpus*replica_id].contiguous().view(-1, c), 
                                feat_full[(i+1)+b//ngpus*replica_id:].contiguous().view(-1, c)], 0))
    pools = torch.stack(pools, 0).view(b//ngpus, 1, num_neg, c).repeat([1, n_pos, 1, 1]).\
        view(b//ngpus*n_pos, num_neg, c)
    pos = torch.einsum('bnc,bmc->bnm', feat, feat) / 0.07
    neg = torch.einsum('nc,nmc->nm', feat.contiguous().view(b//ngpus*n_pos, c), pools) / 0.07
    eff = torch.ones(n_pos, n_pos).to(device)
    eff[torch.arange(n_pos), torch.arange(n_pos)] = 0
    pos = torch.sum(torch.exp(pos)*eff.unsqueeze(0), -1).view(b//ngpus*n_pos)
    neg = torch.sum(torch.exp(neg), 1)
    prob = torch.stack([pos, neg], 1)
    prob = prob / prob.sum(1, keepdim=True)
    labels = torch.zeros(prob.shape[0]).to(device).long()
    loss = criterion(torch.log(prob), labels)
    acc = torch.sum(torch.argmax(prob, 1)==0).float()
    return loss.mean(), acc/(b//ngpus*n_pos), similarity

def graph_contrast(feat, feat_full, adj):
    """Graph constraint inferred from feature distribution as self-supervision"""
    b, n, c = feat_full.shape
    replica_id = dist.get_rank()
    ngpus = torch.cuda.device_count()
    assert feat_full.shape[0] // feat.shape[0] == ngpus
    feat = F.normalize(feat, dim=-1)
    feat_full = F.normalize(feat_full, dim=-1)
    similarity = torch.mm(feat[:, 0], feat_full[:, 1].transpose(0, 1))
    similarity = torch.softmax(similarity/0.07, dim=1)
    loss = - torch.log(similarity+1e-10)
    loss = loss * adj[b//ngpus*replica_id: b//ngpus*(replica_id+1)]
    return loss.mean()

def motion_contrast(feat, feat_full, criterion):
    """Temporal enhanced motion pattern contrastive learning"""
    num_aug = 2
    b, n, c = feat_full.shape
    b = b // num_aug
    replica_id = dist.get_rank()
    ngpus = torch.cuda.device_count()
    assert feat_full.shape[0] // feat.shape[0] == ngpus
    feat = feat.view(num_aug, b//ngpus, n ,c)
    feat_full = feat_full.view(ngpus, num_aug, b//ngpus, n, c).permute(1, 0, 2, 3, 4).contiguous()
    feat_full = feat_full.view(num_aug, b, n, c)
    n_pos = n
    feat = F.normalize(feat, dim=-1)
    feat_full = F.normalize(feat_full, dim=-1)
    query = feat_full[0]
    num_neg = (b-1) * n_pos + (num_aug-1) * n_pos
    pools = []
    for i in range(b//ngpus):
        pools.append(torch.cat([query[:i+b//ngpus*replica_id].contiguous().view(-1, c), 
                                feat[1:, i].contiguous().view(-1, c),
                                query[(i+1)+b//ngpus*replica_id:].contiguous().view(-1, c)], 0))
    pools = torch.stack(pools, 0).view(b//ngpus, 1, num_neg, c).repeat([1, n_pos, 1, 1]).\
        view(b//ngpus*n_pos, num_neg, c)
    pos = torch.einsum('bnc,bmc->bnm', feat[0], feat[0]) / 0.07
    neg = torch.einsum('nc,nmc->nm', feat[0].contiguous().view(b//ngpus*n_pos, c), pools) / 0.07
    eff = torch.ones(n_pos, n_pos).to(device)
    eff[torch.arange(n_pos), torch.arange(n_pos)] = 0
    pos = torch.sum(torch.exp(pos)*eff.unsqueeze(0), -1).view(b//ngpus*n_pos)
    neg = torch.sum(torch.exp(neg), 1)
    prob = torch.stack([pos, neg], 1)
    prob = prob / prob.sum(1, keepdim=True)
    labels = torch.zeros(prob.shape[0]).to(device).long()
    loss = criterion(torch.log(prob), labels)
    acc = torch.sum(torch.argmax(prob, 1)==0).float()
    return loss.mean(), acc/(b//ngpus*n_pos)

def local_global(feat, criterion):
    """Timestamp retrieval between clips"""
    b, n, c, t = feat.shape
    former = feat[:, 0]
    latter = feat[:, 1]
    overview = feat[:, -1]
    former = F.adaptive_avg_pool1d(former, t//2)
    latter = F.adaptive_avg_pool1d(latter, t//2)
    original = torch.cat([former, latter], dim=-1)
    assert original.shape == overview.shape
    original = F.normalize(original, dim=1)
    overview = F.normalize(overview, dim=1)
    l2g = torch.einsum('bcm,bcn->bmn', original, overview) / 0.07
    l2g = l2g.view(b*t, t)
    g2l = torch.einsum('bcm,bcn->bmn', overview, original) / 0.07
    g2l = g2l.view(b*t, t)
    label = torch.arange(t).unsqueeze(0).repeat([b, 1]).view(b*t).to(device)
    acc = torch.sum(torch.argmax(l2g, 1)==label).float() + torch.sum(torch.argmax(g2l, 1)==label).float()
    loss = criterion(l2g, label).mean() + criterion(g2l, label).mean()
    return loss/2, acc/(2*b*t)

def proto_contrast(logit, prob):
    """Prototypical contrastive learning on high-level semantics"""
    b, n, k = logit.shape
    replica_id = dist.get_rank()
    ngpus = torch.cuda.device_count()
    q_s = sinkhorn(prob[:, 0])[-b*ngpus:]
    q_t = sinkhorn(prob[:, 1])[-b*ngpus:]
    pq_s = q_s[-b*(ngpus-replica_id): -b*(ngpus-replica_id-1)] if (ngpus-replica_id-1) > 0 else q_s[-b*(ngpus-replica_id):]
    pq_t = q_t[-b*(ngpus-replica_id): -b*(ngpus-replica_id-1)] if (ngpus-replica_id-1) > 0 else q_t[-b*(ngpus-replica_id):]
    prob = F.softmax(logit/0.1, dim=-1)
    loss = - 0.5 * (torch.sum(pq_t*torch.log(prob[:, 0]+1e-10), dim=-1)+
                    torch.sum(pq_s*torch.log(prob[:, 1]+1e-10), dim=-1))
    return loss.mean(), q_s, q_t

def sinkhorn(scores, eps=0.05, niters=3):
    """SK cluster, from SWAV"""
    with torch.no_grad():
        M = torch.max(scores/eps)
        Q = scores/eps - M
        Q = torch.exp(Q).transpose(0, 1)
        Q = shoot_infs(Q)
        Q = Q / torch.sum(Q)
        K, B = Q.shape
        u, r, c = torch.zeros(K).to(device), torch.ones(K).to(device)/K, \
            torch.ones(B).to(device)/B
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q = Q * u.unsqueeze(1)
            Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).transpose(0, 1)

def shoot_infs(inp_tensor):
    """SK cluster, from SWAV"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def all_gather(tensor, expand_dim=0, num_replicas=4):
    """Gathers a tensor from other replicas, concat on expand_dim and return"""
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    other_replica_tensors[dist.get_rank()] = tensor
    return torch.cat([o for o in other_replica_tensors], expand_dim)

def train_label(model, dataloader, optimizer, nllloss, celoss, epoch, queue, args):
    model.train()
    total_loss = 0
    for idx, data in enumerate(dataloader):
        with torch.no_grad():
            w = model.module.prototype.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            model.module.prototype.weight.copy_(w)
        data = data.cuda(args.gpu, non_blocking=True)
        if epoch >= args.split:
            feat, fine, low, low_fine, mid, mid_fine, logit, feat_graph, mid_graph, low_graph = model(data, mode='multi')
        else:
            feat, fine, low, low_fine, mid, mid_fine, logit = model(data, mode='proto')
        
        del data
        
        feat_full = all_gather(feat)
        low_full = all_gather(low)
        mid_full = all_gather(mid)
        logit_full = all_gather(logit.contiguous())
        if epoch >= args.split:
            feat_graph_full = all_gather(feat_graph)
            mid_graph_full = all_gather(mid_graph)
            low_graph_full = all_gather(low_graph)
        process_queue(queue, logit_full)
        if epoch >= 10:
            prob = queue.clone()
        else:
            prob = logit_full.clone()
        lsd, q_s, q_t = proto_contrast(logit, prob.detach())
        label = torch.argmax(q_s+q_t, dim=-1)
        diverse = torch.unique(label).shape[0]
        lhigh, ach, ins = feat_contrast(feat[:, :-1].contiguous(), feat_full[:, :-1].contiguous(), nllloss)
        lfine, acf = local_global(fine, celoss)
        lmid, acm = motion_contrast(mid[:, :-1].contiguous(), mid_full[:, :-1].contiguous(), nllloss)
        lfmid, acfm = local_global(mid_fine, celoss)
        llow, acl = motion_contrast(low[:, :-1].contiguous(), low_full[:, :-1].contiguous(), nllloss)
        lflow, acfl = local_global(low_fine, celoss)
        
        if epoch >= args.split:
            ins, sem, com = graph(ins.detach(), label.detach(), args)
            lgraph = graph_contrast(feat_graph[:, :-1], feat_graph_full[:, :-1], com) + \
                0.5 * graph_contrast(mid_graph[:, :-1], mid_graph_full[:, :-1], com) + \
                0.3 * graph_contrast(low_graph[:, :-1], low_graph_full[:, :-1], com)
            loss = lhigh + lfine + lgraph + 0.5*(lmid+lfmid) + 0.3*(llow+lflow) + 0.5*lsd
        else:
            loss = lhigh + lfine + 0.5*(lmid+lfmid) + 0.3*(llow+lflow) + 0.5*lsd
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        if epoch == 0:
            for name, p in model.named_parameters():
                if "prototype" in name:
                    p.grad = None
        optimizer.step()
    
    return total_loss/idx

def graph(ins, label, args):
    """Graph distribution inference"""
    val = ins[torch.arange(ins.shape[0]), torch.arange(ins.shape[0])]
    ins[ins<args.thres] = 0
    ins[torch.arange(ins.shape[0]), torch.arange(ins.shape[0])] = val
    sem = torch.zeros_like(ins)
    for i in range(label.shape[0]):
        index = torch.arange(label.shape[0])[label==label[i]]
        sem[i, index] = 1
    com = ins * sem
    com = com / torch.sum(com, dim=1, keepdim=True)
    return ins, sem, com

def process_queue(queue, logit):
    """Prototype probability queue maintainance"""
    b, n, c = logit.shape
    length = b
    queue[:-length] = queue[length:].clone()
    queue[-length:] = logit.detach()
    
def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    
    global device; device = torch.device('cuda')
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = GreatModel(args.net, args.seq, args.sample, args.img_dim, args.channel)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model_without_ddp = model.module
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    celoss = nn.CrossEntropyLoss(reduce=True, size_average=True).cuda(args.gpu)
    nllloss = nn.NLLLoss(reduce=False, size_average=False).cuda(args.gpu)
    
    transform = transforms.Compose([
        RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
        RandomHorizontalFlip(consistent=True),
        RandomGray(consistent=False, p=0.5),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
        ToTensor(),
        Normalize()
    ])
    dataset = Kinetics_Data(file=args.csv_file, mode='train',
                            seq=args.seq, sample=args.sample, sr=[1.0, args.rate],
                            transform=transform)
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = data.DataLoader(dataset, batch_size=args.train_batch, sampler=train_sampler,
                                 collate_fn=DataAllocate, num_workers=args.workers, drop_last=True)
    
    start = 0
    if args.resume:
        state_dict = torch.load(args.path, map_location=torch.device('cpu'))
        model_without_ddp.load_state_dict(state_dict)
        print('load weight')
    torch.backends.cudnn.benchmark = True

    queue = torch.randn(1024, 2, args.cluster).cuda(args.gpu, non_blocking=True)
    print("Use RANK: {} for training".format(args.rank))
    # eval_grad(model, dataloader, nllloss)
    for e in range(start, args.epoch):
        print('Epoch %d'%e)
        if args.distributed:
            dataloader.sampler.set_epoch(e)
        if e == args.lr_decay:
            args.lr = 0.1 * args.lr
            for param in optimizer.param_groups:
                param['lr'] = args.lr
        
        loss = train_label(model, dataloader, optimizer, nllloss, celoss, e, queue, args)

        if (not args.multiprocessing_distributed and args.rank == 0) \
            or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            state_dict = model_without_ddp.state_dict()
            torch.save(state_dict, '../ckpt/r3d/r3d_%d_%.3f.pth'%(e, loss))

if __name__ == '__main__':
    args = parse_args()
    main(args)
