import torch
from torch.utils import data
import torch.optim as optim
import os
import sys
import cv2
from augmentation import *
from modeltemporal import *
from dataloader import Temporal_Data, DataAllocate
import argparse
from tqdm import tqdm
from progress.bar import Bar
import time
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='finetune', type=str)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--seq', default=16, type=int)
parser.add_argument('--sample', default=1, type=int)
parser.add_argument('--rate', default=1.0, type=float)
parser.add_argument('--test_rate', default=1.0, type=float)
parser.add_argument('--train_batch', default=256, type=int)
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--test_dim', default=224, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--channel', default=128, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--epoch', default=50, type=int)
parser.add_argument('--path', default='', type=str)

args = parser.parse_args()

def train(model, dataloader, optimizer, criterion):
    model.eval()
    start = time.time()
    bar = Bar('Processing', max=len(dataloader))
    total_loss = 0
    for idx, data in enumerate(dataloader):
        data = data.to(device)
        end = time.time()
        logit = model(data)
        label = torch.zeros(logit.shape[0])
        label[1::2] = 1 #one normal order, and one reverse order
        loss = criterion(logit, label.type(torch.LongTensor).to(device))
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.suffix = '({batch}/{size}) Time:{time:.3f}|Loss:{loss:.3f}'.format(
            batch=idx + 1,
            size=len(dataloader),
            time=end-start,
            loss=loss.item()
        )
        bar.next()
        start = time.time()
    bar.finish()
    return total_loss/len(dataloader)

def visualize(model, dataloader):
    model.eval()
    start = time.time()
    bar = Bar('Processing', max=len(dataloader))
    for idx, data in enumerate(dataloader):
        data = data.to(device)
        end = time.time()
        logit = model(data, training=False)
        logit = logit.view(logit.shape[0], 2, 7, 7, 2) #(n,t,h,w,c)
        returncam(data, logit, idx)
        bar.suffix = '({batch}/{size}) Time:{time:.3f}'.format(
            batch=idx + 1,
            size=len(dataloader),
            time=end-start
        )
        bar.next()
        start = time.time()
    bar.finish()

def returncam(data, activation, idx):
    batchsize = data.shape[0] // 2
    activation = activation.permute(0, 4, 1, 2, 3).contiguous() #(n,c,t,h,w)
    activation = torch.nn.functional.interpolate(activation, size=(16, 224, 224), mode='trilinear')
    assert activation.shape[2:] == data.shape[2:]
    images = data.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
    heatmaps = activation.permute(0, 2, 3, 4, 1).contiguous().cpu().detach().numpy() #(n,t,h,w,c)
    for b in range(batchsize):
        batch_idx = 2*b
        for k in range(16):
            img = images[batch_idx, k]*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = img * 255
            heatmap = heatmaps[batch_idx, k, :, :, 0]
            heatmap = 255 * (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
            colormap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            final = 0.5 * img[:, :, ::-1] + 0.5 * colormap
            cv2.imwrite('visualization/%d_%d.jpg'%(b+idx*batchsize,k), final)
            
            img = images[batch_idx+1, k]*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = img * 255
            heatmap = heatmaps[batch_idx+1, k, :, :, 1]
            heatmap = 255 * (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
            colormap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            final = 0.5 * img[:, :, ::-1] + 0.5 * colormap
            cv2.imwrite('visualreverse/%d_%d.jpg'%(b+idx*batchsize,k), final)

def evaluate(model, dataloader):
    model.eval()
    start = time.time()
    bar = Bar('Processing', max=len(dataloader))
    acc = 0
    sample = 0
    for idx, data in enumerate(dataloader):
        data = data.to(device)
        end = time.time()
        with torch.no_grad():
            logit = model(data)
            label = torch.zeros(logit.shape[0])
            label[1::2] = 1
        acc += torch.sum(torch.argmax(logit, 1)==label).item()
        sample += label.shape[0]
        bar.suffix = '({batch}/{size}) Time:{time:.3f}|Accuracy:{acc:.3f}'.format(
            batch=idx + 1,
            size=len(dataloader),
            time=end-start,
            acc=torch.sum(torch.argmax(logit, 1)==label).item()/label.shape[0]
        )
        bar.next()
        start = time.time()
    bar.finish()
    return acc/sample

def main():
    global device; device = torch.device('cuda')
    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=1.0),
        Scale(size=(args.img_dim,args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])

    val_transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(args.test_dim,args.test_dim)),
        ToTensor(),
        Normalize()
    ])
    
    trainset = Temporal_Data(file='ucf_train.csv',
                            seq=args.seq, sample=args.sample, sr=[1.0, args.rate],
                            transform=transform)
    trainloader = data.DataLoader(trainset, shuffle=True, batch_size=args.train_batch,
                                 collate_fn=DataAllocate, num_workers=args.workers,
                                 drop_last=True)

    valset = Temporal_Data(file='ucf_test.csv',
                            seq=args.seq, sample=args.sample, sr=[1.0, args.rate],
                            transform=val_transform)
    valloader = data.DataLoader(valset, shuffle=False, batch_size=1,
                                 collate_fn=DataAllocate, num_workers=args.workers,
                                 drop_last=True)
        
    model = GreatModel(args.net, args.seq, args.sample, args.img_dim, args.channel)
    start_epoch = 0
    if args.mode == 'train':
        model.load_state_dict(torch.load(args.path), strict=False)
    if args.mode == 'eval':
        model.load_state_dict(torch.load(args.path), strict=True)
    if args.resume:
        model.load_state_dict(torch.load(args.path), strict=True)
    model = model.to(device)
    
    if args.mode == 'eval':
        acc = evaluate(model, valloader)
        print('UCF-101 Temporal Order Accuracy: %.3f'%acc)
        visualize(model, valloader)
        return
    
    # freeze the backbone parameter and only train the FC head
    params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            params.append({'params': param, 'lr': args.lr})
        
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    for e in range(start_epoch, args.epoch):
        if e == 30:
            args.lr = 0.1 * args.lr
            for param in optimizer.param_groups:
                param['lr'] = 0.1 * param['lr']
        loss = train(model, trainloader, optimizer, criterion)
        torch.save(model.state_dict(), '../ckpt/temporal/%d.pth'%e)
    visualize(model, valloader)

if __name__ == '__main__':
    main()
