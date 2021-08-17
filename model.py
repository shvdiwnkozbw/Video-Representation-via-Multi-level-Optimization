import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from r3d import generate_model        

class GreatModel(nn.Module):
    def __init__(self, network='resnet18', seq=8, sample=4, size=224, channel=128, K=1000):
        super(GreatModel, self).__init__()
        self.backbone = generate_model(18)
        self.size = size / 32
        self.seq = seq
        self.sample = sample
        self.channel = channel
        self.cluster = K
        self.project = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=512, out_channels=self.channel, kernel_size=1)
        )
        self.project_graph = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=512, out_channels=self.channel, kernel_size=1)
        )
        self.project_fine = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=512, out_channels=self.channel, kernel_size=1)
        )
        self.project_low = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=self.channel, kernel_size=1)
        )
        self.project_low_graph = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=self.channel, kernel_size=1)
        )
        self.project_low_fine = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=self.channel, kernel_size=1)
        )
        self.project_mid = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=256, out_channels=self.channel, kernel_size=1)
        )
        self.project_mid_graph = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=256, out_channels=self.channel, kernel_size=1)
        )
        self.project_mid_fine = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=256, out_channels=self.channel, kernel_size=1)
        )
        self.project_global = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.channel)
        )
        self.prototype = nn.Linear(self.channel, K, bias=False)
        self.avgpool_temporal = nn.AdaptiveAvgPool1d(1)
        self.avgpool_spatial = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool_spatial = nn.AdaptiveMaxPool2d((1,1))
        
    def forward_grad(self, seq):
        b, n = seq.shape[:2]
        seq = seq.view(-1, *seq.shape[2:])
        feat, low, mid = self.backbone(seq)
        del seq
        low_fine = self.project_low_fine(low[:b*n])
        low = self.project_low(low)
        low = F.adaptive_avg_pool1d(low, 1)
        mid_fine = self.project_mid_fine(mid[:b*n])
        mid = self.project_mid(mid)
        mid = F.adaptive_avg_pool1d(mid, 1)
        fine = self.project_fine(feat)
        feat = self.project(feat)
        feat = F.adaptive_avg_pool1d(feat, 1)
        return feat.view(b, n, self.channel), fine.view(b, n, self.channel, 2), \
            low.view(2*b, n, self.channel), low_fine.view(b, n, self.channel, 8), \
            mid.view(2*b, n, self.channel), mid_fine.view(b, n, self.channel, 4)
    
    def forward_proto(self, seq):
        b, n = seq.shape[:2]
        seq = seq.view(-1, *seq.shape[2:])
        feat, low, mid = self.backbone(seq)
        del seq
        feat_global = self.project_global(self.avgpool_temporal(feat).flatten(1))
        feat_global = F.normalize(feat_global, dim=1)
        logit = self.prototype(feat_global)
        low_fine = self.project_low_fine(low[:b*n])
        low = self.project_low(low)
        low = F.adaptive_avg_pool1d(low, 1)
        mid_fine = self.project_mid_fine(mid[:b*n])
        mid = self.project_mid(mid)
        mid = F.adaptive_avg_pool1d(mid, 1)
        fine = self.project_fine(feat)
        feat = self.project(feat)
        feat = F.adaptive_avg_pool1d(feat, 1)
        return feat.view(b, n, self.channel), fine.view(b, n, self.channel, 2), \
            low.view(2*b, n, self.channel), low_fine.view(b, n, self.channel, 8), \
            mid.view(2*b, n, self.channel), mid_fine.view(b, n, self.channel, 4), \
            logit.view(b, n, self.cluster)[:, :-1]
    
    def forward_multi(self, seq):
        b, n = seq.shape[:2]
        seq = seq.view(-1, *seq.shape[2:])
        feat, low, mid = self.backbone(seq)
        del seq
        feat_global = self.project_global(self.avgpool_temporal(feat).flatten(1))
        feat_global = F.normalize(feat_global, dim=1)
        logit = self.prototype(feat_global)
        low_fine = self.project_low_fine(low[:b*n])
        low_graph = self.project_low_graph(low[:b*n])
        low = self.project_low(low)
        low = F.adaptive_avg_pool1d(low, 1)
        mid_fine = self.project_mid_fine(mid[:b*n])
        mid_graph = self.project_mid_graph(mid[:b*n])
        mid = self.project_mid(mid)
        mid = F.adaptive_avg_pool1d(mid, 1)
        fine = self.project_fine(feat)
        feat_graph = self.project_graph(feat)
        feat_graph = F.adaptive_avg_pool1d(feat_graph, 1)
        feat = self.project(feat)
        feat = F.adaptive_avg_pool1d(feat, 1)
        return feat.view(b, n, self.channel), fine.view(b, n, self.channel, 2), \
            low.view(2*b, n, self.channel), low_fine.view(b, n, self.channel, 8), \
            mid.view(2*b, n, self.channel), mid_fine.view(b, n, self.channel, 4), \
            logit.view(b, n, self.cluster)[:, :-1], feat_graph.view(b, n, self.channel), \
            mid_graph.view(b, n, self.channel, 4).mean(dim=-1), low_graph.view(b, n, self.channel, 8).mean(dim=-1)

    def forward(self, seq, mode):
        if mode == 'proto':
            return self.forward_proto(seq)
        elif mode == 'multi':
            return self.forward_multi(seq)
        else:
            return self.forward_grad(seq)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
