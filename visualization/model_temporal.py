import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from r3dtemporal import generate_model

class GreatModel(nn.Module):
    def __init__(self, network='resnet18', seq=16, sample=2, size=112, channel=128, K=100):
        super(GreatModel, self).__init__()
        self.backbone = generate_model(18)
        self.seq = seq
        self.sample = sample
        self.channel = channel
        self.head = nn.Linear(512, 2, bias=False)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, seq, training=True):
        x = self.backbone(seq)
        if training:
            x = self.avgpool(x).flatten(1)
        else:
            x = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous() #(n,thw,c)
        logit = self.head(x)
        return logit
