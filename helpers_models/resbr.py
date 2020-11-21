import numpy as np
import sklearn.metrics as met
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

import sys
import math

class ResBR(nn.Module):
    """
    Binary relevance constructor for ResNetBR
    """

    def __init__(self, params, num_classes):
        super(ResBR, self).__init__()
        self.num_classes = num_classes
        
        self.branches = nn.ModuleList([])
        for i in range(num_classes):
            self.branches.append(ResNetBR(params))

    def forward(self, s):
        
        s_br = []
        for i in range(self.num_classes):
            s_br.append(self.branches[i](s))
        
        s = torch.cat(s_br, 1)
        
        return s