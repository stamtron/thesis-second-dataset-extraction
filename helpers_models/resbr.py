import numpy as np
import sklearn.metrics as met
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

import sys
import math




class ResNetBR(nn.Module):
    """
    Modified from ResNet14 above, different only in last layer's output
    """
    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(ResNetBR, self).__init__()
        layers = [2,2,2,2]
        self.num_channels = 3 #params.num_channels
        self.inchannels   = 3 #params.num_channels
        
        self.conv1 = nn.Conv2d(1,self.inchannels, 3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inchannels)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(Block,  self.num_channels,layers[0], stride=2)
        self.layer2 = self._make_layer(Block,2*self.num_channels,layers[1], stride=2)
        self.layer3 = self._make_layer(Block,4*self.num_channels,layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(4,stride=1)
        self.fc1 = nn.Linear(2*128*int(np.ceil(params.width/128.))*2, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, num_layers, stride=1):
        downsample = None
        if stride != 1 or self.inchannels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inchannels, channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = []
        layers.append(block(self.inchannels, channels, stride, downsample))
        self.inchannels = channels
        for i in range(1, num_layers):
            layers.append(block(self.inchannels, channels))
        return nn.Sequential(*layers)

    def forward(self, s):
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s =  self.maxpool(s)
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = self.avgpool(s)
        s = s.view(s.size(0),-1)
        return self.fc1(s)

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