import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from pathlib import Path
from PIL import ImageFilter
import numpy
from torchvision import transforms
import re
from PIL import Image
import sys
from collections import namedtuple
from torchsummary1 import summary
import glob
from dataloader import *
import math

sys.path.append('../3D-ResNets-PyTorch/')

import model

options = {
    "model_depth": 50,
    "model": 'resnet',
    "n_classes": 400,
    "n_finetune_classes": 5,
    "resnet_shortcut": 'B',
    "sample_size": (576,704),
    "sample_duration": 16,
    "pretrain_path": '../3D-ResNets-PyTorch/resnet-50-kinetics.pth',
    "no_cuda": False,
    "arch": 'resnet-50',
    "ft_begin_index": 0
}

opts = namedtuple("opts", sorted(options.keys()))

myopts2 = opts(**options)

from model import generate_model

model, parameters = generate_model(myopts2)

state_dict = torch.load('./save-model-3d/save_2.pth')['state_dict']
model.load_state_dict(state_dict)

root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]

transform = transforms.Compose([
    transforms.Resize((576, 704)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
one_hot_classes = [[0,1,1,0,0],[0,1,0,0,1],[0,1,0,0,0],[0,1,0,1,0],[1,0,0,0,0]]

df = pd.read_csv('./train-valid-splits-video/valid.csv')

bs = 16

class_image_paths = []
end_idx = []
for c, class_path in enumerate(class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            if d.path in df.videos.values:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                # Add class idx to paths
                paths = [(p, one_hot_classes[c]) for p in paths]
                class_image_paths.extend(paths)
                end_idx.extend([len(paths)])
                
end_idx = [0, *end_idx]
end_idx = torch.cumsum(torch.tensor(end_idx), 0)

seq_length = 16

sampler = MySampler(end_idx, seq_length)

dataset = MyDataset(
    image_paths=class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(sampler))

valid_loader = DataLoader(
    dataset,
    batch_size=bs,
    sampler=sampler,
    num_workers=0,
    drop_last = True
)

torch.cuda.empty_cache()

device = torch.device("cuda:0") 

model.eval()

test_loss = 0
all_y = []
all_y_pred = []
y_pred = []
all_y_true = np.empty((16,5))
all_y_pre = np.empty((16,5))

with torch.no_grad():
    for X, y in valid_loader:
        # distribute data to device
        X, y = X.to(device), y.to(device)
        X = X.permute(0,2,1,3,4)
        y = y.squeeze(dim=1)
        #y = y.type_as(output) # comment that line the first time and uncomment it after that
        y = y.float()
        output = model(X)
        loss = F.binary_cross_entropy_with_logits(output, y)
        test_loss += loss.item()   
        # sum up batch loss
        y_pred = output.sigmoid()
        all_y.extend(y)
        all_y_pred.extend(y_pred)
        # collect all y and y_pred in all batches
        y_true = np.array(y.cpu())
        y_pre = np.array(y_pred.cpu())

        all_y_true = np.append(all_y_true, y_true, axis=0)
        all_y_pre = np.append(all_y_pre, y_pre, axis=0)


import pickle

f1 = open('all_y_pre.pckl', 'wb')
pickle.dump(all_y_pre, f1)
f1.close()

f2 = open('all_y_true.pckl', 'wb')
pickle.dump(all_y_true, f2)
f2.close()
        
        