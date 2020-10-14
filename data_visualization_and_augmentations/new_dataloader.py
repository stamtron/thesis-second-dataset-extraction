import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import glob
import torch
from PIL import Image
import re
import torchvision

def parse_img_num(img):
    m = re.search(r"frame(\d+).png", img)
    if m:
        return f"{int(m.groups()[0]):06d}"
    print(img)
    raise Exception("Couldn't parse number")
    
    
def imshow(inp, title=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(30,30))
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=30)
    plt.pause(0.001)  # pause a bit so that plots are updated 


def show_batch(loader, bs, resnet3d=False):
    class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
    one_hot_classes = [[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,1,0,0],[0,1,0,0,0]]
    inputs, classes = next(iter(loader))
    if resnet3d:
        if inputs.ndim == 5:
            inputs = inputs.permute(0,2,1,3,4)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(classes[j].numpy(), np.asarray(f)):
                title = class_names[i]
        imshow(out, title=title)
    #return title   
    
    
class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1] - seq_length
            if start > end:
                pass
            else:
                indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)
    
    
class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, temp_transform, spat_transform, tensor_transform, length, lstm=False, oned = False): #csv_file, 
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.temp_transform = temp_transform
        self.spat_transform = spat_transform
        self.tensor_transform = tensor_transform
        self.length = length
        self.lstm = lstm
        self.oned = oned
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        #tr = self.transform
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            images.append(image)
        x = images
        if not self.oned:
            x = self.temp_transform(x)
        x = self.spat_transform(x)
        x = self.tensor_transform(x)
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        y = y.squeeze(dim=0)
        y = y.float()
        #print(y.shape)
        if self.lstm:
            x = x.permute(1,0,2,3)
        if self.oned:
            x = x.squeeze(dim=1)
        return x, y
    
    def __len__(self):
        return self.length