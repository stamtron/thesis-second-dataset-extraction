import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(30,30))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated    
    
    
class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1] - seq_length
            if start > end:
                indices.append(torch.arange(end, start))
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
    def __init__(self, image_paths, seq_length, transform, length): #csv_file, 
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        
        return x, y
    
    def __len__(self):
        return self.length
    
    
    