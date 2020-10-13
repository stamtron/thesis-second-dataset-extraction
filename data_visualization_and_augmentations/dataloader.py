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
#         rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
#         return iter(rand_tensor.tolist())
    
    def __len__(self):
        return len(self.indices)    
    
    
class WeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True, generator=None):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples
    
    
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
    
    
    