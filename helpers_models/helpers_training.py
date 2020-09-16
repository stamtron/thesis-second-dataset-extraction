import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets ,models , transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split
from PIL import Image
from pathlib import Path


def unfreeze(model,percent=0.25):
    l = int(np.ceil(len(model._modules.keys())* percent))
    l = list(model._modules.keys())[-l:]
    print(f"unfreezing these layer {l}",)
    for name in l:
        for params in model._modules[name].parameters():
            params.requires_grad_(True)

def check_freeze(model):
    for name ,layer in model._modules.items():
        s = []
        for l in layer.parameters():
            s.append(l.requires_grad)
        print(name ,all(s))