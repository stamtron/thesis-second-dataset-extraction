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
import matplotlib.pyplot as plt
import torchvision

import sys
sys.path.append('../../torch_videovision/')

from torchvideotransforms.video_transforms import Compose as vidCompose
from torchvideotransforms.video_transforms import Normalize as vidNormalize
from torchvideotransforms.volume_transforms import ClipToTensor

import vidaug.augmentors as va

from new_dataloader import *

def get_tensor_transform(finetuned_dataset):
    if finetuned_dataset == 'ImageNet':
        video_transform_list = [
            ClipToTensor(channel_nb=3),
            vidNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    if finetuned_dataset == 'Kinetics':
        norm_value=255
        video_transform_list = [
            ClipToTensor(channel_nb=3),
            vidNormalize(mean=[110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value], std=[38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value]),
        ]
    tensor_transform = vidCompose(video_transform_list)
    return tensor_transform


def get_video_transform(n):
    transform = va.SomeOf([
        va.RandomRotate(degrees=20), #andomly rotates the video with a degree randomly choosen from [-10, 10]  
        va.HorizontalFlip(),# horizontally flip the video with 100% probability
        va.ElasticTransformation(0.1,0.1),
        va.GaussianBlur(sigma=0.1),
        va.InvertColor(),
        va.Superpixel(0.2,2),
        va.Multiply(2.0),
        va.Add(10),
        va.Pepper(),
        va.PiecewiseAffineTransform(0.3,0.3,0.3),
        va.Salt(),
        va.TemporalRandomCrop(size=16),
        va.TemporalElasticTransformation(),
        va.InverseOrder(),
    ], N=n)
    return transform


def get_df(df, seq_length, valid=False):
    #df = pd.read_csv('../important_csvs/events_with_number_of_frames_stratified.csv')
    df_new = df[df.number_of_frames>=seq_length]
    if valid:
        df_new = df_new[df_new.fold==0]
    else:
        df_new = df_new[df_new.fold!=0]
    return df_new


def get_indices(df):
    root_dir = '/media/raid/astamoulakatos/nsea_frame_sequences/centre_Ch2/'
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    class_names = ['exp_fs','bur','exp','exp_and','exp_fj']
    one_hot_classes = [[1,0,0,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,1,0],[1,0,0,1,0]]
    class_image_paths = []
    end_idx = []
    for c, class_path in enumerate(class_paths):
         for d in os.scandir(class_path):
            if d.is_dir:
                if d.path in df.event_path.values:
                    paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                    # Add class idx to paths
                    paths = [(p, one_hot_classes[c]) for p in paths]
                    class_image_paths.extend(paths)
                    end_idx.extend([len(paths)])
                    
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    return class_image_paths, end_idx


def get_loader(seq_length, bs, end_idx, class_image_paths, transform, tensor_transform):
    sampler = MySampler(end_idx, seq_length)
    dataset = MyDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        tensor_transform=tensor_transform,
        length=len(sampler))
    loader = DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler,
        drop_last=True,
        num_workers=0)
    return loader


def show_batch(loader):
    # Get a batch of training data
    inputs, classes = next(iter(loader))
    inputs = inputs.squeeze(dim = 0)
    # Make a grid from batch
    class_names = ['exp_fs','bur','exp','exp_and','exp_fj']
    one_hot_classes = [[1,0,0,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,1,0],[1,0,0,1,0]]
    out = torchvision.utils.make_grid(inputs)
    for i, f in enumerate(one_hot_classes):
        if np.array_equal(classes[0][0].numpy(), np.asarray(f)):
            title = class_names[i]
    imshow(out, title=title)