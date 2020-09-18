import os
import torch
import pandas as pd
#from skimage import io, transform
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
from torchvideotransforms.video_transforms import Resize as vidResize
from torchvideotransforms.volume_transforms import ClipToTensor

import vidaug.augmentors as va

from new_dataloader import *

# def show_batch_yo(loader, bs):
#     class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
#     one_hot_classes = [[1,0,1,0,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,0,1,0],[0,1,0,0,0]]
#     inputs, classes = next(iter(loader))
#     inputs = inputs.permute(0,2,1,3,4)
#     inputs = inputs.squeeze(dim = 0)
#     for j in range(bs):
#         # Make a grid from batch
#         out = torchvision.utils.make_grid(inputs[j])
#         for i, f in enumerate(one_hot_classes):
#             if np.array_equal(classes[j].numpy(), np.asarray(f)):
#                 title = class_names[i]
#         imshow(out, title=title)

def get_tensor_transform(finetuned_dataset, resize = False):
    if finetuned_dataset == 'ImageNet':
        video_transform_list = [
            ClipToTensor(channel_nb=3),
            vidNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if resize:
            video_transform_list.insert(0,vidResize((288,352)))
    if finetuned_dataset == 'Kinetics':
        norm_value=255
        video_transform_list = [
            ClipToTensor(channel_nb=3),
            vidNormalize(mean=[110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value], std=[38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value]),
        ]
        if resize:
            video_transform_list.insert(0,vidResize((288,352)))
    tensor_transform = vidCompose(video_transform_list)
    return tensor_transform


def get_temporal_transform():
    temp_transform = va.OneOf([
        va.TemporalBeginCrop(size=16),
        va.TemporalCenterCrop(size=16),
        va.TemporalRandomCrop(size=16),
        va.TemporalFit(size=16),
        va.Sequential([
            va.TemporalElasticTransformation(),
            va.TemporalFit(size=16),
        ]),
        va.Sequential([     
            va.InverseOrder(),
            va.TemporalFit(size=16),
        ]),
    ])
    return temp_transform


def get_spatial_transform(n):
    transform = va.SomeOf([
        va.RandomRotate(degrees=20), #andomly rotates the video with a degree randomly choosen from [-10, 10]  
        va.HorizontalFlip(),# horizontally flip the video with 100% probability
        va.ElasticTransformation(0.1,0.1),
        va.GaussianBlur(sigma=0.1),
        va.InvertColor(),
        #va.Superpixel(0.2,2),
        va.OneOf([
            va.Multiply(1.5),
            va.Multiply(0.75),
        ]),
        va.Add(10),
        va.Pepper(),
        va.PiecewiseAffineTransform(0.3,0.3,0.3),
        va.Salt(),
    ], N=n)
    return transform


def get_df(df, seq_length, train=False, valid=False, test=False):
    #df = pd.read_csv('../important_csvs/events_with_number_of_frames_stratified.csv')
    df_new = df[df.number_of_frames>=seq_length]
    if test:
        df_new = df_new[df_new.fold==0]
    if valid:
        df_new = df_new[df_new.fold==1]
    if train:
        df_new = df_new[df_new['fold'].isin([2,3,4])]
    return df_new


def get_indices(df, root_dir):
    #root_dir = '/media/raid/astamoulakatos/nsea_frame_sequences/centre_Ch2/'
    #root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
    root_dir = root_dir
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    #class_names = ['exp_fs','bur','exp','exp_and','exp_fj']
    #one_hot_classes = [[1,0,0,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,1,0],[1,0,1,0,0]]
    class_image_paths = []
    end_idx = []
    for c, class_path in enumerate(class_paths):
         for d in os.scandir(class_path):
            if d.is_dir:
                if d.path in df.event_path.values:
                    paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                    if 'bur' in class_path:
                        label = [0,1,0,0,0]
                    if 'exp' in class_path:
                        label = [1,0,0,0,0]
                    if 'exp_fj' in class_path:
                        label = [1,0,1,0,0]
                    if 'exp_fs' in class_path:
                        label = [1,0,0,0,1]
                    if 'exp_and' in class_path:
                        label = [1,0,0,1,0]
                    new_paths = [(p, label) for p in paths]
                    class_image_paths.extend(new_paths)
                    end_idx.extend([len(paths)])
                    
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    return class_image_paths, end_idx


def get_loader(seq_length, bs, end_idx, class_image_paths, temp_transform, spat_transform, tensor_transform, lstm, oned):
    sampler = MySampler(end_idx, seq_length)
    dataset = MyDataset(
        image_paths = class_image_paths,
        seq_length = seq_length,
        temp_transform = temp_transform,
        spat_transform = spat_transform,
        tensor_transform = tensor_transform,
        length = len(sampler),
        lstm = lstm,
        oned = oned)
    loader = DataLoader(
        dataset,
        batch_size = bs,
        sampler = sampler,
        drop_last = True,
        num_workers = 0)
    return loader


def show_one_batch(loader):
    # Get a batch of training data
    inputs, classes = next(iter(loader))
    inputs = inputs.permute(0,2,1,3,4)
    inputs = inputs.squeeze(dim = 0)

    # Make a grid from batch
    class_names = ['exp_fs','bur','exp','exp_and','exp_fj']
    one_hot_classes = [[1,0,0,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,1,0],[1,0,0,1,0]]
    out = torchvision.utils.make_grid(inputs)
    for i, f in enumerate(one_hot_classes):
        if np.array_equal(classes[0].numpy(), np.asarray(f)):
            title = class_names[i]
    imshow(out, title=title)
    
def show_batch(loader, bs):
    class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
    one_hot_classes = [[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,1,0,0],[0,1,0,0,0]]
    inputs, classes = next(iter(loader))
    inputs = inputs.permute(0,2,1,3,4)
    inputs = inputs.squeeze(dim = 0)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(classes[j].numpy(), np.asarray(f)):
                title = class_names[i]
        imshow(out, title=title)