import sys
sys.path.append('../data_visualization_and_augmentations/')
from new_dataloader import *
from load_data_and_augmentations import *
from torchsummaryX import summary
import pickle
from livelossplot import PlotLosses
from torch_lr_finder import LRFinder
from torch.autograd import Variable
from sklearn.metrics import precision_score,f1_score, accuracy_score, jaccard_score
from tqdm import trange
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
import glob
import pandas as pd
#from dataloader import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support


def nsea_compute_thresholds(y_true, y_pred):
#     y_pred = numpy.asarray(y[0])
#     y_true = numpy.asarray(y[1])
    precisions = dict()
    recalls = dict()
    Thresholds = dict()
    for i in range(5):
        precisions[i], recalls[i], Thresholds[i] = precision_recall_curve(y_true[:, i], y_pred[:, i])
    
    result = {}
    ###############
    ###############  FIX THE UGLY STAFF BELOW
    ###############
    classes = ['exp', 'bur', 'and', 'fj', 'fs']
    opt_id = []
    for i,event_type in enumerate(classes): 
        re = recalls[i]
        pre = precisions[i]
        dist = [ np.sqrt((1-re)**2 + (1-pre)**2) for re, pre in zip(re, pre) ]
        opt_id.append(dist.index(min(dist)))
        t = Thresholds[i]
        opt_thres = t[opt_id[i]]
        result[event_type] = opt_thres
    return result

def new_compute_metrics(y_true, y_pred, thresholds):
    th = np.array([thresholds[key] for key in thresholds])
    classes = ['exp', 'bur', 'and', 'fj', 'fs']
    ## Apply digitisation on the outputs
    for idx, event_type in enumerate(classes):
        y_pred[:,idx] = np.where(y_pred[:,idx] >= thresholds[event_type], 1, 0)
    acc = []
    for idx, event in enumerate(classes):
        acc.append(accuracy_score(y_true[:,idx], y_pred[:, idx]))
    acc = np.array(acc)
    agg_acc = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    
    cm=multilabel_confusion_matrix(y_true, y_pred)
    cmm=cm.reshape(-1,4)
    
    res_labels=pd.DataFrame({'Event': classes, 'Threshold': th, 'Exact Matching Score': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
    
    res_labels = pd.concat([res_labels, pd.DataFrame(cmm, columns=['tn', 'fp', 'fn', 'tp'])], axis=1)
    
    agg_precision, agg_recall, agg_f1, agg_support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    agg=pd.DataFrame({'Event': ['Aggregate'], 'Threshold': [np.nan], 'Exact Matching Score': agg_acc, 'Precision': agg_precision, 'Recall': agg_recall, 'F1-Score': agg_f1})
    
    
    agg = pd.concat([agg, pd.DataFrame(data=[np.nan]*4, index=['tn', 'fp', 'fn', 'tp']).T], axis=1)

    res=pd.concat([res_labels, agg]).reset_index(drop=True)
    
    return res

class AdaptiveConcatPool2d(torch.nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
class Flatten(torch.nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full=False): 
        super(Flatten, self).__init__()
        self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)


class Head(torch.nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.flatten = Flatten()
        self.headbn1 = nn.BatchNorm1d(4096)
        self.headdr1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(4096, 512) #num_classes)
        self.headre1 = nn.ReLU(inplace=True)
        self.headbn2 = nn.BatchNorm1d(512)
        self.headdr2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512,5)


    def forward(self, x):
        x = self.headbn1(x)
        x = self.fc1(x)
        x = self.headre1(x)
        x = self.headbn2(x)
        x = self.fc2(x)
        return x
    
def load_data(df, bs, seq_length):
    root_dir = '/media/hdd/astamoulakatos/nsea_video_jpegs/'
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    transform = transforms.Compose([
        transforms.Resize((576, 704)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
    one_hot_classes = [[1,0,1,0,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,0,1,0],[0,1,0,0,0]]
    bs = bs
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
    seq_length = seq_length
    sampler = MySampler(end_idx, seq_length)

    dataset = MyDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler))

    loader = DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler,
        num_workers=0,
        drop_last = True
    )
    return loader

def show_batch(loader, bs):
    class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
    one_hot_classes = [[1,0,1,0,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,0,1,0],[0,1,0,0,0]]
    inputs, classes = next(iter(loader))
    inputs = inputs.squeeze(dim = 1)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(classes[j][0].numpy(), np.asarray(f)):
                title = class_names[i]
        imshow(out, title=title)
        
# we assume that y contains a tuple of y_pred and targets
def nsea_compute_thresholds(y):
    y_pred = y[0].numpy()
    y_true = y[1].numpy()
    precisions = dict()
    recalls = dict()
    Thresholds = dict()
    for i in range(5):
        precisions[i], recalls[i], Thresholds[i] = precision_recall_curve(y_true[:, i], y_pred[:, i])

    result = {}
    ###############
    ###############  FIX THE UGLY STAFF BELOW
    ###############
    opt_id = []
    for i,event_type in enumerate(classes): 
        re = recalls[i]
        pre = precisions[i]
        dist = [ np.sqrt((1-re)**2 + (1-pre)**2) for re, pre in zip(re, pre) ]
        opt_id.append(dist.index(min(dist)))
        t = Thresholds[i]
        opt_thres = t[opt_id[i]]
        result[event_type] = opt_thres
    return result