from barbar import Bar
from torchsummaryX import summary
import pickle
from livelossplot import PlotLosses
from torch_lr_finder import LRFinder
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
from torchsummary1 import summary_old
import glob
from tqdm import trange
sys.path.append('../data_visualization_and_augmentations/')
from new_dataloader import *
from load_data_and_augmentations import *
sys.path.append('../../3D-ResNets-PyTorch/')
import model
from model import generate_model
import time
from utils import AverageMeter
from sklearn.metrics import precision_score,f1_score, accuracy_score, jaccard_score

class AdaptiveConcatPool3d(torch.nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool3d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool3d(self.output_size)
        self.mp = nn.AdaptiveMaxPool3d(self.output_size)

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
        self.headdr1 = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(4096, 512) #num_classes)
        self.headre1 = nn.ReLU(inplace=True)
        self.headbn2 = nn.BatchNorm1d(512)
        self.headdr2 = nn.Dropout(p=0.15)
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
    norm_value=255
    transform = transforms.Compose([
        transforms.Resize((576, 704)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value], std=[38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value])
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
    inputs = inputs.permute(0,2,1,3,4)
    inputs = inputs.squeeze(dim = 0)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(classes[j].numpy(), np.asarray(f)):
                title = class_names[i]
        imshow(out, title=title)
        
        
def train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=6):
    #liveloss = PlotLosses()
    model = model.to(device)
    val_loss = 100
    
    val_losses = []
    val_acc = []
    val_f1 = []
    train_losses = []
    train_acc = []
    train_f1 = []
    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0  
            running_f1 = 0.0
            #train_result = []
            for counter, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                preds = torch.sigmoid(outputs).data > 0.5
                preds = preds.to(torch.float32) 
                
                running_loss += loss.item() * inputs.size(0)
                running_acc += accuracy_score(labels.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  inputs.size(0)
                running_f1 += f1_score(labels.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  inputs.size(0)
           
                if (counter!=0) and (counter%400==0):
                    if phase == 'train':
                        result = '  Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(running_loss/(inputs.size(0)*counter),
                                                                                         running_acc/(inputs.size(0)*counter),
                                                                                         running_f1/(inputs.size(0)*counter))
                        print(result)
                    if phase == 'validation':
                        result = '  Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(running_loss/(inputs.size(0)*counter),
                                                                                         running_acc/(inputs.size(0)*counter),
                                                                                         running_f1/(inputs.size(0)*counter))
                        print(result)
                        
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)
            epoch_f1 = running_f1 / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_f1.append(epoch_f1)
            
            #prefix = ''
            if phase == 'validation':
                #prefix = 'val_'
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
                val_f1.append(epoch_f1)
                
                if epoch_loss < val_loss:
                    val_loss = epoch_loss
                    save_path = f'{save_model_path}best-checkpoint-{str(epoch).zfill(3)}epoch.pth'
                    states = {  'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'val_loss': epoch_loss,
                                'epoch': epoch,  }
                    
                    torch.save(states, save_path)
                    for path in sorted(glob.glob(f'{save_model_path}best-checkpoint-*epoch.pth'))[:-3]:
                        os.remove(path)
                
#             logs[prefix + 'log loss'] = epoch_loss.item()
#             logs[prefix + 'accuracy'] = epoch_acc.item()
#             logs[prefix + 'f1_score'] = epoch_f1.item()
            
#         liveloss.update(logs)
#         liveloss.send()
        with open("resnet_3d_val_losses.txt", "wb") as fp:   #Pickling
            pickle.dump(val_losses, fp)
        with open("resnet_3d_val_acc.txt", "wb") as fp:   #Pickling
            pickle.dump(val_acc, fp)
        with open("resnet_3d_val_f1.txt", "wb") as fp:   #Pickling
            pickle.dump(val_f1, fp)
        with open("resnet_3d_train_losses.txt", "wb") as fp:   #Pickling
            pickle.dump(train_losses, fp)
        with open("resnet_3d_train_acc.txt", "wb") as fp:   #Pickling
            pickle.dump(train_acc, fp)
        with open("resnet_3d_train_f1.txt", "wb") as fp:   #Pickling
            pickle.dump(train_f1, fp)
            
        

def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().cpu().numpy()/len(original)  

def train(train_loader, optimizer, device, criterion, epoch, model):
    model.train()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    train_result = []
    for i, (inputs, targets) in enumerate(train_loader):                                 
        inputs = inputs.to(device).to(device)      
        targets = Variable.to(device) 
        targets = targets.squeeze(dim=1)
        inputs = inputs.permute(0,2,1,3,4)
        targets = targets.float()
        outputs = model(inputs)
        loss = criterion(outputs, y)
        preds = torch.sigmoid(outputs).data > 0.5
        preds = preds.to(torch.float32)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        unning_acc += accuracy_score(targets.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  inputs.size(0)
        running_f1 += f1_score(targets.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)
    
    train_result.append('Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(train_result)
    
    
def validation(valid_loader, optimizer, device, criterion, epoch, model):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    valid_result = []
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            X = X.permute(0,2,1,3,4)
            y = y.squeeze(dim=1)
            y = y.float()
            output = model(X)
            loss = criterion(output, y)
            preds = torch.sigmoid(outputs).data > 0.5
            preds = preds.to(torch.float32)  
            running_loss += loss.item() * inputs.size(0)
            unning_acc += accuracy_score(targets.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  inputs.size(0)
            running_f1 += f1_score(targets.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)
    
    valid_result.append('Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(valid_result)

    save_file_path = os.path.join(save_model_path, 'save_{}.pth'.format(epoch))
    states = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() }
    torch.save(states, save_file_path)