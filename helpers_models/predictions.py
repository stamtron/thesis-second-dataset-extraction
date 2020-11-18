from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helpers_resnet import *

plt.rcParams['figure.figsize'] = (12,6)
font = {'family' : 'DejaVu Sans',  'weight' : 'normal',  'size'  : 24}
plt.rc('font', **font)

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


def show_batch(loader, bs):
    class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
    one_hot_classes = [[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,1,0,0],[0,1,0,0,0]]
    inputs, classes = next(iter(loader))
    #inputs = inputs.permute(0,2,1,3,4)
    #inputs = inputs.squeeze(dim = 0)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(classes[j].numpy(), np.asarray(f)):
                title = class_names[i]
        imshow(out, title=title)
    #return title


def plot_predictions_actuals(loader, bs, net, device, resnet3d = False):
    class_names = ['exp_and','exp_fs','exp','exp_fj','bur']
    one_hot_classes = [[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,1,0,0],[0,1,0,0,0]]
    X, y = next(iter(loader))
    X = X.to(device)
    y = Variable(y.float()).to(device) 
    y = y.squeeze(dim=1)
    y = y.float()
    output, _ = net(X)
    y = y.detach().cpu()
    #loss = criterion(output, y)
    preds = torch.sigmoid(output)
    preds = preds.to(torch.float32) 
    preds = preds.detach().cpu()
    if resnet3d:
        if X.ndim == 5:
            X = X.permute(0,2,1,3,4)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(X[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(y[j].numpy(), np.asarray(f)):
                title = class_names[i] + " / " + str(y[j].numpy()) + " / " + str(preds[j].numpy())
        imshow(out, title=title)
    #return title


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

