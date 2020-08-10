import sys
sys.path.append('../video-classification/ResNetCRNN/')
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions_new import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from torchsummary1 import summary
from dataloader import *
from sklearn.metrics import precision_score,f1_score, accuracy_score, jaccard_score


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
    inputs = inputs.squeeze(dim = 0)
    for j in range(bs):
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[j])
        for i, f in enumerate(one_hot_classes):
            if np.array_equal(classes[j][0].numpy(), np.asarray(f)):
                title = class_names[i]
        imshow(out, title=title)

def load_model(bs):
    save_model_path = '/media/hdd/astamoulakatos/save-model-lstm/'
    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512   # latent dim extracted by 2D CNN
    res_size = (576, 704)      # ResNet image size
    dropout_p = 0.6    # dropout probability
    # DecoderRNN architecture
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256
    # training parameters
    k = 5            # number of target category
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu") 
    # Data loading parameters
    params = {'batch_size': bs, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    # Create model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                                drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
    return cnn_encoder, rnn_decoder

def parallelize_model(cnn_encoder, rnn_decoder):
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder, device_ids=[0,1,2,3])
        rnn_decoder = nn.DataParallel(rnn_decoder, device_ids=[0,1,2,3])
        # Combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
#         crnn_params = list(cnn_encoder.resnet[8].parameters()) + list(cnn_encoder.headbn1.parameters()) + list(cnn_encoder.fc1.parameters()) + list(rnn_decoder.parameters())    
    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # Combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
    return crnn_params, cnn_encoder, rnn_decoder


def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().cpu().numpy()/len(original)


def train(cnn_encoder, rnn_decoder, device, criterion, train_loader, optimizer, epoch):
    cnn_encoder.train()
    rnn_decoder.train()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    result = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device) #.view(-1, )
        y = y.squeeze(dim=1)
        y = y.float()
        N_count += X.size(0)
        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X)) 
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        preds = torch.sigmoid(output).data > 0.5
        preds = preds.to(torch.float32)     
        #step_score = pred_acc(y, output.sigmoid())
        #scores.append(step_score) 
        running_loss += loss.item() * X.size(0)
        running_acc += jaccard_score(y.detach().cpu().numpy() , 
              (preds.cpu().detach().numpy()), 
               average="samples")  *  X.size(0)
        running_f1 += f1_score(y.detach().cpu().numpy() ,
              (preds.detach().cpu().numpy()), 
               average="samples")  *  X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)
    
    result.append('Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(result)
    return result

    
def validation(cnn_encoder, rnn_decoder, device, criterion, valid_loader, optimizer, epoch, save_model_path):
    cnn_encoder.eval()
    rnn_decoder.eval()
    result = []
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    with torch.no_grad():
        for X, y in valid_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            output = rnn_decoder(cnn_encoder(X))
            loss = criterion(output, y)
            running_loss += loss.item() * X.size(0)   
            preds = torch.sigmoid(output).data > 0.5
            preds = preds.to(torch.float32)  
            running_acc += jaccard_score(y.detach().cpu().numpy() , 
              (preds.cpu().detach().numpy()), 
               average="samples")  *  X.size(0)
            running_f1 += f1_score(y.detach().cpu().numpy() ,
              (preds.detach().cpu().numpy()), 
               average="samples")  *  X.size(0)
            print(running_acc)
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_acc / len(valid_loader.dataset)
    epoch_f1 = running_f1 / len(valid_loader.dataset)
    result.append('Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(result)
    
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch)))      # save optimizer
    print("Epoch {} model saved!".format(epoch))
    
    return result
