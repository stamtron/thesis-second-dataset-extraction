from helpers_weighted_loss import *
from helpers_training import *
from helpers_thresholds import *
from torch.utils.tensorboard import SummaryWriter
from barbar import Bar
from torchsummaryX import summary
import pickle
from livelossplot import PlotLosses
from torch_lr_finder import LRFinder
import seaborn as sns
from tqdm import trange
import torch.optim as optim
import sklearn
import sys
sys.path.append('../../video-classification/ResNetCRNN/')
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
from torchsummary1 import summary_old
sys.path.append('../data_visualization_and_augmentations/')
from new_dataloader import *
from load_data_and_augmentations import *
from sklearn.metrics import precision_score,f1_score, accuracy_score, jaccard_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss


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
        cnn_encoder = nn.DataParallel(cnn_encoder, device_ids=[0,1]) #1
        rnn_decoder = nn.DataParallel(rnn_decoder, device_ids=[0,1]) #1
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



def train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, writer, num_epochs=6):
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
            running_f1_micro = 0.0
            running_f1_macro = 0.0
            running_zero_one = 0.0
            running_hamming_loss = 0.0
            running_loss_bce = 0.0
            y_true = []
            y_pred = []
            #train_result = []
            for counter, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                #lab = labels
                labels = labels.to(device)
                label_smoothing = 0.03
                labels_smo = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #no_of_classes = 5
                    #beta = 0.99
                    #samples_per_cls = [46.9, 12.2, 16, 10.8, 14.1]
                    #wei = CB_weights(lab, samples_per_cls, no_of_classes, beta)
                    #wei = wei.to(device)
                    #pos_wei = torch.tensor([100/46.9, 100/12.2, 100/16, 100/10.8, 100/14.1])
                    #pos_wei = pos_wei.to(device)
                    #criterion = nn.BCEWithLogitsLoss(weight = wei, pos_weight = pos_wei)
                    loss = criterion(outputs, labels_smo)
                    pos_wei = torch.tensor([1, 1, 1.5, 0.7, 1])
                    pos_wei = pos_wei.to(device)
                    criterion2 = nn.BCEWithLogitsLoss(pos_weight = pos_wei)
                    loss_bce = criterion2(outputs, labels_smo)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    lrate = optimizer.param_groups[0]['lr']
                    lrate = np.array(lrate)

                preds = torch.sigmoid(outputs).data > 0.5
                preds = preds.to(torch.float32) 
                y = labels.detach().cpu()
                pred = preds.detach().cpu()
                y_pred.append(pred)
                y_true.append(y)
                
                running_acc += accuracy_score(y.numpy(), pred.numpy()) *  inputs.size(0)
                running_f1 += f1_score(y.numpy(), pred.numpy(), average="samples")  *  inputs.size(0)
                running_loss_bce += loss_bce.item() * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_zero_one += zero_one_loss(y.numpy(), pred.numpy()) *  inputs.size(0)
                running_hamming_loss += hamming_loss(y.numpy(), pred.numpy()) *  inputs.size(0)
                running_f1_micro += f1_score(y.numpy(), pred.numpy(), average="micro")  *  inputs.size(0)
                running_f1_macro += f1_score(y.numpy(), pred.numpy(), average="macro")  *  inputs.size(0)
                running_jac += jaccard_score(y.numpy(), pred.numpy(), average="samples")  *  inputs.size(0)
           
                if (counter!=0) and (counter%10==0):
                    if phase == 'train':
                        result = '  Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(running_loss/(inputs.size(0)*counter),
                                                                                         running_acc/(inputs.size(0)*counter),
                                                                                         running_f1/(inputs.size(0)*counter))
                        print(result)
                        classes = ['Exposure', 'Burial', 'Field Joint', 'Anode', 'Free Span']
                        y_tr = np.vstack([t.__array__() for tensor in y_true for t in tensor])
                        y_pr = np.vstack([t.__array__() for tensor in y_pred for t in tensor])
                        acc_labels, f1_labels = compute_label_metrics(y_tr, y_pr, 0.5, classes)
                        writer.add_scalar('training Exp acc', acc_labels[0], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training Bur acc', acc_labels[1], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training FJ acc', acc_labels[2], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training And acc', acc_labels[3], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training FS acc', acc_labels[4], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training Exp f1', f1_labels[0], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training Bur f1', f1_labels[1], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training FJ f1', f1_labels[2], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training And f1', f1_labels[3], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training FS f1', f1_labels[4], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training jaccard score', 
                                        running_jac/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)  
                        writer.add_scalar('training bce loss',
                                        running_loss_bce/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training focal loss',
                                        running_loss/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training acc',
                                        running_acc/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training f1 samples',
                                        running_f1/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training f1 micro',
                                        running_f1_micro/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training f1 macro',
                                        running_f1_macro/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training hamming loss',
                                        running_hamming_loss/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('training zero one loss',
                                        running_zero_one/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('learning rate', lrate, epoch * len(dataloaders[phase]) + counter)
                        y_true = []
                        y_pred = []
                        
                    if phase == 'validation':
                        result = '  Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(running_loss/(inputs.size(0)*counter),
                                                                                         running_acc/(inputs.size(0)*counter),
                                                                                         running_f1/(inputs.size(0)*counter))
                        print(result)
                        writer.add_scalar('validation bce loss',
                                        running_loss_bce/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation focal loss',
                                        running_loss/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation acc',
                                        running_acc/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation f1 samples',
                                        running_f1/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation f1 micro',
                                        running_f1_micro/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation f1 macro',
                                        running_f1_macro/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation hamming loss',
                                        running_hamming_loss/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation zero one loss',
                                        running_zero_one/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation jaccard score', 
                                        running_jac/(inputs.size(0)*counter),
                                        epoch * len(dataloaders[phase]) + counter) 
                        classes = ['Exposure', 'Burial', 'Field Joint', 'Anode', 'Free Span']
                        y_tr = np.vstack([t.__array__() for tensor in y_true for t in tensor])
                        y_pr = np.vstack([t.__array__() for tensor in y_pred for t in tensor])
                        acc_labels, f1_labels = compute_label_metrics(y_tr, y_pr, 0.5, classes)
                        writer.add_scalar('validation Exp acc', acc_labels[0], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation Bur acc', acc_labels[1], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation FJ acc', acc_labels[2], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation And acc', acc_labels[3], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation FS acc', acc_labels[4], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation Exp f1', f1_labels[0], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation Bur f1', f1_labels[1], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation FJ f1', f1_labels[2], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation And f1', f1_labels[3], epoch * len(dataloaders[phase]) + counter)
                        writer.add_scalar('validation FS f1', f1_labels[4], epoch * len(dataloaders[phase]) + counter)
                        y_true = []
                        y_pred = []
                        
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
        with open("resnet_lstm_val_losses.txt", "wb") as fp:   #Pickling
            pickle.dump(val_losses, fp)
        with open("resnet_lstm_val_acc.txt", "wb") as fp:   #Pickling
            pickle.dump(val_acc, fp)
        with open("resnet_lstm_val_f1.txt", "wb") as fp:   #Pickling
            pickle.dump(val_f1, fp)
        with open("resnet_lstm_train_losses.txt", "wb") as fp:   #Pickling
            pickle.dump(train_losses, fp)
        with open("resnet_lstm_train_acc.txt", "wb") as fp:   #Pickling
            pickle.dump(train_acc, fp)
        with open("resnet_lstm_train_f1.txt", "wb") as fp:   #Pickling
            pickle.dump(train_f1, fp)