import sys

sys.path.append('./helpers_models/')
sys.path.append('./data_visualization_and_augmentations/')
sys.path.append('../torch_videovision/')
sys.path.append('./important_csvs/')
sys.path.append('../video-classification/ResNetCRNN/')

from helpers_lstm import *
from helpers_training import *

tensor_transform = get_tensor_transform('ImageNet', True)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform(16)
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
df = pd.read_csv('./important_csvs/more_balanced_dataset/small_stratified.csv')

################################################################## Make function for that junk of code
df_train = get_df(df, 20, True, False, False)
class_image_paths, end_idx, idx_label = get_indices(df_train, root_dir)
seq_length = 20
indices = []
for i in range(len(end_idx) - 1):
    start = end_idx[i]
    end = end_idx[i + 1] - seq_length
    if start > end:
        pass
    else:
        indices.append(torch.arange(start, end))
indices = torch.cat(indices)
indices = indices[torch.randperm(len(indices))]
labels = []
for i in class_image_paths:
    labels.append(i[1])
labels = np.array(labels)
train_sampler = MultilabelBalancedRandomSampler(
    labels, indices, class_choice="least_sampled"
)
dataset = MyDataset(
        image_paths = class_image_paths,
        seq_length = seq_length,
        temp_transform = valid_temp_transform,
        spat_transform = valid_spat_transform,
        tensor_transform = tensor_transform,
        length = len(train_sampler),
        lstm = True,
        oned = False,
        augment = True,
        multi = 1)
train_loader = DataLoader(
        dataset,
        batch_size = 18,
        sampler = train_sampler,
        drop_last = True,
        num_workers = 0)
##########################################################################################

#train_loader = get_loader(16, 64, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, False)
df_valid = get_df(df, 20, False, True, False)
class_image_paths, end_idx, idx_label= get_indices(df_valid, root_dir)
valid_loader = get_loader(16, 18, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, True, False, True, 1)
df_test = get_df(df, 20, False, False, True)
class_image_paths, end_idx, idx_label = get_indices(df_test, root_dir)
test_loader = get_loader(16, 18, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, True, False, True, 1)

device = torch.device('cuda')
cnn_encoder = ResCNNEncoder().to(device)
adaptive_pool = AdaptiveConcatPool2d()
cnn_encoder.resnet[8] = adaptive_pool
for param in cnn_encoder.parameters():
    param.requires_grad = False
for param in cnn_encoder.resnet[8].parameters():
    param.requires_grad = True
for param in cnn_encoder.headbn1.parameters():
    param.requires_grad = True
for param in cnn_encoder.fc1.parameters():
    param.requires_grad = True
    
rnn_decoder = DecoderRNNattention(batch_size=64).to(device)
for param in rnn_decoder.parameters():
    param.requires_grad = True
    
crnn_params, cnn_encoder, rnn_decoder = parallelize_model(cnn_encoder, rnn_decoder)

model = nn.Sequential(cnn_encoder,rnn_decoder)
torch.cuda.empty_cache()

load = True
if load:
    checkpoint = torch.load('/media/scratch/astamoulakatos/saved-lstm-models/first-round-same-dataset/best-checkpoint-000epoch.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loading pretrained freezed model!')
    
    unfreeze(model[0].module ,0.6)
    unfreeze(model[0].module.resnet ,0.3)
    unfreeze(model[1].module, 1)
    
    check_freeze(model[0].module)
    check_freeze(model[0].module.resnet)
    check_freeze(model[1].module)

# check_freeze(model)

check_freeze(model[0].module)
check_freeze(model[0].module.resnet)
check_freeze(model[1].module)

lr = 1e-2
epochs = 15
optimizer = optim.AdamW(resnet.parameters(), lr=lr, weight_decay=1e-2)
pos_wei = torch.tensor([1, 1, 0.75, 1.5, 1])
pos_wei = pos_wei.cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight = pos_wei)
criterion = FocalLoss2d(weight=pos_wei,reduction='mean',balance_param=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)


if load:
    epochs = 15
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    
dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

save_model_path = '/media/scratch/astamoulakatos/saved-lstm-models/'
device = torch.device('cuda')
writer = SummaryWriter('runs/ResNet2D_LSTM_small')
train_model_yo(save_model_path, dataloaders, device, resnet, criterion, optimizer, scheduler, writer, epochs)
writer.close()
