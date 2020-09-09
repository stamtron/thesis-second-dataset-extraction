import sys

sys.path.append('./helpers_models/')
sys.path.append('./data_visualization_and_augmentations/')
sys.path.append('../torch_videovision/')
sys.path.append('./important_csvs/')
sys.path.append('../video-classification/ResNetCRNN/')

from helpers_lstm import *

tensor_transform = get_tensor_transform('ImageNet')
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

df = pd.read_csv('./important_csvs/events_with_number_of_frames_stratified_less_exp.csv')
df = get_df(df, 50, False, True, False)
class_image_paths, end_idx = get_indices(df)
train_loader = get_loader(16, 4, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, True, False)
df = pd.read_csv('./important_csvs/events_with_number_of_frames_stratified_less_exp.csv')
df = get_df(df, 50, False, False, True)
class_image_paths, end_idx = get_indices(df)
valid_loader = get_loader(16, 4, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, True, False)

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
    
rnn_decoder = DecoderRNNattention(batch_size=4).to(device)
for param in rnn_decoder.parameters():
    param.requires_grad = True
    
crnn_params, cnn_encoder, rnn_decoder = parallelize_model(cnn_encoder, rnn_decoder)

model = nn.Sequential(cnn_encoder,rnn_decoder)
torch.cuda.empty_cache()

lr = 1e-1
epochs = 6
optimizer = optim.Adam(crnn_params, lr=lr, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}
save_model_path = '/media/raid/astamoulakatos/saved-lstm-models/'

train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=epochs)
