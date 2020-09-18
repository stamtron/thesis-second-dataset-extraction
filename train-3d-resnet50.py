import sys

sys.path.append('./helpers_models/')
sys.path.append('./data_visualization_and_augmentations/')
sys.path.append('../torch_videovision/')
sys.path.append('../3D-ResNets-PyTorch/')
sys.path.append('./important_csvs/')

from helpers_3d import *
from helpers_training import *

options = {
    "model_depth": 50,
    "model": 'resnet',
    "n_classes": 400,
    "n_finetune_classes": 5,
    "resnet_shortcut": 'B',
    "sample_size": (288,352), #(576,704),
    "sample_duration": 16,
    "pretrain_path": '../3D-ResNets-PyTorch/resnet-50-kinetics.pth',
    "no_cuda": False,
    "arch": 'resnet-50',
    "ft_begin_index": 0
}

opts = namedtuple("opts", sorted(options.keys()))

myopts = opts(**options)

model, parameters = generate_model(myopts)

adaptive_pooling = AdaptiveConcatPool3d()
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'
#torch.cuda.empty_cache()
device = torch.device('cuda') 
head = Head()
adaptive_pooling = adaptive_pooling.to(device)
head = head.to(device)
model.module.avgpool = adaptive_pooling
model.module.fc = head

for param in model.module.parameters():
    param.requires_grad = False
    
unfreeze(model.module ,0.3)

check_freeze(model.module)
    
tensor_transform = get_tensor_transform('Kinetics', True)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
df = pd.read_csv('./small_dataset_csvs/events_with_number_of_frames_stratified.csv')
df_train = get_df(df, 20, True, False, False)
class_image_paths, end_idx = get_indices(df_train, root_dir)
train_loader = get_loader(16, 24, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, False)
df_valid = get_df(df, 20, False, True, False)
class_image_paths, end_idx = get_indices(df_valid, root_dir)
valid_loader = get_loader(16, 24, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, False)
df_test = get_df(df, 20, False, False, True)
class_image_paths, end_idx = get_indices(df_test, root_dir)
test_loader = get_loader(16, 24, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, False)

lr = 6e-2
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
torch.cuda.empty_cache()

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_model_path = '/media/raid/astamoulakatos/saved-3d-models/'
#device = torch.device('cuda')

train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=epochs)


