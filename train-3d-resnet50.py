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
    "sample_size": (576,704), #(288,352),
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

load = False
if load:
    checkpoint = torch.load('/media/raid/astamoulakatos/saved-3d-models/fifth-round-full-resolution/best-checkpoint-009epoch.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loading pretrained freezed model!')

    for param in model.module.parameters():
        param.requires_grad = False
    
    # unfreeze 50% of the model
    unfreeze(model.module , 1)

    check_freeze(model.module)
    
tensor_transform = get_tensor_transform('Kinetics', False)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
df = pd.read_csv('./small_dataset_csvs/events_with_number_of_frames_stratified.csv')
df_train = get_df(df, 20, True, False, False)
class_image_paths, end_idx = get_indices(df_train, root_dir)
train_loader = get_loader(16, 2, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, False)
df_valid = get_df(df, 20, False, True, False)
class_image_paths, end_idx = get_indices(df_valid, root_dir)
valid_loader = get_loader(16, 2, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, False)
df_test = get_df(df, 20, False, False, True)
class_image_paths, end_idx = get_indices(df_test, root_dir)
test_loader = get_loader(16, 2, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, False)

lr = 1e-2
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
torch.cuda.empty_cache()

if load:
    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr = 5e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_model_path = '/media/raid/astamoulakatos/saved-3d-models/'
#device = torch.device('cuda')
writer = SummaryWriter('runs/ResNet3D_experiment')
train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, writer, num_epochs=epochs)
writer.close()


