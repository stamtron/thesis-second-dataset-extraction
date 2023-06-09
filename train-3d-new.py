import sys

sys.path.append('./helpers_models/')
sys.path.append('./data_visualization_and_augmentations/')
sys.path.append('../torch_videovision/')
sys.path.append('../3D-ResNets-PyTorch/')
sys.path.append('./important_csvs/')

from helpers_3d import *
from helpers_training import *
from focal_loss_2 import *
from load_data_and_augmentations import *
from imbalanced_sampler_3 import MultilabelBalancedRandomSampler

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
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
#torch.cuda.empty_cache()
device = torch.device('cuda') 
head = Head()
adaptive_pooling = adaptive_pooling.to(device)
head = head.to(device)
model.module.avgpool = adaptive_pooling
model.module.fc = head

for param in model.module.parameters():
    param.requires_grad = False
    
for param in model.module.avgpool.parameters():
    param.requires_grad = True
    
for param in model.module.fc.parameters():
    param.requires_grad = True

load = True
if load:
    checkpoint = torch.load('/media/scratch/astamoulakatos/saved-models/saved-3d-models/unfreezed-last-2/best-checkpoint-002epoch.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loading pretrained freezed model!')

#     for param in model.module.parameters():
#         param.requires_grad = True
        
#     for param in model.module.fc.parameters():
#         param.requires_grad = True
    # unfreeze 50% of the model
    unfreeze(model.module ,0.5)

    check_freeze(model)
    
check_freeze(model.module)
#model = nn.DataParallel(model)
    
tensor_transform = get_tensor_transform('Kinetics', False)
train_spat_transform = get_spatial_transform(1)
train_temp_transform = va.TemporalFit(size=16)
#train_temp_transform = get_temporal_transform(16)
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/centre_Ch2/'
df = pd.read_csv('./important_csvs/more_balanced_dataset/big_stratified_new.csv')

####################################################
bs = 20
df_train = get_df(df, 50, True, False, False)
class_image_paths, end_idx, idx_label = get_indices(df_train, root_dir)
seq_length = 50
indices, labels = get_final_indices_train(idx_label, end_idx, 'overlap', skip_frames=15, set_step=25, seq_length=seq_length, per_label=True)
indices = torch.cat(indices)
indices = indices[torch.randperm(len(indices))]
labels = []
for i in class_image_paths:
    labels.append(i[2])
labels = np.array(labels)
train_sampler = MultilabelBalancedRandomSampler(
    labels, indices, class_choice="least_sampled"
)
train_dataset = MyDataset(
        image_paths = class_image_paths,
        seq_length = seq_length,
        temp_transform = train_temp_transform,
        spat_transform = train_spat_transform,
        tensor_transform = tensor_transform,
        length = len(train_sampler),
        lstm = False,
        oned = False,
        augment = True,
        multi = 1)
train_loader = DataLoader(
        train_dataset,
        batch_size = bs,
        sampler = train_sampler,
        drop_last = True,
        num_workers = 0)
##########################################################################################

#train_loader = get_loader(16, 64, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, False)
df_valid = get_df(df, 50, False, True, False)
class_image_paths, end_idx, idx_label= get_indices(df_valid, root_dir)
indices, labels = get_final_indices_valid(idx_label, end_idx, 'overlap', skip_frames=15, set_step=25, seq_length=seq_length, per_label=True)
valid_loader, valid_dataset = get_loader_new(seq_length, bs, indices, class_image_paths, valid_temp_transform,
                                             valid_spat_transform, tensor_transform, False, False, True, 1)

df_test = get_df(df, 50, False, False, True)
class_image_paths, end_idx, idx_label = get_indices(df_test, root_dir)
indices, labels = get_final_indices_valid(idx_label, end_idx, 'overlap', skip_frames=15, set_step=25, seq_length=seq_length, per_label=True)
test_loader, test_dataset = get_loader_new(seq_length, bs, indices, class_image_paths, valid_temp_transform, 
                                            valid_spat_transform, tensor_transform, False, False, True, 1)

lr = 1e-5
epochs = 30
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
pos_wei = torch.tensor([1, 1, 1, 1, 1])
pos_wei = pos_wei.cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight = pos_wei)
criterion = FocalLoss2d(weight=pos_wei,reduction='mean',balance_param=1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
torch.cuda.empty_cache()

load = True
if load:
    epochs = 20
    lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
    

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_model_path = '/media/scratch/astamoulakatos/saved-models/saved-3d-models/'
#device = torch.device('cuda')
writer = SummaryWriter('runs/ResNet3D_16seq_5')
train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, writer, num_epochs=epochs)
writer.close()