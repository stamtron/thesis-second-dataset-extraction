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

tensor_transform = get_tensor_transform('Kinetics', False)
train_spat_transform = get_spatial_transform(1)
train_temp_transform = va.TemporalFit(size=16)
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/centre_Ch2/'
df = pd.read_csv('./important_csvs/more_balanced_dataset/big_stratified_new.csv')

train_loader = torch.load('train_loader.pth')
valid_loader = torch.load('valid_loader.pth')

print(len(train_loader), len(valid_loader))

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
    
check_freeze(model.module)

lr = 1e-2
epochs = 30
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
pos_wei = torch.tensor([1, 1, 1, 1, 1])
pos_wei = pos_wei.cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight = pos_wei)
criterion = FocalLoss2d(weight=pos_wei,reduction='mean',balance_param=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
torch.cuda.empty_cache()

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_model_path = '/media/scratch/astamoulakatos/saved-3d-models/'
#device = torch.device('cuda')
writer = SummaryWriter('runs/ResNet3D_final_exp_16_seq_freezed')
train_model_yo(save_model_path, dataloaders, device, model, criterion, optimizer, scheduler, writer, num_epochs=epochs)
writer.close()