import sys

sys.path.append('./helpers_models/')
sys.path.append('./data_visualization_and_augmentations/')
sys.path.append('../torch_videovision/')
sys.path.append('./important_csvs/')

from helpers_resnet import *
from focal_loss_2 import *
from load_data_and_augmentations import *
from imbalanced_sampler_3 import MultilabelBalancedRandomSampler
from warp_lsep import *

resnet = torchvision.models.resnet50(pretrained=True)
adaptive_pooling = AdaptiveConcatPool2d()
head = Head()
resnet.avgpool = adaptive_pooling
resnet.fc = head

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

resnet = resnet.cuda()

for param in resnet.parameters():
    param.requires_grad = False
    
for param in resnet.avgpool.parameters():
    param.requires_grad = True
    
for param in resnet.fc.parameters():
    param.requires_grad = True

resnet = nn.DataParallel(resnet)
check_freeze(resnet.module)

#summary(resnet.module, torch.zeros(2,3,576,704).cuda())

tensor_transform = get_tensor_transform('ImageNet', False)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/centre_Ch2/'
df = pd.read_csv('./important_csvs/more_balanced_dataset/big_stratified_new.csv')

####################################################
bs = 40
seq_length = 1
df_train = get_df(df, 50, True, False, False)
class_image_paths, end_idx, idx_label = get_indices(df_train, root_dir)
indices, labels = get_final_indices_2d_train(idx_label, end_idx, skip_frames=15, set_step=4, seq_length=1)
indices = torch.cat(indices)
indices = indices[torch.randperm(len(indices))]
labels = []
for i in class_image_paths:
    labels.append(i[2])
labels = np.array(labels)
train_sampler = MultilabelBalancedRandomSampler(
    labels, indices, class_choice="least_sampled"
)
dataset = MyDataset(
        image_paths = class_image_paths,
        seq_length = seq_length,
        temp_transform = train_temp_transform,
        spat_transform = train_spat_transform,
        tensor_transform = tensor_transform,
        length = len(train_sampler),
        lstm = False,
        oned = True,
        augment = True,
        multi = 1)
train_loader = DataLoader(
        dataset,
        batch_size = bs,
        sampler = train_sampler,
        drop_last = True,
        num_workers = 0)
##########################################################################################

#train_loader = get_loader(16, 64, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, False)
df_valid = get_df(df, 50, False, True, False)
class_image_paths, end_idx, idx_label= get_indices(df_valid, root_dir)
indices, labels = get_final_indices_2d_valid(idx_label, end_idx, skip_frames=15, set_step=4, seq_length=1)
valid_loader, valid_dataset = get_loader_new(seq_length, bs, indices, class_image_paths, valid_temp_transform,
                                             valid_spat_transform, tensor_transform, False, True, True, 1)

df_test = get_df(df, 50, False, False, True)
class_image_paths, end_idx, idx_label = get_indices(df_test, root_dir)
indices, labels = get_final_indices_2d_valid(idx_label, end_idx, skip_frames=15, set_step=4, seq_length=1)
test_loader, test_dataset = get_loader_new(seq_length, bs, indices, class_image_paths, valid_temp_transform, 
                                            valid_spat_transform, tensor_transform, False, True, True, 1)

torch.cuda.empty_cache()

load = True
if load:
    checkpoint = torch.load('/media/scratch/astamoulakatos/saved-resnet-models/secondnew/best-checkpoint-014epoch.pth')
    resnet.load_state_dict(checkpoint['model_state_dict'])
    print('loading pretrained freezed model!')
    
    for param in resnet.module.parameters():
        param.requires_grad = True
        
    for param in resnet.module.avgpool.parameters():
        param.requires_grad = True
    
    for param in resnet.module.fc.parameters():
        param.requires_grad = True    
        
    check_freeze(resnet.module)

lr = 1e-2
epochs = 30
optimizer = optim.AdamW(resnet.parameters(), lr=lr, weight_decay=1e-2)
pos_wei = torch.tensor([1, 1, 1.5, 3, 1])
pos_wei = pos_wei.cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight = pos_wei)
criterion = FocalLoss2d(weight=pos_wei,reduction='mean',balance_param=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

# criterion = LSEPLoss()

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

if load:
    epochs = 30
    optimizer = optim.AdamW(resnet.parameters(), lr=lr, weight_decay=1e-2)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr = 1e-05
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
    
save_model_path = '/media/scratch/astamoulakatos/saved-resnet-models/'
device = torch.device('cuda')
writer = SummaryWriter('runs/ResNet2D_newdata_third')
train_model_yo(save_model_path, dataloaders, device, resnet, criterion, optimizer, scheduler, writer, epochs)
writer.close()