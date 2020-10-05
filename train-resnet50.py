import sys

sys.path.append('./helpers_models/')
sys.path.append('./data_visualization_and_augmentations/')
sys.path.append('../torch_videovision/')
sys.path.append('./important_csvs/')

from helpers_resnet import *

resnet = torchvision.models.resnet50(pretrained=True)
adaptive_pooling = AdaptiveConcatPool2d()
head = Head()
resnet.avgpool = adaptive_pooling
resnet.fc = head

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'

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

tensor_transform = get_tensor_transform('ImageNet', True)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
df = pd.read_csv('./small_dataset_csvs/events_with_number_of_frames_stratified.csv')
df_train = get_df(df, 20, True, False, False)
class_image_paths, end_idx = get_indices(df_train, root_dir)
train_loader = get_loader(1, 270, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, True)
df_valid = get_df(df, 20, False, True, False)
class_image_paths, end_idx = get_indices(df_valid, root_dir)
valid_loader = get_loader(1, 270, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, True)
df_test = get_df(df, 20, False, False, True)
class_image_paths, end_idx = get_indices(df_test, root_dir)
test_loader = get_loader(1, 270, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, True)

torch.cuda.empty_cache()

lr = 1e-2
epochs = 10
optimizer = optim.AdamW(resnet.parameters(), lr=lr, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}

save_model_path = '/media/scratch/astamoulakatos/saved-resnet-models/'
device = torch.device('cuda')
writer = SummaryWriter('runs/ResNet2D_vol3')
train_model_yo(save_model_path, dataloaders, device, resnet, criterion, optimizer, scheduler, writer, epochs)
writer.close()