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

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

resnet = resnet.cuda()

for param in resnet.parameters():
    param.requires_grad = True
    
for param in resnet.avgpool.parameters():
    param.requires_grad = True
    
for param in resnet.fc.parameters():
    param.requires_grad = True

#resnet = nn.DataParallel(resnet)

#summary(resnet.module, torch.zeros(2,3,576,704).cuda())

tensor_transform = get_tensor_transform('ImageNet')
train_transform = get_video_transform(2)
valid_transform = get_video_transform(0)
df = pd.read_csv('./important_csvs/events_with_number_of_frames_stratified.csv')
df = get_df(df, 16, False)
class_image_paths, end_idx = get_indices(df)
train_loader = get_loader(1, 8, end_idx, class_image_paths, train_transform, tensor_transform, False, True)
df = pd.read_csv('./important_csvs/events_with_number_of_frames_stratified.csv')
df = get_df(df, 16, True)
class_image_paths, end_idx = get_indices(df)
valid_loader = get_loader(1, 8, end_idx, class_image_paths, valid_transform, tensor_transform, False, True)

torch.cuda.empty_cache()

lr = 5e-2
optimizer = optim.Adam(resnet.parameters(), lr=lr, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=6)
dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}
save_model_path = '/media/raid/astamoulakatos/saved-resnet-models/'
device = torch.device('cuda')

train_model_yo(save_model_path, dataloaders, device, resnet, criterion, optimizer, scheduler, num_epochs=6)