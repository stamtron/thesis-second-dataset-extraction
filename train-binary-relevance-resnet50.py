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

resnets = []

for i in range(5):
    resnet = torchvision.models.resnet50(pretrained=True)
    adaptive_pooling = AdaptiveConcatPool2d()
    head = HeadBR()
    resnet.avgpool = adaptive_pooling
    resnet.fc = head

    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3' #0,1

    resnet = resnet.cuda()

    for param in resnet.parameters():
        param.requires_grad = False

    for param in resnet.avgpool.parameters():
        param.requires_grad = True

    for param in resnet.fc.parameters():
        param.requires_grad = True
    
    resnet = nn.DataParallel(resnet)
    resnets.append(resnet)
    #check_freeze(resnet.module)
    
tensor_transform = get_tensor_transform('ImageNet', False)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(2)
valid_temp_transform = va.TemporalFit(size=16)

root_dir = '/media/scratch/astamoulakatos/nsea_video_jpegs/'
df = pd.read_csv('./important_csvs/more_balanced_dataset/small_stratified.csv')
###################################################################################
bs = 20
df_train = get_df(df, 20, True, False, False)
class_image_paths, end_idx, idx_label = get_indices(df_train, root_dir)
#train_loader = get_loader(1, 32, end_idx, class_image_paths, train_temp_transform, train_spat_transform, tensor_transform, False, True)
seq_length = 1
indices = []
for i in range(len(end_idx) - 1):
    start = end_idx[i]
    end = end_idx[i + 1] - seq_length - 20
    if start > end:
        pass
    else:
        indices.append(torch.arange(start, end))
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
        temp_transform = valid_temp_transform,
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

df_valid = get_df(df, 20, False, True, False)
class_image_paths, end_idx, idx_label = get_indices(df_valid, root_dir)
valid_loader, valid_dataset = get_loader(1, bs, end_idx, class_image_paths, valid_temp_transform, valid_spat_transform, tensor_transform, False, True, True, 1)

lr = 1e-2
epochs = 20
optimizer = optim.AdamW(resnets[0].parameters(), lr=lr, weight_decay=1e-2)
#pos_wei = torch.tensor([1, 1, 1.5, 1.5, 1])
#pos_wei = pos_wei.cuda()
#criterion = nn.BCEWithLogitsLoss()
#criterion = FocalLoss2d(weight=pos_wei,reduction='mean',balance_param=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.000001, patience=3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

#criterion = LSEPLoss()

dataloaders = {
    "train": train_loader,
    "validation": valid_loader
}


save_models_paths = ['/media/scratch/astamoulakatos/saved-binary-resnets/exposure/', '/media/scratch/astamoulakatos/saved-binary-resnets/burial/',
                    '/media/scratch/astamoulakatos/saved-binary-resnets/fieldjoint/', '/media/scratch/astamoulakatos/saved-binary-resnets/anode/',
                    '/media/scratch/astamoulakatos/saved-binary-resnets/freespan/']


writers  = ['runs/Binary-Resnets-exp', 'runs/Binary-Resnets-bur', 'runs/Binary-Resnets-fj', 'runs/Binary-Resnets-and', 'runs/Binary-Resnets-fs']

for l in range(5):
    if l==0:
        pos_wei = torch.tensor([0.25])
        pos_wei = pos_wei.cuda()
    else:
        pos_wei = torch.tensor([4])
        pos_wei = pos_wei.cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wei)
    writer = SummaryWriter(writers[l])
    train_models_binary_relevance(save_models_paths[l], dataloaders, device, resnets[l],
                                  criterion, optimizer, scheduler, writer, l, True, True, epochs)
    writer.close()
    del resnets[l]
    torch.cuda.empty_cache()