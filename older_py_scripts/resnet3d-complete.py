from helpers_3d import *
#import seaborn as sns
from tqdm import trange

df_train = pd.read_csv('./train-valid-splits-video/train.csv')
df_valid = pd.read_csv('./train-valid-splits-video/valid.csv')

train_loader = load_data(df_train, 18, 16)
valid_loader = load_data(df_valid, 18, 16)

options = {
    "model_depth": 50,
    "model": 'resnet',
    "n_classes": 400,
    "n_finetune_classes": 5,
    "resnet_shortcut": 'B',
    "sample_size": (576,704),
    "sample_duration": 16,
    "pretrain_path": '../3D-ResNets-PyTorch/resnet-50-kinetics.pth',
    "no_cuda": False,
    "arch": 'resnet-50',
    "ft_begin_index": 0
}

opts = namedtuple("opts", sorted(options.keys()))

myopts2 = opts(**options)

model, parameters = generate_model(myopts2)

adaptive_pooling = AdaptiveConcatPool3d()

os.environ['CUDA_VISIBLE_DEVICES']='0,2,3'
#torch.cuda.empty_cache()
device = torch.device('cuda') 
head = Head()
adaptive_pooling = adaptive_pooling.to(device)
head = head.to(device)
model.module.avgpool = adaptive_pooling
model.module.fc = head

for param in model.parameters():
    param.requires_grad = False
    
for param in model.module.avgpool.parameters():
    param.requires_grad = True
    
for param in model.module.fc.parameters():
    param.requires_grad = True

    
epochs = 4
#learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)

criterion = nn.BCEWithLogitsLoss()

save_model_path = '/media/hdd/astamoulakatos/save-model-3d/'

for epoch in trange(epochs, desc="Epochs"):   
    model.train()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    train_result = []
    for i, (inputs, targets) in enumerate(valid_loader):                                 
        inputs = inputs.to(device)  
        targets = Variable(targets.float()).to(device) 
        targets = targets.squeeze(dim=1)
        inputs = inputs.permute(0,2,1,3,4)
        targets = targets.float()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        preds = torch.sigmoid(outputs).data > 0.5
        preds = preds.to(torch.float32)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy_score(targets.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  inputs.size(0)
        running_f1 += f1_score(targets.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  inputs.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_acc / len(valid_loader.dataset)
    epoch_f1 = running_f1 / len(valid_loader.dataset)
    
    train_result.append('Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(train_result)
    
    model.eval()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    valid_result = []
    with torch.no_grad():
        for X, y in valid_loader:
            X = X.to(device)
            y = Variable(y.float()).to(device) 
            X = X.permute(0,2,1,3,4)
            y = y.squeeze(dim=1)
            y = y.float()
            output = model(X)
            loss = criterion(output, y)
            preds = torch.sigmoid(output).data > 0.5
            preds = preds.to(torch.float32)  
            running_loss += loss.item() * X.size(0)
            running_acc += accuracy_score(y.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  X.size(0)
            running_f1 += f1_score(y.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_acc / len(valid_loader.dataset)
    epoch_f1 = running_f1 / len(valid_loader.dataset)
    
    valid_result.append('Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(valid_result)

    save_file_path = os.path.join(save_model_path, 'save_freezed_{}.pth'.format(epoch))
    states = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() }
    torch.save(states, save_file_path)