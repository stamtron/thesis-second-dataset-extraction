from helpers_resnet import *

resnet = torchvision.models.resnet50(pretrained=True)
adaptive_pooling = AdaptiveConcatPool2d()
head = Head()
resnet.avgpool = adaptive_pooling
resnet.fc = head

os.environ['CUDA_VISIBLE_DEVICES']='1,2'

resnet = resnet.cuda()

df_train = pd.read_csv('./train-valid-splits-video/train.csv')
df_valid = pd.read_csv('./train-valid-splits-video/valid.csv')

bs = 192

train_loader = load_data(df_train, bs, 1)
valid_loader = load_data(df_valid, bs, 1)

for param in resnet.parameters():
    param.requires_grad = False
    
for param in resnet.avgpool.parameters():
    param.requires_grad = True
    
for param in resnet.fc.parameters():
    param.requires_grad = True

resnet = nn.DataParallel(resnet)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.01, weight_decay=0.001)

epochs = 10
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)

save_model_path = '/media/hdd/astamoulakatos/save-resnet/'

for epoch in trange(epochs, desc="Epochs"):  # loop over the dataset multiple times
    resnet.train()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    train_result = []
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.squeeze(dim=1)
        labels = labels.squeeze(dim=1)
        labels = labels.float()
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        preds = torch.sigmoid(outputs).data > 0.5
        preds = preds.to(torch.float32)  
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy_score(labels.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  inputs.size(0)
        running_f1 += f1_score(labels.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)
    
    train_result.append('Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(train_result)
    
    resnet.eval()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    valid_result = []
    with torch.no_grad():
        for X, y in valid_loader:
            X = X.cuda()
            X = X.squeeze(dim = 1)
            y = Variable(y.float()).cuda()
            y = y.squeeze(dim=1)
            X = X.squeeze(dim = 1)
            y = y.float()
            output = resnet(X)
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
    states = {'epoch': epoch, 'state_dict': resnet.state_dict(), 'optimizer': optimizer.state_dict() }
    torch.save(states, save_file_path)
    