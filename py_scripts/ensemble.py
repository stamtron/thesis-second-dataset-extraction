from helpers_3d import *
from tqdm import trange
from helpers_lstm import *
import seaborn as sns
from tqdm import trange
import sklearn

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
head = Head()

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device = torch.device('cuda') 

adaptive_pooling = adaptive_pooling.to(device)
head = head.to(device)
model.module.avgpool = adaptive_pooling
model.module.fc = head

checkpoint = torch.load('/media/hdd/astamoulakatos/save-model-3d/save_freezed_1.pth')
model.load_state_dict(checkpoint['state_dict'])
print('loading pretrained freezed model!')

model.module.fc = nn.Sequential(*list(model.module.fc.children())[:-4])

cnn_encoder = ResCNNEncoder().to(device)
adaptive_pool = AdaptiveConcatPool2d()
cnn_encoder.resnet[8] = adaptive_pool
    
rnn_decoder = DecoderRNNattention(batch_size=16).to(device)
    
crnn_params, cnn_encoder, rnn_decoder = parallelize_model(cnn_encoder, rnn_decoder)
rnn_decoder.load_state_dict(torch.load('/media/hdd/astamoulakatos/save-model-lstm/rnn_decoder_epoch_attention_freezed_2.pth'), strict=False)
cnn_encoder.load_state_dict(torch.load('/media/hdd/astamoulakatos/save-model-lstm/cnn_encoder_epoch_attention_freezed2.pth'),strict=False)

rnn_decoder.module.fc1 = nn.Identity()

df_train = pd.read_csv('./train-valid-splits-video/train.csv')
df_valid = pd.read_csv('./train-valid-splits-video/valid.csv')

train_loader = load_data(df_train, 16, 16)
valid_loader = load_data(df_valid, 16, 16)

class MyEnsemble(nn.Module):
    def __init__(self, model, cnn_encoder, rnn_decoder, nb_classes=5):
        super(MyEnsemble, self).__init__()
        self.model = model
        self.cnn_encoder = cnn_encoder
        self.rnn_decoder = rnn_decoder

        # Create new classifier
        self.re = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(1024)
        self.dr = nn.Dropout(p=0.15)
        self.classifier = nn.Linear(512+512, nb_classes)
        
    def forward(self, x):
        x1 = x
        x2 = x
        x1 = x1.permute(0,2,1,3,4)
        x1 = self.model(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.rnn_decoder(self.cnn_encoder(x))
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.re(x)
        x = self.bn(x)
        x = self.classifier(x)
        return x

# Freeze these models
for param in model.parameters():
    param.requires_grad_(False)

for param in rnn_decoder.parameters():
    param.requires_grad_(False)

for param in cnn_encoder.parameters():
    param.requires_grad_(False)
    
ensemble_model = MyEnsemble(model, cnn_encoder, rnn_decoder)
device = torch.device('cuda') 
ensemble_model = ensemble_model.to(device)

for param in ensemble_model.parameters():
    param.requires_grad_(False)

for param in ensemble_model.bn.parameters():
    param.requires_grad_(True)

for param in ensemble_model.classifier.parameters():
    param.requires_grad_(True)
    

epochs = 5
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.01, weight_decay = 0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)

save_model_path = '/media/hdd/astamoulakatos/save-ensemble-model/'
for epoch in trange(epochs, desc="Epochs"):    
    ensemble_model.train()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    train_result = []
    #N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # distribute data to device
        inputs, targets = inputs.to(device), targets.to(device) #.view(-1, )
        targets = targets.squeeze(dim=1)
        targets = targets.float()
        #N_count += X.size(0)
        optimizer.zero_grad()
        outputs = ensemble_model(inputs) 
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        preds = torch.sigmoid(outputs).data > 0.5
        preds = preds.to(torch.float32)     
        #step_score = pred_acc(y, output.sigmoid())
        #scores.append(step_score) 
        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy_score(targets.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  inputs.size(0)
        running_f1 += f1_score(targets.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)
    
    train_result.append('Training Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(train_result)
    
    ensemble_model.eval()
    valid_result = []
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    with torch.no_grad():
        for X, y in valid_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device)
            y = y.squeeze(dim=1)
            y = y.float()
            output = ensemble_model(X)
            loss = criterion(output, y)
            running_loss += loss.item() * X.size(0)   
            preds = torch.sigmoid(output).data > 0.5
            preds = preds.to(torch.float32)  
            running_acc += accuracy_score(y.detach().cpu().numpy(), preds.cpu().detach().numpy()) *  X.size(0)
            running_f1 += f1_score(y.detach().cpu().numpy(), (preds.detach().cpu().numpy()), average="samples")  *  X.size(0)
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_acc / len(valid_loader.dataset)
    epoch_f1 = running_f1 / len(valid_loader.dataset)
    valid_result.append('Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
    print(valid_result)
    
    save_file_path = os.path.join(save_model_path, 'save_ensemble_yo_{}.pth'.format(epoch))
    states = {'epoch': epoch, 'state_dict': ensemble_model.state_dict(), 'optimizer': optimizer.state_dict() }
    torch.save(states, save_file_path)
    print('model saved!')