from helpers_lstm import *
import seaborn as sns
from tqdm import trange
import sklearn

df_train = pd.read_csv('./train-valid-splits-video/train.csv')
df_valid = pd.read_csv('./train-valid-splits-video/valid.csv')
train_loader = load_data(df_train, 48, 16)
valid_loader = load_data(df_valid, 48, 16)

device = torch.device('cuda')

cnn_encoder = ResCNNEncoder().to(device)
adaptive_pool = AdaptiveConcatPool2d()
cnn_encoder.resnet[8] = adaptive_pool
for param in cnn_encoder.parameters():
    param.requires_grad = False
for param in cnn_encoder.resnet[8].parameters():
    param.requires_grad = True
for param in cnn_encoder.headbn1.parameters():
    param.requires_grad = True
for param in cnn_encoder.fc1.parameters():
    param.requires_grad = True
    
rnn_decoder = DecoderRNNattention(batch_size=48).to(device)
for param in rnn_decoder.parameters():
    param.requires_grad = True
    
crnn_params, cnn_encoder, rnn_decoder = parallelize_model(cnn_encoder, rnn_decoder)

learning_rate = 0.001
#optimizer =  torch.optim.SGD(crnn_params, lr=learning_rate, momentum=0.9, weight_decay=1e-3)
epochs = 6
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate, weight_decay=1e-3)
#optimizer = torch.optim.SGD(crnn_params, lr=learning_rate, momentum=0.9, weight_decay=1e-3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=2)

save_model_path = '/media/scratch/astamoulakatos/save-model-lstm/'
for epoch in trange(epochs, desc="Epochs"):    
    cnn_encoder.train()
    rnn_decoder.train()
    running_loss = 0.0
    running_acc = 0.0  
    running_f1 = 0.0
    train_result = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # distribute data to device
        inputs, targets = inputs.to(device), targets.to(device) #.view(-1, )
        targets = targets.squeeze(dim=1)
        targets = targets.float()
        #N_count += X.size(0)
        optimizer.zero_grad()
        outputs = rnn_decoder(cnn_encoder(inputs)) 
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        #scheduler.step()
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
    
    cnn_encoder.eval()
    rnn_decoder.eval()
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
            output = rnn_decoder(cnn_encoder(X))
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
    
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch_attention_freezed{}.pth'.format(epoch)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch_attention_freezed_{}.pth'.format(epoch)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch_attention_freezed_{}.pth'.format(epoch)))      # save optimizer
    print("Epoch {} model saved!".format(epoch))
