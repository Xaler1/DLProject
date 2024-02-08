
import torch, torch.nn as nn, torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import numpy as np,  matplotlib.pyplot as plt, pandas as pd, pickle
from ResnetModel import *
torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model_name = 'rhythm'
seq_len = 200

path  = '/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/03_Project/src/'
X_train = np.load(f'{path}{model_name}Train.npz')['X_train']
X_train = torch.from_numpy(np.transpose(X_train, (0, 2, 1))).float()
y_train = np.load(f'{path}{model_name}Train.npz')['y_train']
X_test = np.load(f'{path}{model_name}Val.npz')['X_val']
X_test = torch.from_numpy(np.transpose(X_test, (0, 2, 1))).float()
y_test = np.load(f'{path}{model_name}Val.npz')['y_val']
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
y_train[y_train>0] = 1.0
y_test[y_test>0] = 1.0

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle=True)

"""Test AUC metric"""
ml_auroc = MultilabelAUROC(num_labels=y_train.shape[1], average="macro", thresholds=None)

criterion = nn.BCELoss()
epochs = 10
model = model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=y_train.shape[1]).float()
model = model.to(device)
lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)

train_loss = []
lrs = []

for i, (signal, labels) in enumerate(train_loader):
    signal = signal.to(device)
    labels = labels.to(device)
    output = model(signal)
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    train_loss.append(loss.item())
    lrs.append(lr)
    lr *= 1.1

    for g in optimizer.param_groups:
        g['lr'] = lr 

    optimizer.step()

    if i > 200 or lr > 1:
        break

lrs = np.array(lrs)
train_loss = np.array(train_loss)

lr_max = lrs[np.where(train_loss == train_loss.min())[0]]

with open(f'{model_name}_{seq_len}lr_max.pickle', 'wb') as f:
    pickle.dump((lrs, train_loss), f)


lr_max = 0.135/10
lr = lr_max
criterion = nn.BCELoss()
epochs = 150
model = model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=y_train.shape[1]).float()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)

for g in optimizer.param_groups:
    g['lr'] = lr

t = 0
steps_per_epoch = len(train_loader)
T_max = steps_per_epoch*epochs
T_0 = T_max/5 
learning_rates = []
train_losses = []
test_losses = []

for epoch in range(epochs):
    for i, (signal, labels) in enumerate(train_loader):
        idx = np.random.randint(0, 1000-seq_len)
        signal = (signal[:,:,idx:idx+seq_len]).to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(signal)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        if t <= T_0:
            lr = 10**(-4) + (t/T_0)*lr_max  
        else: 
            lr = lr_max*np.cos((np.pi/2)*((t-T_0)/(T_max-T_0))) + 10**(-6) 

        for g in optimizer.param_groups:
            g['lr'] = lr 
        learning_rates.append(lr)
        train_losses.append(loss.item())
        optimizer.step()
        t+=1
        
        train_AUC = ml_auroc(outputs, (labels>0).int())
        

        if i%(len(train_loader)//5) == 0:
            print(f"Step: {i+1}/{len(train_loader)}  |  Train loss: {loss.item():.4f}  |  Train AUC: {train_AUC:.4f}")
           
    test_auc = 0
    with torch.no_grad():
        for i, (signal, labels) in enumerate(test_loader):
            idx = np.random.randint(0, 1000-seq_len)
            signal = (signal[:,:,idx:idx+seq_len]).to(device); labels = labels.to(device)
            outputs = model(signal)
            test_auc += ml_auroc(outputs, (labels>0).int())
        test_auc /= len(test_loader)
        test_losses.append(test_auc)
        print("___________________________________________________\n")
        print(f"Epoch:  {epoch}/{epochs}  |  Train loss: {loss.item():.4f}  |  Train AUC: {train_AUC:.4f}  |  Test AUC: {test_auc}")
        print("___________________________________________________\n")
    
    if epoch%2 ==0:
        with open(f'{model_name}_{seq_len}SeqLenModel.pth', 'wb') as f:
            pickle.dump(model, f)
        with open(f'{model_name}_{seq_len}losses.pickle', 'wb') as f:
            pickle.dump((train_losses, test_losses, learning_rates), f)