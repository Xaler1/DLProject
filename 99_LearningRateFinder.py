
import torch, torch.nn as nn, torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import numpy as np,  matplotlib.pyplot as plt, pandas as pd, pickle
from ResnetModel import *
torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model_name = 'form'
seq_len = 1000

path  = '/home/ubuntu/Anirudh/dataset/Datasets/'
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle=True)

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

while lr < 1.0:
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

        if lr > 1:
            break

lrs = np.array(lrs)
train_loss = np.array(train_loss)

lr_max = lrs[np.where(train_loss == train_loss.min())[0]]

with open(f'{model_name}_{seq_len}lr_max.pickle', 'wb') as f:
    pickle.dump((lrs, train_loss), f)