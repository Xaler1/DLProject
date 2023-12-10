# %%
# torch and torchvision imports

import torch, torch.nn as nn, torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import numpy as np,  matplotlib.pyplot as plt, pandas as pd, pickle, os
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from ResnetModel import *
from transformer import *
writer = SummaryWriter()

if not 'src' in os.getcwd():
    for (root, dir, files) in os.walk(os.getcwd()):
        if 'src' in dir:
            os.chdir(os.path.join(root, 'src'))
            break

torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")




# %%
X_train = torch.from_numpy(np.transpose(np.load('./data/X_train.npz')['arr_0'], axes = (0,2,1))).float()
X_test = torch.from_numpy(np.transpose(np.load('./data/X_val.npz')['arr_0'], axes = (0,2,1))).float()
y_train = pd.read_csv('./data/Y_train.csv')[['Diag', 'Form', 'Rhythm']].to_numpy()
y_test = pd.read_csv('./data/Y_val.csv')[['Diag', 'Form', 'Rhythm']].to_numpy()

# %%
y_train = torch.from_numpy(y_train).int()
y_test = torch.from_numpy(y_test).int()

# %%
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
x = X_train[0:1]
print(x.shape)

# %% [markdown]
# ### Creating the Transformer Model

# %%
"""Transformer needs X input as (seq_len, batch_size, channels)"""

# %%
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 12, shuffle=True)

# %%
"""Test AUC metric"""
ml_auroc = MultilabelAUROC(num_labels=3, average="macro", thresholds=None)

# %%
lr_max = 0.0025/10
lr = lr_max
epochs = 150
criterion = nn.BCELoss()
model = Transformer(nhead=12, num_classes=3, drop_p=0.25, hidden_size=128, depth=3, seq_length=200).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)

for g in optimizer.param_groups:
    g['lr'] = lr


# %%
lr =lr_max

# %%
t = 0
steps_per_epoch = len(train_loader)
T_max = steps_per_epoch*epochs
T_0 = T_max/5 
learning_rates = []
train_losses = []

for epoch in range(epochs):
    for i, (signal, labels) in enumerate(train_loader):
        for idx in range(800):
            signal_section = (signal[:, :, idx:idx+200]).to(device).transpose(0,1).transpose(0,2)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(signal_section)
            loss = criterion(outputs, labels.float())
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
            
            train_AUC = ml_auroc(outputs, labels.int())
            writer.add_scalar("Train_Loss", loss, t)
            writer.add_scalar("Learning rate", lr, t)
            writer.add_scalar("Batch Train AUC", train_AUC, t)

        if i%100 == 0:
            print(f"Step: {i+1}/{len(train_loader)}  |  Train loss: {loss.item():.4f}  |  Train AUC: {train_AUC:.4f}")
           

    # model.eval()
    test_auc = 0
    with torch.no_grad():
        for i, (signal, labels) in enumerate(test_loader):
            idx = np.random.randint(0, 1000-200)
            signal = (signal[:, :, idx:idx+200]).to(device).transpose(0,1).transpose(0,2)
            labels = labels.to(device)
            outputs = model(signal)
            test_auc += ml_auroc(outputs, labels.int())
        test_auc /= len(test_loader)
    writer.add_scalar("Test AUC", test_auc, epoch)


