import torch, torch.nn as nn, torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import numpy as np, pandas as pd, pickle
from ResnetModel import *
from TransformerModel import *
torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.from_numpy(np.transpose(np.load('./X_train.npz')['arr_0'], axes = (0,2,1))).float()
X_test = torch.from_numpy(np.transpose(np.load('./X_val.npz')['arr_0'], axes = (0,2,1))).float()
y_train = pd.read_csv('./Y_train.csv')[['Diag', 'Form', 'Rhythm']].to_numpy()
y_test = pd.read_csv('./Y_val.csv')[['Diag', 'Form', 'Rhythm']].to_numpy()

y_train = torch.from_numpy(y_train).int()
y_test = torch.from_numpy(y_test).int()
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle=True)
ml_auroc = MultilabelAUROC(num_labels=3, average="macro", thresholds=None)

lr_max = 0.0003/10
lr = lr_max
epochs = 100
criterion = nn.BCELoss()
model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=3).to(device)
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
ts = []

for epoch in range(epochs):
    for i, (signal, labels) in enumerate(train_loader):
        idx = np.random.randint(0, 1000-200)
        signal_sample = (signal[:, :, idx:idx+200]).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(signal_sample)
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
        ts.append(t)
        t+=1
        
        train_AUC = ml_auroc(outputs, labels.int())

        if i%(len(train_loader)//10) == 0:
            print(f"Step: {i+1}/{len(train_loader)}  |  Train loss: {loss.item():.4f}  |  Train AUC: {train_AUC:.4f}")
           

    # model.eval()
    test_auc = 0
    with torch.no_grad():
        for i, (signal, labels) in enumerate(test_loader):
            idx = np.random.randint(0, 1000-200)
            signal = (signal[:, :, idx:idx+200]).to(device)
            labels = labels.to(device)
            outputs = model(signal)
            test_auc += ml_auroc(outputs, labels.int())
        test_auc /= len(test_loader)
        test_losses.append(test_auc)
    
    if epoch%2 ==0:
        with open('model.pth', 'wb') as f:
            pickle.dump(model, f)
        with open('losses.pickle', 'wb') as f:
            pickle.dump((train_losses, test_losses, learning_rates), f)