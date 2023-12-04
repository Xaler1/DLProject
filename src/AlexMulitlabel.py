# %%
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC
from torchsummary import summary
import os
import ast
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# %%
data_root = os.path.join("src/")

x_train = torch.from_numpy(np.transpose(np.load('src/X_train.npy'), axes = (0,2,1))).float()
x_test = torch.from_numpy(np.transpose(np.load('src/X_test.npy'), axes = (0,2,1))).float()
y_train_raw = pd.read_pickle('src/y_train.pickle').to_numpy()
y_test_raw = pd.read_pickle('src/y_test.pickle').to_numpy()

# x_train = np.load(os.path.join(data_root, "X_train.npy"))

# y_train_raw = pd.read_csv(os.path.join(data_root, "y_train.csv"), header=None)

# convert strings to corresponding arrays
y_train_raw[0] = y_train_raw[0].apply(lambda x: ast.literal_eval(x))
y_train_raw = y_train_raw[0].values

x_test = np.load(os.path.join(data_root, "X_test.npy"))
y_test_raw = pd.read_csv(os.path.join(data_root, "y_test.csv"), header=None)
y_test_raw[0] = y_test_raw[0].apply(lambda x: ast.literal_eval(x))
y_test_raw = y_test_raw[0].values

class_to_index = {
    "NORM": 0,
    "MI": 1,
    "HYP": 2,
    "STTC": 3,
    "CD": 4
}

# Encoding the labels for multi-label classification
y_test = torch.zeros((len(y_test_raw), len(class_to_index)), dtype=torch.float32)
for i, classification in enumerate(y_test_raw):
    for class_name in classification:
        y_test[i, class_to_index[class_name]] = 1

y_train = torch.zeros((len(y_train_raw), len(class_to_index)), dtype=torch.float32)
for i, classification in enumerate(y_train_raw):
    for class_name in classification:
        y_train[i, class_to_index[class_name]] = 1

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)

# Free up some memory
del y_train_raw
del y_test_raw

# %%
BATCH_SIZE = 64

train_set = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1000, emb_size=12):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-np.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Transformer):
    def __init__(self, emb_size=12, nhead=6, depth=6, hidden_size=128, seq_length=1000, num_classes=5):
        super(Transformer, self).__init__(d_model=emb_size, nhead=nhead, num_encoder_layers=depth, num_decoder_layers=depth, dim_feedforward=hidden_size)
    
        self.pos_encoder = PositionalEncoding(seq_length, emb_size)
        self.decoder = nn.Linear(emb_size, 256)
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        #x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        x = torch.relu(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
    

# %%
def train(net, optimizer, criterion, train_loader, epochs=10, scheduler=None, metric=None):
    net = net.to(device)

    train_losses = []

    for _ in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        last_i = 0
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # exact match ratio
            acc = metric(y_pred, y.int())
            #acc = accuracy_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy().round())
            running_loss += loss.item()
            running_acc += acc
            
            if i % 20 == 1:
                running_loss /= (i - last_i)
                running_acc /= (i - last_i)
                pbar.set_description(f"loss: {running_loss:.4f}, acc: {running_acc:.4f}")
                running_acc = 0.0
                running_loss = 0.0
                last_i = i
                
            
            if scheduler is not None:
                scheduler.step(loss.item())

    return train_losses

# %%
gc.collect()
torch.cuda.empty_cache()

metric = MultilabelAUROC(num_labels=len(class_to_index), average="macro", thresholds=None)
net = Transformer(nhead=6, hidden_size=512, depth=3)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True, cooldown=20, factor=0.5, min_lr=1e-6)
train(net, optimizer, criterion, train_loader, epochs=10, scheduler=scheduler, metric=metric)
print("done")

# %%


# %%



