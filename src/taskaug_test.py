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

from TaskAug import full_policy, zero_hypergrad, hyper_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

data_root = os.path.join("../", "data/")

x_train = np.load(os.path.join(data_root, "X_train.npy"))
y_train_raw = pd.read_csv(os.path.join(data_root, "y_train.csv"), header=None)

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

BATCH_SIZE = 64

train_set = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x



def train(
        net,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        aug,
        aug_optim,
        epochs=10,
        metric=None,
        P=5,
        num_neumann_steps=1,
        aug_warmup=20
):
    net = net.to(device)
    aug = aug.to(device)
    aug_params = list(aug.parameters())

    train_losses = []

    for _ in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        last_i = 0
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            zero_hypergrad(aug_params)

            x = aug(x, y)

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

            if i >= aug_warmup and i % P == 0:
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    break

                hypg = hyper_step(net, aug, aug_params, train_loader, optimizer, val_loader, cur_lr, num_neumann_steps, criterion)
                hypg = hypg.norm().item()
                aug_optim.step()

            if i % 20 == 1:
                running_loss /= (i - last_i)
                running_acc /= (i - last_i)
                pbar.set_description(f"loss: {running_loss:.4f}, acc: {running_acc:.4f}")
                running_acc = 0.0
                running_loss = 0.0
                last_i = i

    return train_losses


net = Perceptron(1000 * 12, 5)
criterion = nn.BCELoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

aug = full_policy(num_classes=5, batch_first=True, input_len=1000)
aug_optim = torch.optim.RMSprop(aug.parameters(), lr=1e-3)

train(net, optim, criterion, train_loader, test_loader, aug, aug_optim, epochs=10, metric=MultilabelAUROC(num_labels=5))