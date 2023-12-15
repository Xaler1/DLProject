import os, numpy as np, matplotlib.pyplot as plt, pickle, pandas as pd
import torch

path = './'
losses = []
loss_names = []
for file in os.listdir(path):
    if 'loss' in file:
        with open(path+file, 'rb') as f:
            losses.append(pickle.load(f))
        loss_names.append(file.split('.')[0])

for i in range(len(losses)):
    print(loss_names[i],len(losses[i][1]))