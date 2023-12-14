# %%
import os, numpy as np, matplotlib.pyplot as plt, pickle, pandas as pd
import torch
from utils import *

# %% [markdown]
# ## Data Loading from Utils

# %%
data_folder = '/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/03_Project/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
data, raw_labels = load_dataset(data_folder, 100)

# %%
tasks = ['diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm']
task_labels = []
for task in tasks:
    task_labels.append(compute_label_aggregations(raw_labels, folder= data_folder, ctype=task))

# %%
outputfolder = '/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/03_Project/src/data/01_12-12-2023/'
experiment_name = 'Plain'
if not os.path.exists(outputfolder+experiment_name):
            os.makedirs(outputfolder+experiment_name)
            if not os.path.exists(outputfolder+experiment_name+'/results/'):
                os.makedirs(outputfolder+experiment_name+'/results/')
            if not os.path.exists(outputfolder+experiment_name+'/models/'):
                os.makedirs(outputfolder+experiment_name+'/models/')
            if not os.path.exists(outputfolder+experiment_name+'/data/'):
                os.makedirs(outputfolder+experiment_name+'/data/')

# %%
task_data, labels, Y, _ = select_data(data, task_labels[0], tasks[0], 0, outputfolder+experiment_name+'/data/')

# %%
test_fold = 10; val_fold = 9; train_fold = 8
X_test = task_data[labels.strat_fold == test_fold]
y_test = Y[labels.strat_fold == test_fold]
# 9th fold for validation (8th for now)
X_val = task_data[labels.strat_fold == val_fold]
y_val = Y[labels.strat_fold == val_fold]
# rest for training
X_train = task_data[labels.strat_fold <= train_fold]
y_train = Y[labels.strat_fold <= train_fold]

# Preprocess signal data
X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test, outputfolder+experiment_name+'/data/')
n_classes = y_train.shape[1]

# save train and test labels
y_train.dump(outputfolder + experiment_name+ '/data/y_train.npy')
y_val.dump(outputfolder + experiment_name+ '/data/y_val.npy')
y_test.dump(outputfolder + experiment_name+ '/data/y_test.npy')


# %%



