import pandas as pd
import numpy as np
import wfdb
import ast
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
path = os.path.join(data_root, 'ptbxl_database.csv')

sampling_rate=100

Y = pd.read_csv(str(path), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(os.path.join(data_root, "scp_statements.csv"), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Load raw signal data
X = load_raw_data(Y, sampling_rate, str(data_root))

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

np.save(os.path.join(data_root, 'X_train.npy'), X_train)
np.save(os.path.join(data_root, 'X_test.npy'), X_test)
y_train.to_csv(os.path.join(data_root, 'y_train.csv'), index=False, header=False)
y_test.to_csv(os.path.join(data_root, 'y_test.csv'), index=False, header=False)