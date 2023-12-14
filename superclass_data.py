import numpy as np, pandas as pd, wfdb, ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/home/ubuntu/Anirudh/dataset/'
print("Hello")
sampling_rate=100
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
scp_codes = pd.read_csv(path+'scp_statements.csv', index_col=0)
scp_codes.loc[:, ['diagnostic', 'form', 'rhythm']].fillna(0, inplace=True)
scp_codes[['diagnostic', 'form', 'rhythm']] = scp_codes[['diagnostic', 'form', 'rhythm']].apply(pd.to_numeric)
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
Y[['Diag', 'Form', 'Rhythm']] = 0
for idx in Y.index.values:
    labels = Y.loc[idx].scp_codes
    for key in labels.keys():
        if labels[key] > 0:
            Y.loc[idx, ['Diag', 'Form', 'Rhythm']] = scp_codes.loc[key][['diagnostic', 'form', 'rhythm']].values
Y.loc[:,['Diag', 'Form', 'Rhythm']].fillna(0, inplace=True)
X = load_raw_data(Y, sampling_rate, path)

Y.loc[:, ['Diag','Form', 'Rhythm']].fillna(0, inplace=True)
Y.fillna(0, inplace=True)

Y_train = Y[(Y.strat_fold != 10)&(Y.strat_fold!= 9)].reset_index()
X_train = X[np.where((Y.strat_fold != 10)&(Y.strat_fold!= 9))]

Y_val = Y[Y.strat_fold == 9].reset_index()
X_val = X[np.where(Y.strat_fold ==9)]

Y_test = Y[Y.strat_fold == 10].reset_index()
X_test = X[np.where(Y.strat_fold == 10)]

np.savez_compressed('./X_train.npz', X_train)
np.savez_compressed('./X_val.npz', X_val)
np.savez_compressed('./X_test.npz', X_test)
Y_train.to_csv('./Y_train.csv')
Y_val.to_csv('./Y_val.csv')
Y_test.to_csv('./Y_test.csv')