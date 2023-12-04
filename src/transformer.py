import torch, torch.nn as nn, torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import numpy as np,  matplotlib.pyplot as plt, pandas as pd, pickle
from torch.nn.functional import pad
from ResnetModel import *


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
    def __init__(self, emb_size=12, nhead=6, depth=6, hidden_size=128, seq_length=1000, num_classes=6):
        super(Transformer, self).__init__(d_model=emb_size, nhead=nhead, num_encoder_layers=depth, num_decoder_layers=depth, dim_feedforward=hidden_size)
        self.seq_len = seq_length
        self.pos_encoder = PositionalEncoding(seq_length, emb_size)
        self.decoder = nn.Linear(emb_size, 128)
        self.bndecoder = nn.BatchNorm1d(128*seq_length)
        self.linear1 = nn.Linear(128*seq_length, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1024, num_classes)

        
    def __forward_impl(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        """The output is (seq_len, batch_size, embedding_dim), 
        we need (batch_size, seq_len, embedding_dim)"""
        x.transpose_(0,1).transpose_(1,2)
        x = pad(x, pad = (0, self.seq_len-x.shape[-1]), mode='constant', value = 0)
        x = self.bndecoder(x.reshape(x.shape[0], -1))
        x = torch.relu(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x):
        return self.__forward_impl(x)



