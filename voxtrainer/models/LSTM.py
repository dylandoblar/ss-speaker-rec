#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEmbedder(nn.Module):
    
    def __init__(self, n_feats, n_layers=3, hidden_size=768, out_size=256):
        super(LSTMEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(n_feats, hidden_size, num_layers=n_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x

def MainModel(nOut=256, **kwargs):
    # Number of filters
    model = LSTMEmbedder(kwargs['n_feats'], out_size=nOut)
    return model
