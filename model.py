# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:49:32 2019

@author: zhang
"""

import torch
import torch.nn as nn
#import torch.nn.functional as f


class SparseAutoencoder(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inp)

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded

class net(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(net, self).__init__()
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.classifier = nn.Linear(n_hidden, 2)

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        output = torch.softmax(self.classifier(encoded),dim=1)
        return output

class SparseAutoencoder2(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(SparseAutoencoder2, self).__init__()
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inp)
        self.classifier =  nn.Linear(n_hidden, 2)
           

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        output =  torch.softmax(self.classifier(encoded),dim=1)
        return encoded, decoded, output