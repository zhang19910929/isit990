# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:48:12 2019

@author: zhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#from torch.autograd import Variable
#from torchvision import datasets, transforms

#import os
import matplotlib.pyplot as plt
#import math

import mysql.connector

from model import SparseAutoencoder
from model import net
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm

class MyDataset(torch.utils.data.Dataset): 
    def __init__(self,input_data, label): 
        self.data = input_data
        self.label = label
    def __getitem__(self, index):

        line = self.data[index]
        flag = self.label[index]

        flag = np.asarray(flag)
        flag = torch.from_numpy(flag)
        flag = flag.long()
        line = np.asarray(line)
        line = torch.from_numpy(line)
        line = line.float()
        return (line, flag)
 
    def __len__(self): 
        return len(self.data)

def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def data_process(input):
    output=[]
    label = []
    for line in input:
        line = list(line[2:])
        
        if int(line[106]) >= 60:
            flag = 1
        else:
            flag = 0
        label.append(flag)

        line.pop(106)
        line.pop(105)
        line.pop(25)


        for index in range(len(line)):
            
            try:
                int(line[index])
            except:
                count = 0
                letter = line[index]
                word_len = len(letter)                
                new_letter = ''
                while count < word_len:
                    try:
                        int(letter[count])
                    except:
                        if letter[count].isalnum():
                            new_letter += str(ord(letter[count])) 
                        else:                           
                            new_letter += ''
                    else:
                        new_letter += (letter[count])                   
                    count += 1
                if new_letter == '': new_letter = '0'
                line[index] = int(new_letter)
            else:
                line[index] = int(line[index])  
        
        output.append(line)
        

    return output, label

  



# global constants
test = True
figure = False
BATCH_SIZE = 800
BETA = 3
RHO = 0.01
N_INP = 129
N_HIDDEN = 100
N_EPOCHS = 20
use_sparse = False
train_data_numbers = 2400
test_data_numbers = 800
total_numbers = train_data_numbers + test_data_numbers

rho = torch.FloatTensor([RHO for _ in range(N_HIDDEN)]).unsqueeze(0)

db_conn = mysql.connector.connect(host='localhost',port=3306,user='admin', passwd='910929',database="test")           
db_cur = db_conn.cursor()
command = 'SELECT * FROM seer LIMIT '+ str(total_numbers)
db_cur.execute(command)
seer = list(db_cur.fetchall())
db_conn.close()
        
#index = 0
train_data, train_label = data_process(seer[:train_data_numbers])
test_data, test_label = data_process(seer[train_data_numbers:]) 

train_set = MyDataset(train_data, train_label)
test_set = MyDataset(test_data, test_label)

train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False)

auto_encoder = SparseAutoencoder(N_INP, N_HIDDEN)
optimizer = optim.Adam(auto_encoder.parameters(), lr=0.001)

if figure == True:
    plt.figure(figsize=(100, 10), dpi=300)
    plt.xticks(np.linspace(0,N_INP-1,N_INP))

x = []
y = []

print("train autoencoder:")  
for epoch in range(N_EPOCHS):

    for i, (line, label) in enumerate(train_loader):
        line = F.normalize(line)
        optimizer.zero_grad()

        _, decoded= auto_encoder(line)
#        encoded, decoded,_= auto_encoder(line)
        
        criterion = nn.MSELoss()
        loss = criterion(decoded,line)
#        MSE_loss = (line - decoded) ** 2
#        MSE_loss = MSE_loss.view(1, -1).sum(1) / BATCH_SIZE
#        if use_sparse:
#            rho_hat = torch.sum(encoded, dim=0, keepdim=True)
#            sparsity_penalty = BETA * kl_divergence(rho, rho_hat)
#            loss = MSE_loss + sparsity_penalty
#        else:
#            loss = MSE_loss
            
        
        loss.backward()
        optimizer.step()
        
        if figure == True:
            if (epoch==N_EPOCHS-1):
                decoded = decoded.tolist()
                for line in decoded:

                    line_len = len(line)
                    number = 0
                    while number < line_len:
                        x.append(number)
                        y.append(line[number])
                        number += 1
                  
    
    print("Epoch: [%d], Loss: %.4f" %(epoch + 1, loss.data))


if figure == True:
    plt.scatter(x, y,s=5,alpha=0.5,marker = '.')
    plt.title('feature')
    #    plt.show()
    plt.savefig("1.png")
 
for param in auto_encoder.parameters():
    param.requires_grad = False
    
net_model = net(N_HIDDEN,25)
#net_model = net(N_INP, N_HIDDEN) 

#net_model.load_state_dict(auto_encoder.state_dict(),strict=False) 
#net_model.encoder.requires_grad=False
#net_model.encoder.bias.requires_grad=False
    
optimizer = optim.Adam(net_model.parameters(), lr=0.001)  

print("train classifier:")      
for epoch in range(N_EPOCHS):

    for i, (line, label) in enumerate(train_loader):
        line = F.normalize(line)
        optimizer.zero_grad()
        
        encoded,_= auto_encoder(line)
        decoded= net_model(encoded)
        criterion = nn.CrossEntropyLoss()
        
        loss = criterion(decoded,label)
        loss.backward()
        optimizer.step()                        
        
    print("Epoch: [%d], Loss: %.4f" %(epoch + 1, loss.data))   


#net_model.encoder.requires_grad=True
#net_model.encoder.bias.requires_grad=True
#    
#print("train all:")  
#for epoch in range(N_EPOCHS):
#
#    for i, (line, label) in enumerate(train_loader):
#        line = F.normalize(line)
#        optimizer.zero_grad()
#        
#        encoded,_= auto_encoder(line)
#        decoded= net_model(encoded)
#        criterion = nn.CrossEntropyLoss()
#        
#        loss = criterion(decoded,label)
#        loss.backward()
#        optimizer.step()                        
#        
#    print("Epoch: [%d], Loss: %.4f" %(epoch + 1, loss.data))   
#   
#     
if test == True:
    correct =0
    total =0
    with torch.no_grad():
       for data in test_loader:
           images, labels = data
           line = F.normalize(line)
           encoded,_= auto_encoder(line)
           decoded= net_model(encoded)
           _, predicted = torch.max(decoded.data, 1)
           total += labels.size(0)
#           print(predicted)
#           print(labels)
           correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test data: %d %%'% (
       100* correct / total))
    
