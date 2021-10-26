import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import scipy.io     # used to load .mat data

        
class SiliconColor(Dataset):
    
    def __init__(self, filepath, split='train', inverse=False):
        super(SiliconColor).__init__()
        temp = scipy.io.loadmat(filepath)
        self.data = np.array(list(temp.items())[3][1])
        
        x = self.data[:,:4]
        y = self.data[:,4:]
        if inverse:
            x, y = y, x
        self.data = np.hstack((x, y))
        

        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data[:6000])
        #self.scaler.fit(self.data)
        range_, min_ = self.scaler.data_range_, self.scaler.data_min_
        #print(range_, min_)
        #self.scaler.fit(self.data[:6000])
        
        if split is 'train':
            self.data = self.data[:6000]
        elif split is 'val':
            self.data = self.data[6000:7000]
        else:
            self.data = self.data[7000:]
            
        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data).float()
        if inverse:
            self.x, self.y = self.data[:, :3], self.data[:, 3:]
        else:
            self.x, self.y = self.data[:, :4], self.data[:, 4:]
            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(model, batch_=128):
    parentpath = os.getcwd()
    #datapath = parentpath + '/Meta_learning_photonics_structure/Model/data/RCWA_xyY_all.mat'
    datapath = './data/RCWA_xyY_all.mat'
    if model in ['forward_model', 'tandem_net','vae', 'inn', 'vae_new', 'vae_GSNN', 'vae_Full','vae_tandem', 'vae_hybrid']:
        train_dt = SiliconColor(datapath, 'train')
        val_dt = SiliconColor(datapath, 'val')
        test_dt = SiliconColor(datapath, 'test')

    else:
        train_dt = SiliconColor(datapath, 'train', inverse = True)
        val_dt = SiliconColor(datapath, 'val', inverse = True)
        test_dt = SiliconColor(datapath, 'test', inverse = True)
        
    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader

    

