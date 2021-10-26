import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import scipy.io     # used to load .mat data
from utils import *

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python forward_new.py


class SiliconSpectrum(Dataset):
    
    def __init__(self, split='train', en=1, old=0, mode=0):
        # mode: o for using both TE and TM mode; 1 for TE mode, 2 for TM mode. 
        super(SiliconSpectrum).__init__()

        num_train = 53750
        num_val = 58750
        num_test = 63759


        if en ==0:
            gap = np.load('./data/all_width_cut_pm_0.8_exp_2.7.npy')
            spectrum = np.load('./data/all_spec_cut_pm_0.8_exp_2.7.npy')
            shape = np.load('./data/all_shape_cut_pm_0.8_exp_2.7.npy')
            
            if split == 'train':
                self.gap = torch.tensor(gap[:num_train]).float()
                self.spectrum = torch.tensor(spectrum[:num_train, :]).float()
                self.shape = torch.tensor(shape[:num_train, :, :, :]).float()
            elif split == 'val':
                self.gap = torch.tensor(gap[num_train:num_val]).float()
                self.spectrum = torch.tensor(spectrum[num_train:num_val, :]).float()
                self.shape = torch.tensor(shape[num_train:num_val, :, :, :]).float()
            else:
                self.gap = torch.tensor(gap[num_val:]).float()
                self.spectrum = torch.tensor(spectrum[num_val:, :]).float()
                self.shape = torch.tensor(shape[num_val:, :, :, :]).float()

        else:
            gap = np.load('./data/all_width_cut_pm_en_0.8_exp_2.7.npy')
            spectrum = np.load('./data/all_spec_cut_pm_en_0.8_exp_2.7.npy')
            shape = np.load('./data/all_shape_cut_pm_en_0.8_exp_2.7.npy')
            
            Num_train = [*range(0,num_train),*range(num_test,num_test+num_train,1),*range(num_test*2,num_test*2+num_train,1),*range(num_test*3,num_test*3+num_train,1)]
            Num_val = [*range(num_train,num_val,1),*range(num_test+num_train,num_test+num_val,1),*range(num_test*2+num_train,num_test*2+num_val,1),*range(num_test*3+num_train,num_test*3+num_val,1)]
            Num_test = [*range(num_val,num_test),*range(num_test+num_val,num_test*2,1),*range(num_test*2+num_val,num_test*3,1),*range(num_test*3+num_val,num_test*4,1)]

            if split == 'train':
                self.gap = torch.tensor(gap[Num_train]).float()
                self.spectrum = torch.tensor(spectrum[Num_train, :]).float()
                self.shape = torch.tensor(shape[Num_train, :, :, :]).float()
            elif split == 'val':
                self.gap = torch.tensor(gap[Num_val]).float()
                self.spectrum = torch.tensor(spectrum[Num_val, :]).float()
                self.shape = torch.tensor(shape[Num_val, :, :, :]).float()
            else:
                self.gap = torch.tensor(gap[Num_test]).float()
                self.spectrum = torch.tensor(spectrum[Num_test, :]).float()
                self.shape = torch.tensor(shape[Num_test, :, :, :]).float()

        # mode: o for using both TE and TM mode; 1 for TE mode, 2 for TM mode. 
        
        if mode!=0:
            if mode==1:
                self.spectrum = self.spectrum[:,:29]
            elif mode==2:
                self.spectrum = self.spectrum[:,29:]
            else: 
                raise NotImplmentedError


    def __len__(self):
        return len(self.gap)
    
    def __getitem__(self, idx):
        return self.gap[idx], self.spectrum[idx, :], self.shape[idx, :, :, :]


def get_dataloaders(batch_=1024, en = 1, if_irre = 0, err = 1.0, old=0, mode=0):
    
    if mode==0:
        print('Dataset for TE + TM modes.')
    elif mode==1:
        print('Dataset for TE modes.')
    elif mode==2:
        print('Dataset for TM modes. ')
    else:
        raise NotImplmentedError
    
    train_dt = SiliconSpectrum('train', en, old, mode)
    val_dt = SiliconSpectrum('val', en, old, mode)
    test_dt = SiliconSpectrum('test', en, old, mode)
        
    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)

    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader

    

