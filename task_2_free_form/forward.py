import random

import sys
import torch
from torch import nn
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import visdom
import numpy as np
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader, TensorDataset
from net.ArbitraryShape import GeneratorNet, SimulatorNet, SimulatorNet_new, SimulatorNet_small, SimulatorNet_new_linear
from utils import *
from tqdm import tqdm
import cv2
import pytorch_ssim
from image_process import MetaShape
from datasets import SiliconSpectrum, get_dataloaders
from torch.optim.lr_scheduler import StepLR, ExponentialLR
 
seed = 42
torch.manual_seed(seed)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For forward model for free form inverse design 


def train(model, train_loader, optimizer, criterion):

    model.train()
    loss_epoch = 0

    for gap, spectrum, shape in train_loader:
        
        gap, spectrum, shape = gap.to(DEVICE), spectrum.to(DEVICE), shape.to(DEVICE)

        optimizer.zero_grad()

        spectrum_pred = model(shape, gap)
        loss = criterion(spectrum, spectrum_pred)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        loss_epoch = 0
        
        for gap, spectrum, shape in train_loader:
            gap, spectrum, shape = gap.to(DEVICE), spectrum.to(DEVICE), shape.to(DEVICE)

            spectrum_pred = model(shape, gap)
            loss = criterion(spectrum, spectrum_pred)
            loss_epoch += loss*len(gap)
        
        loss_epoch = loss_epoch / len(train_loader.dataset)
    

    return loss_epoch


def evaluate(model, val_loader, test_loader, optimizer, criterion, test=False):

    model.eval()
    dataloader = test_loader if test else val_loader

    with torch.no_grad():
        loss_epoch = 0
        
        for gap, spectrum, shape in dataloader:
            gap, spectrum, shape = gap.to(DEVICE), spectrum.to(DEVICE), shape.to(DEVICE)

            spectrum_pred = model(shape, gap)
            loss = criterion(spectrum, spectrum_pred)
            loss_epoch += loss*len(gap)
        
        loss_epoch = loss_epoch / len(dataloader.dataset)

    return loss_epoch


def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_all':loss_all,
            'configs':configs,
            'seed':seed,
        }, path)

def check_configs(configs):
    # check if the input parameters are in the correct range
    if (configs.spec_mode==0) & (configs.spec_dim!=58):

        print('Wrong spectrum dimension.')
        return 0
    if (configs.spec_mode!=0) & (configs.spec_dim!=29):
        print('Wrong spectrum dimension.')
        return 0
    return 1


def main(configs):
    
    check_configs(configs)
        
    train_loader, val_loader, test_loader = get_dataloaders(configs.batch_size, en=1,mode=configs.spec_mode)

    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    model = SimulatorNet_new_linear(spec_dim=configs.spec_dim, d=configs.net_depth, thickness = configs.layers, k_size = configs.k_size, k_pad = configs.k_pad).to(DEVICE)
    
    model.weight_init(mean=0, std=0.02)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)

    if configs.lr_de_type == 0:
        scheduler = StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        de_str = '_Step'
        epo_str = '_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)

    else:
        scheduler = ExponentialLR(optimizer, configs.lr_de)
        de_str = '_Exp'
        epo_str = '_'+str(configs.epochs)

    criterion = nn.MSELoss()

    print('Model {}, Number of parameters {}'.format(configs.model, count_params(model)))
    
    # start training 


    path =  './models/trained/forward_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_'+str(configs.lr)+de_str+'_decay_'+str(configs.lr_de)+epo_str+'_layers_'+str(configs.layers)+'_kernel_'+str(configs.k_size)+'_trained.pth'
    path_temp = './models/trained/forward_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_'+str(configs.lr)+de_str+'_decay_'+str(configs.lr_de)+epo_str+'_layers_'+str(configs.layers)+'_kernel_'+str(configs.k_size)+'_trained_temp.pth'

    epochs = configs.epochs
    loss_all = np.zeros([2, configs.epochs])
    loss_val_best = 100
    early_stop = 710
    early_temp = 0

    for e in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion)
        loss_val = evaluate(model, val_loader, test_loader, optimizer, criterion)
        loss_all[0,e] = loss_train
        loss_all[1,e] = loss_val

        if loss_val_best >= loss_all[1,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[1,e]
            save_checkpoint(model, optimizer, e, loss_all, path, configs)
            early_temp  = 0
        else:
            early_temp = early_temp+1

        if early_temp>=early_stop:
            print('Reached early stopping, stopped straining.')
            break

        lr = scheduler.get_lr()[0]

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, best val loss {:.6f}, lr {:.6f}, early_stop {:d}.'.format(e, loss_train, loss_val, loss_val_best, lr, early_temp))
        if e%10==0:
            save_checkpoint(model, optimizer, e, loss_all, path_temp, configs)

        scheduler.step()

if __name__  == '__main__':

    parser = argparse.ArgumentParser('Forward model for free form spectrum prediction ')
    parser.add_argument('--model', type=str, default='forward_model_new_linear')
    parser.add_argument('--spec_dim', type=int, default=58, help='Dimension of spectrum')
    parser.add_argument('--spec_mode', type=int, default=0, help='0 for TM+TM, 1 for TE, 2 for TM')
    parser.add_argument('--net_depth', type=int, default=16, help='Output dimension of y')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    # try to make sure all models use the same epochs for comparision 

    parser.add_argument('--layers', nargs="+", type=int, default=[1,1], help='Number of layers for gap, img, spec when stack together')
    parser.add_argument('--k_size', type=int, default=3, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=1, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for forward model')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.5, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=700, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    args = parser.parse_args()

    main(args)