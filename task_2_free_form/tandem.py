
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
from net.ArbitraryShape import GeneratorNet, SimulatorNet, InverseNet, TandemNet, SimulatorNet_new, InverseNet_new, SimulatorNet_new_linear
from utils import *
from tqdm import tqdm
import cv2
import pytorch_ssim
from image_process import MetaShape
from datasets import SiliconSpectrum, get_dataloaders
from torch.optim.lr_scheduler import StepLR, ExponentialLR

seed = 123
torch.manual_seed(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For forward model for free form inverse design 

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python tandem.py

def train(model, train_loader, optimizer, criterion, criterion_shape, configs):
    # x: structure ; y: CIE 

    model.inverse_model.train()
    model.forward_model.eval()

    for gap, spectrum, shape in train_loader:
        
        gap, spectrum, shape = gap.to(DEVICE), spectrum.to(DEVICE), shape.to(DEVICE)

        if configs.alpha != 0.0:
            optimizer.zero_grad()
            shape_pred, gap_pred = model.pred(spectrum)
            spectrum_pred = model.forward_model(shape_pred, gap_pred)
            loss = criterion(spectrum, spectrum_pred) - configs.alpha * criterion_shape(shape, shape_pred)
            loss.backward()
            optimizer.step()
        
        else: 
            optimizer.zero_grad()
            spectrum_pred = model(spectrum)
            loss = criterion(spectrum, spectrum_pred) 
            loss.backward()
            optimizer.step()

    model.eval()
    loss_epoch = 0

    with torch.no_grad():

        for gap, spectrum, shape in train_loader:
        
            gap, spectrum, shape = gap.to(DEVICE), spectrum.to(DEVICE), shape.to(DEVICE)

            if configs.alpha != 0.0:

                shape_pred, gap_pred = model.pred(spectrum)
                spectrum_pred = model.forward_model(shape_pred, gap_pred)
                loss = criterion(spectrum, spectrum_pred) - configs.alpha * criterion_shape(shape, shape_pred)
            
            else: 

                spectrum_pred = model(spectrum)
                loss = criterion(spectrum, spectrum_pred) 

            loss_epoch += loss*len(gap)

    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch


def evaluate(model, val_loader, test_loader, criterion, criterion_shape, forward_model, epoch, configs,  test=False):
    # x: structure ; y: CIE 

    model.eval()
    forward_model.eval()
    dataloader = test_loader if test else val_loader

    with torch.no_grad():
        loss_epoch = 0
        loss_epoch_2 = 0
        
        for gap, spectrum, shape in dataloader:

            gap, spectrum, shape = gap.to(DEVICE), spectrum.to(DEVICE), shape.to(DEVICE)

            shape_pred, gap_pred = model.pred(spectrum)
            spectrum_pred = model.forward_model(shape_pred, gap_pred)

            if configs.alpha != 0.0:
                loss = criterion(spectrum, spectrum_pred) - configs.alpha *criterion_shape(shape, shape_pred)
            else:
                loss = criterion(spectrum, spectrum_pred)

            loss_epoch += loss*len(gap)

            shape_pred, gap_pred = model.pred(spectrum)
            spectrum_pred_2 = forward_model(shape_pred, gap_pred)

            if configs.alpha != 0.0:
                loss_2 = criterion(spectrum, spectrum_pred_2) - configs.alpha *criterion_shape(shape, shape_pred)
            else:
                loss_2 = criterion(spectrum, spectrum_pred_2)

            loss_epoch_2 += loss_2*len(gap)

        loss_epoch = loss_epoch / len(dataloader.dataset)
        loss_epoch_2 = loss_epoch_2 / len(dataloader.dataset)

    return loss_epoch, loss_epoch_2


def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_all':loss_all,
            'configs':configs,
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

    tandem_path =   './models/examples/tandem_new_linear_TEM_depth_16_batch_256_lr_0.0001_Step_decay_0.5_5000_700_kernel_3_alpha_0.05_trained_Num_42.pth'

    configs = torch.load(tandem_path)['configs']

    check_configs(configs)

    train_loader, val_loader, test_loader = get_dataloaders(configs.batch_size, en=0, mode=configs.spec_mode)
    print(configs)
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    

    # For TEM modes
    training_path =  './models/examples/forward_new_linear_TEM_depth_16_batch_1024_lr_0.0005_Step_decay_0.5_5000_700_layers_[1, 1]_kernel_9_trained.pth'
    configs_train = torch.load(training_path)['configs']
    eval_path =  './models/examples/forward_new_linear_TEM_depth_12_batch_1024_lr_0.0005_Step_decay_0.5_5000_700_layers_[1, 1]_kernel_9_trained.pth'
    configs_eval = torch.load(eval_path)['configs']


    forward_model = SimulatorNet_new_linear(spec_dim=configs_train.spec_dim, d=configs_train.net_depth, thickness = configs_train.layers, k_size = configs_train.k_size, k_pad = configs_train.k_pad).to(DEVICE)
    forward_model.load_state_dict(torch.load(training_path)['model_state_dict'])

    forward_model_evaluate = SimulatorNet_new_linear(spec_dim=configs_eval.spec_dim, d=configs_eval.net_depth, k_size = configs_eval.k_size, k_pad = configs_eval.k_pad).to(DEVICE)
    forward_model_evaluate.load_state_dict(torch.load(eval_path)['model_state_dict'])


    inverse_model = InverseNet_new(spec_dim=configs.spec_dim, d=configs.net_depth, k_size = configs.k_size, k_pad = configs.k_pad).to(DEVICE)
    inverse_model.weight_init(mean=0, std=0.02)

    model = TandemNet(forward_model, inverse_model)

    optimizer = torch.optim.Adam(model.inverse_model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)

    if configs.lr_de_type == 0:
        # set up learning rate decaying type
        scheduler = StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        de_str = '_Step_decay_'
        epo_str = '_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)
    else:
        scheduler = ExponentialLR(optimizer, configs.lr_de)
        de_str = '_Exp_decay_'
        epo_str = '_'+str(configs.epochs)

    criterion = nn.MSELoss()
    criterion_shape = pytorch_ssim.SSIM(window_size=11)

    # start training 
    print('Model {}, Number of parameters {}'.format(configs.model, count_params(model)))
 
    path =  './models/trained/tandem_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_'+str(configs.lr)+de_str+str(configs.lr_de)+epo_str+'_kernel_'+str(configs.k_size)+'_alpha_'+str(configs.alpha)+'_trained_Num_'+str(seed)+'.pth'
    path_temp = './models/trained/tandem_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_'+str(configs.lr)+de_str+str(configs.lr_de)+epo_str+'_kernel_'+str(configs.k_size)+'_alpha_'+str(configs.alpha)+'_trained_Num_'+str(seed)+'_temp.pth'
     
    epochs = configs.epochs
    loss_all = np.zeros([3, configs.epochs])
    loss_val_best = 100
    early_stop = 710
    early_temp = 0
    
    for e in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion, criterion_shape, configs)
        loss_val, loss_val_2 = evaluate(model, val_loader, test_loader, criterion, criterion_shape, forward_model_evaluate, e, configs)
        loss_all[0,e] = loss_train
        loss_all[1,e] = loss_val
        loss_all[2,e] = loss_val_2

        if loss_val_best >= loss_all[2,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[2,e]
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

    parser = argparse.ArgumentParser('nn models for inverse design: tandem')
    parser.add_argument('--model', type=str, default='tandem')
    parser.add_argument('--spec_dim', type=int, default=58, help='Dimension of spectrum, 58 for TEM, 29 for TE/TM')
    parser.add_argument('--spec_mode', type=int, default=0, help='0 for TM+TM, 1 for TE, 2 for TM')
    parser.add_argument('--net_depth', type=int, default=16, help='Dimension of convolution layers')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')

    parser.add_argument('--k_size', type=int, default=3, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=1, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--alpha', type=float, default=0.05, help='coefficients for the shape loss')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for forward model')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.5, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=700, help='Decrease the learning rate after epochs')

    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    args = parser.parse_args()

    main(args)