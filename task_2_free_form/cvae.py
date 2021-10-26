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
from net.ArbitraryShape import GeneratorNet, SimulatorNet, SimulatorNet_new, SimulatorNet_small, InverseNet_new, cVAE_GSNN, cVAE_hybrid, SimulatorNet_new_linear
from utils import *
from tqdm import tqdm
import cv2
import pytorch_ssim
from image_process import MetaShape
from datasets import SiliconSpectrum, get_dataloaders
from torch.optim.lr_scheduler import StepLR, ExponentialLR

#seed = random.randint(1,1000)
seed = 42
torch.manual_seed(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For cVAE or cVAE_hybrid model

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python cvae.py

def  train(model, train_loader, optimizer, criterion, criterion_shape, configs):

    # x: structure ; y: CIE 

    model.vae_model.train()
    model.forward_model.eval()

    loss_epoch = 0
    loss_replace = 0
    loss_vae = 0
    loss_KLD = 0
    loss_forward = 0

    for gap, spec, img in train_loader:
        
        gap, spec, img = gap.to(DEVICE), spec.to(DEVICE), img.to(DEVICE)

        optimizer.zero_grad()

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred =  model(img, gap, spec)

        replace_loss = criterion(gap_hat, gap) - configs.alpha * criterion_shape(img, img_hat)

        vae_loss =  criterion(gap_pred, gap) - configs.alpha * criterion_shape(img, img_pred)

        forward_loss = criterion(spec, spec_pred)

        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = configs.weight_replace*replace_loss + configs.weight_vae*vae_loss + configs.weight_KLD*KLD_loss + configs.weight_forward*forward_loss

        loss.backward()
        optimizer.step()

        loss_epoch += loss*len(gap)

        loss_replace += replace_loss*len(gap)
        loss_vae += vae_loss*len(gap)
        loss_KLD += KLD_loss*len(gap)
        loss_forward += forward_loss*len(gap)

    loss_epoch = loss_epoch / len(train_loader.dataset)

    loss_replace = loss_replace / len(train_loader.dataset)
    loss_vae = loss_vae / len(train_loader.dataset)
    loss_KLD = loss_KLD / len(train_loader.dataset)
    loss_forward = loss_forward / len(train_loader.dataset)


    return [loss_epoch, loss_replace, loss_vae, loss_KLD, loss_forward]


def evaluate(model, val_loader, test_loader, criterion, criterion_shape, forward_model, epoch, configs,  test=False):
 
    # x: structure ; y: CIE 

    model.eval()

    dataloader = test_loader if test else val_loader

    with torch.no_grad():

        gap, spec, img = dataloader.dataset.gap, dataloader.dataset.spectrum,dataloader.dataset.shape
        
        gap, spec, img = gap.to(DEVICE), spec.to(DEVICE), img.to(DEVICE)

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred  = model(img, gap, spec)

        spec_pred_eval = forward_model(img_pred, gap_pred)

        replace_loss = criterion(gap_hat, gap) - configs.alpha * criterion_shape(img, img_hat)

        vae_loss =  criterion(gap_pred, gap) - configs.alpha * criterion_shape(img, img_pred)

        forward_loss = criterion(spec, spec_pred)
        forward_loss_eval = criterion(spec, spec_pred_eval)

        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = configs.weight_replace*replace_loss + configs.weight_vae*vae_loss + configs.weight_KLD*KLD_loss + configs.weight_forward*forward_loss
        loss_eval = configs.weight_replace*replace_loss + configs.weight_vae*vae_loss + configs.weight_KLD*KLD_loss + configs.weight_forward*forward_loss_eval


    return [loss, loss_eval, replace_loss, vae_loss, KLD_loss, forward_loss, forward_loss_eval]


def save_checkpoint(model, optimizer, epoch, loss_all_train, loss_all_eval,  path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_all_train':loss_all_train,
            'loss_all_eval':loss_all_eval,
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
    
    vae_path =  './models/examples/cvae_new_linear_TEM_depth_16_batch_256_lr_0.0001_Step_decay_0.5_5000_700_kernel_5_alpha_0.05_trained_Num_42.pth'
    configs = torch.load(vae_path)['configs']

    configs.epoch_lr_de = 700
    configs.epochs = 5000

    check_configs(configs)

    train_loader, val_loader, test_loader = get_dataloaders(configs.batch_size, en=0, mode=configs.spec_mode)
    print(configs)
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    

    training_path =  './models/examples/forward_new_linear_TEM_depth_16_batch_1024_lr_0.0005_Step_decay_0.5_5000_700_layers_[1, 1]_kernel_9_trained.pth'
    configs_train = torch.load(training_path)['configs']
    eval_path =  './models/examples/forward_new_linear_TEM_depth_12_batch_1024_lr_0.0005_Step_decay_0.5_5000_700_layers_[1, 1]_kernel_9_trained.pth'
    configs_eval = torch.load(eval_path)['configs']


    forward_model = SimulatorNet_new_linear(spec_dim=configs_train.spec_dim, d=configs_train.net_depth, thickness = configs_train.layers, k_size = configs_train.k_size, k_pad = configs_train.k_pad).to(DEVICE)
    forward_model.load_state_dict(torch.load(training_path)['model_state_dict'])

    forward_model_evaluate = SimulatorNet_new_linear(spec_dim=configs_eval.spec_dim, d=configs_eval.net_depth, k_size = configs_eval.k_size, k_pad = configs_eval.k_pad).to(DEVICE)
    forward_model_evaluate.load_state_dict(torch.load(eval_path)['model_state_dict'])

    vae_model = cVAE_GSNN(spec_dim=configs.spec_dim, latent_dim=configs.latent_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)

    model = cVAE_hybrid(forward_model, vae_model)

    # set up optimizer and criterion 

    #optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    
    optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs.lr)

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

    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
    
    path =  './models/trained/cvae_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_'+str(configs.lr)+de_str+str(configs.lr_de)+epo_str+'_kernel_'+str(configs.k_size)+'_alpha_'+str(configs.alpha)+'_trained_Num_'+str(seed)+'.pth'
    path_temp = './models/trained/cvae_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_'+str(configs.lr)+de_str+str(configs.lr_de)+epo_str+'_kernel_'+str(configs.k_size)+'_alpha_'+str(configs.alpha)+'_trained_temp_Num_'+str(seed)+'.pth'


    epochs = configs.epochs
    loss_train = np.zeros([configs.epochs, 5])
    loss_eval = np.zeros([configs.epochs, 7])
    loss_val_best = 100
    early_stop = 710
    early_temp = 0
    

    for e in range(epochs):

        loss_train[e,:] = train(model, train_loader, optimizer, criterion, criterion_shape, configs)
        loss_eval[e,:] = evaluate(model, val_loader, test_loader, criterion, criterion_shape, forward_model_evaluate, e, configs)

        if loss_val_best >= loss_eval[e,1]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_eval[e,1]
            save_checkpoint(model, optimizer, e, loss_train, loss_eval, path, configs)
            early_temp = 0
        else:
            early_temp += 1

        if early_temp >= early_stop:
            print('Reached early stopping, stopped straining.')
            break

        scheduler.step()
        lr = scheduler.get_lr()[0]

        print('Epoch {}, train loss {:.6f}, replace loss {:.6f}, KLD {:.6f}, vae loss {:.6f}, pred loss {:.6f}, lr {:.6f}.'.format(e, loss_train[e,0], loss_train[e,1], loss_train[e,2], loss_train[e,3], loss_train[e,4], lr))
        print('Epoch {}, val loss 1 {:.6f}, val loss 2 {:.6f}, lr {:.6f}, early stop {:d}.'.format(e, loss_eval[e, 0], loss_eval[e, 1], lr, early_temp))
       
        if e%10==0:
            save_checkpoint(model, optimizer, e, loss_train, loss_eval, path_temp, configs)



if __name__  == '__main__':

    parser = argparse.ArgumentParser('nn models for inverse design: cVAE')
    parser.add_argument('--model', type=str, default='cVAE_hybrid')
    parser.add_argument('--img_size', type=int, default=64, help='Input size of image')
    
    
    parser.add_argument('--spec_dim', type=int, default=58, help='Dimension of spectrum, 58 for TEM, 29 for TE/TM')
    parser.add_argument('--spec_mode', type=int, default=0, help='0 for TM+TM, 1 for TE, 2 for TM')
    parser.add_argument('--net_depth', type=int, default=16, help='Dimension of convolution layers')

    parser.add_argument('--latent_dim', type=int, default=50, help='Dimension of latent variable')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--en', type=int, default=0, help='1 for data augmentation')
    
    parser.add_argument('--layers', type=int, default=[1,1,1], help='Number of layers for gap, img, spec when stack together')
    parser.add_argument('--k_size', type=int, default=3, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=1, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--weight_replace', type=float, default=0.1, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--weight_vae', type=float, default=1.0, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--weight_KLD', type=float, default=0.1, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--weight_forward', type=float, default=1.0, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--alpha', type=float, default=0.05, help='Factor of SSIM image loss')
    


    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for forward model')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.5, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=700, help='Decrease the learning rate after epochs, only for step decay')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    parser.add_argument('--Num', type=int, default=0, help='Running times' )
    args = parser.parse_args()

    main(args)