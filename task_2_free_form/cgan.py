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
from net.ArbitraryShape import GeneratorNet, SimulatorNet, SimulatorNet_new, SimulatorNet_small, InverseNet_new, cVAE_GSNN, cVAE_hybrid, Discriminator, Generator, cGAN, SimulatorNet_new_linear
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

# For cGAN model



def train(model, train_loader, optimizer_G, optimizer_D, criterion, criterion_shape, configs):

    model.train()
    loss_epoch = 0
    g_loss_epoch = 0
    d_loss_epoch = 0
    g_loss_1_epoch = 0
    g_loss_2_epoch = 0

    for gap, spec, img in train_loader:

        batch_size = len(gap)
        gap, spec, img = gap.to(DEVICE), spec.to(DEVICE), img.to(DEVICE)
        
        # Ground truth

        valid = torch.ones(batch_size, 1).to(DEVICE)
        fake = torch.zeros(batch_size, 1).to(DEVICE)

        # Train the generator

        optimizer_G.zero_grad()
        z = model.sample_noise(batch_size, configs.prior).to(DEVICE)

        gen_img, gen_gap = model.Generator(spec, z)
        validity = model.Discriminator(gen_img, gen_gap, spec)

        if configs.if_struc ==0:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = 0
            g_loss = g_loss_1
            g_loss.backward()
        else:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = criterion_shape(img, gen_img)
            g_loss = g_loss_1 - configs.alpha*g_loss_2
            g_loss.backward()

        optimizer_G.step()

        # train the discriminator

        optimizer_D.zero_grad()
        # on real data

        real_pred = model.Discriminator(img, gap, spec)

        d_loss_real = criterion(real_pred, valid)

        # on generated data
        gen_img, gen_gap = model.Generator(spec, z)

        fake_pred = model.Discriminator(gen_img, gen_gap, spec)
        d_loss_fake = criterion(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake)/2
        d_loss.backward()
        optimizer_D.step()

        g_loss_epoch += g_loss * batch_size
        d_loss_epoch += d_loss * batch_size
        g_loss_1_epoch += g_loss_1 * batch_size
        g_loss_2_epoch += g_loss_2 * batch_size


    g_loss_epoch, d_loss_epoch, g_loss_1_epoch, g_loss_2_epoch = g_loss_epoch / len(train_loader.dataset), d_loss_epoch / len(train_loader.dataset), g_loss_1_epoch / len(train_loader.dataset), g_loss_2_epoch / len(train_loader.dataset),
    print('Train: generator loss {:.6f}={:.6f} - {:.2f}*{:.6f}, discriminator loss {:.6f}'.format(g_loss_epoch, g_loss_1_epoch, configs.alpha, g_loss_2_epoch, d_loss_epoch))

    return g_loss_epoch+d_loss_epoch, g_loss_epoch, d_loss_epoch, g_loss_1_epoch, g_loss_2_epoch


def evaluate(model, val_loader, test_loader, forward_model, criterion, criterion_shape, configs, test=False):


    model.eval()

    dataloader = test_loader if test else val_loader

    loss_epoch = 0
    g_loss_epoch = 0
    d_loss_epoch = 0
    g_loss_1_epoch = 0
    g_loss_2_epoch = 0

    with torch.no_grad():

        gap, spec, img = dataloader.dataset.gap, dataloader.dataset.spectrum,dataloader.dataset.shape
        
        gap, spec, img = gap.to(DEVICE), spec.to(DEVICE), img.to(DEVICE)
        batch_size = len(gap)
        # Ground truth

        valid = torch.ones(batch_size, 1).to(DEVICE)
        fake = torch.zeros(batch_size, 1).to(DEVICE)

        # loss on generator 

        z = model.sample_noise(batch_size, configs.prior).to(DEVICE)

        gen_img, gen_gap = model.Generator(spec, z)
        validity = model.Discriminator(gen_img, gen_gap, spec)

        if configs.if_struc ==0:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = 0
            g_loss = g_loss_1

        else:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = criterion_shape(img, gen_img)
            g_loss = g_loss_1 - configs.alpha*g_loss_2


        # loss on discriminator

        # on real data

        real_pred = model.Discriminator(img, gap, spec)

        d_loss_real = criterion(real_pred, valid)

        # on generated data

        fake_pred = model.Discriminator(gen_img, gen_gap, spec)
        d_loss_fake = criterion(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake)/2


        g_loss_epoch += g_loss
        d_loss_epoch += d_loss
        g_loss_1_epoch += g_loss_1
        g_loss_2_epoch += g_loss_2 

        spec_pred = forward_model(gen_img, gen_gap)
        criterion_1 = nn.MSELoss()
        loss_predict = criterion_1(spec, spec_pred)


    g_loss_epoch, d_loss_epoch, g_loss_1_epoch, g_loss_2_epoch = g_loss_epoch, d_loss_epoch, g_loss_1_epoch, g_loss_2_epoch
    print('Eval: generator loss {:.6f}={:.6f} - {:.2f}*{:.6f}, discriminator loss {:.6f}, prediction loss {:.6f}'.format(g_loss_epoch, g_loss_1_epoch, configs.alpha, g_loss_2_epoch, d_loss_epoch, loss_predict))



    return g_loss_epoch+d_loss_epoch, g_loss_epoch, d_loss_epoch, g_loss_1_epoch, g_loss_2_epoch, loss_predict


def save_checkpoint(model, optimizer_G, optimizer_D, epoch, loss_all_train, loss_all_eval, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'G_optimizer_state_dict': optimizer_G.state_dict(),
            'D_optimizer_state_dict': optimizer_D.state_dict(),
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

    gan_path =  './models/examples/cgan_new_linear_TEM_depth_16_batch_256_lr_G_0.001_lr_D_0.0001_Step_decay_0.5_5000_700_kernel_5_alpha_0.05_trained_Num_42.pth'
    configs = torch.load(gan_path)['configs']


    check_configs(configs)

    train_loader, val_loader, test_loader = get_dataloaders(configs.batch_size, en=0, mode=configs.spec_mode)
    print(configs)
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    

    # For TEM modes

    eval_path =  './models/examples/forward_new_linear_TEM_depth_12_batch_1024_lr_0.0005_Step_decay_0.5_5000_700_layers_[1, 1]_kernel_9_trained.pth'
    configs_eval = torch.load(eval_path)['configs']

    forward_model_evaluate = SimulatorNet_new_linear(spec_dim=configs_eval.spec_dim, d=configs_eval.net_depth, k_size = configs_eval.k_size, k_pad = configs_eval.k_pad).to(DEVICE)
    forward_model_evaluate.load_state_dict(torch.load(eval_path)['model_state_dict'])


    model = cGAN(img_size=64, gap_dim=1, spec_dim=configs.spec_dim, noise_dim=configs.noise_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)

    optimizer_G = torch.optim.Adam(model.Generator.parameters(), lr=configs.g_lr)
    optimizer_D = torch.optim.Adam(model.Discriminator.parameters(), lr=configs.d_lr)

    if configs.if_lr_de==0:
        # 0 for step case decay, 1 for exponential decay 
        scheduler_G = StepLR(optimizer_G, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        scheduler_D = StepLR(optimizer_D, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        de_str = '_Step_decay_'
        epo_str = '_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)
    else:
        # choose lr_de 0.9, 0.999, 1.0
        # 1 for stepcase decay, 2 for exponential decay
        scheduler_G = ExponentialLR(optimizer_G, configs.lr_de)
        scheduler_D = ExponentialLR(optimizer_D, configs.lr_de)
        de_str = '_Exp_decay_'
        epo_str = '_'+str(configs.epochs)


    criterion = torch.nn.BCELoss()
    criterion_shape = pytorch_ssim.SSIM(window_size=11)

    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))

    path =  './models/trained/cgan_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_G_'+str(configs.g_lr)+'_lr_D_'+str(configs.d_lr)+de_str+str(configs.lr_de)+epo_str+'_kernel_'+str(configs.k_size)+'_alpha_'+str(configs.alpha)+'_trained_Num_'+str(seed)+'.pth'
    path_temp = './models/trained/cgan_new_linear_TEM_depth_'+str(configs.net_depth)+'_batch_'+str(configs.batch_size)+'_lr_G_'+str(configs.g_lr)+'_lr_D_'+str(configs.d_lr)+de_str+str(configs.lr_de)+epo_str+'_kernel_'+str(configs.k_size)+'_alpha_'+str(configs.alpha)+'_trained_Num_'+str(seed)+'_temp.pth'
    

    epochs = configs.epochs
    loss_train = np.zeros([configs.epochs, 5])
    loss_eval = np.zeros([configs.epochs, 6])
    loss_val_best = 100
    early_stop = 1000
    early_temp = 0
    
    for e in range(epochs):

        loss_train[e,:] = train(model, train_loader, optimizer_G, optimizer_D, criterion, criterion_shape, configs)
        loss_eval[e,:] = evaluate(model, val_loader, test_loader, forward_model_evaluate, criterion, criterion_shape, configs)


        if loss_val_best >= loss_eval[e, 5]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_eval[e, 5]
            save_checkpoint(model, optimizer_G, optimizer_D, e, loss_train, loss_eval, path, configs)
            early_temp = 0
        else:
            early_temp +=1

        if early_temp>=early_stop:
            print('Reached early stopping, stopped straining.')
            break
        
        scheduler_D.step()
        scheduler_G.step()
        lr_D = scheduler_D.get_lr()[0]
        lr_G = scheduler_G.get_lr()[0]


        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, lr_D {:.6f}, lr_G {:.6f}, early_stop {:d}'.format(e, loss_train[e, 0], loss_eval[e, 0], lr_D, lr_G, early_temp))
        
        if e%10==0:
            save_checkpoint(model, optimizer_G, optimizer_D, e, loss_train, loss_eval, path_temp, configs)




if __name__  == '__main__':

    parser = argparse.ArgumentParser('nn models for inverse design: cGAN')
    parser.add_argument('--model', type=str, default='cGAN')

    parser.add_argument('--img_size', type=int, default=64, help='Input size of image')
    parser.add_argument('--gap_dim', type=int, default=1, help='Input dimension of gap')
    parser.add_argument('--spec_dim', type=int, default=58, help='Input dimension of spectrum')
    parser.add_argument('--spec_mode', type=int, default=0, help='0 for TEM, 1 for TE, 2 for TM')
    parser.add_argument('--noise_dim', type=int, default=50, help='Dimension of noise variable')
    parser.add_argument('--net_depth', type=int, default=16, help='Depth of neuron layers')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--en', type=int, default=0, help='1 for data augmentation')
    parser.add_argument('--layers', nargs="+", type=int, default=[1,1,1], help='Number of layers for gap, img, spec when stack together')
    parser.add_argument('--k_size', type=int, default=3, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=1, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--prior', type=int, default=1, help='1 for (0,1) normal distribution, 0 for (0,1) uniform distribution')
    
    parser.add_argument('--if_struc', type=int, default=1, help='1 adding SSIM structure loss')
    parser.add_argument('--alpha', type=float, default=0.05, help='Factor for including structure loss')
    
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Learning rate for discriminator')
    parser.add_argument('--if_lr_de',type=int, default=0, help='1 for step decay, 2 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.5, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=700, help='Decrease the learning rate after epochs')
    
    
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    args = parser.parse_args()

    main(args)