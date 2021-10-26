from utils import count_params, weights_init_normal
from datasets import SiliconColor, get_dataloaders
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_new, cVAE_GSNN, cVAE_GSNN1, cVAE_Full, cVAE_tandem, cVAE_hybrid, MLP_4, MLP_ReLu_4
from utils import evaluate_simple_inverse, evaluate_tandem_accuracy, evaluate_vae_inverse, evaluate_gan_inverse, evaluate_inn_inverse
from utils import evaluate_forward_minmax_dataset, evaluate_gan_minmax_inverse, evaluate_tandem_minmax_accuracy, evaluate_vae_GSNN_minmax_inverse, evaluate_forward_minmax


import random

import sys
import torch
from torch import nn
import numpy as np

import argparse

seed = random.randint(1,1000)
torch.manual_seed(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For forward model

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python forward.py

def train(model, train_loader, optimizer, criterion):
    # x: structure ; y: CIE 

    model.train()

    for x, y in train_loader:
        
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(x, None)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()


    model.eval()

    with torch.no_grad():
        loss_epoch = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x, None)
            loss = criterion(y_pred, y)
            loss_epoch += loss*len(x)
        
        loss_epoch = loss_epoch / len(train_loader.dataset)


    return loss_epoch


def evaluate(model, val_loader, test_loader, optimizer, criterion, test=False):
    # x: structure ; y: CIE 

    model.eval()
    dataloader = test_loader if test else val_loader

    with torch.no_grad():
        loss_epoch = 0
        
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x, None)
            loss = criterion(y_pred, y)
            loss_epoch += loss*len(x)
        
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


def main(configs):
    
    train_loader, val_loader, test_loader = get_dataloaders('forward_model', configs.batch_size)

    model = MLP(4, 3).to(DEVICE)

    # set up optimizer and criterion 

    #optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=configs.lr)


    scheduler = StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        #scheduler = ExponentialLR(optimizer, configs.lr_de)

    criterion = nn.MSELoss()

    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
    
    # start training 
    path =  './models/trained/forward_3_batch_'+str(configs.batch_size)+'_epoch_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)+'_lr_'+str(configs.lr)+'_STEP_'+ str(configs.if_lr_de)+'_lr_de_'+str(configs.lr_de)+'_trained_'+str(configs.Num)+'_.pth'
    path_temp = './models/trained/forward_3_batch_'+str(configs.batch_size)+'_epoch_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)+'_lr_'+str(configs.lr)+'_STEP_'+ str(configs.if_lr_de)+'_lr_de_'+str(configs.lr_de)+'_trained_'+str(configs.Num)+'_temp.pth'

    epochs = configs.epochs
    loss_all = np.zeros([2, configs.epochs])
    loss_val_best = 100
    
    for e in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion)
        loss_val = evaluate(model, val_loader, test_loader, optimizer, criterion)
        loss_all[0,e] = loss_train
        loss_all[1,e] = loss_val

        if loss_val_best >= loss_all[1,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[1,e]
            save_checkpoint(model, optimizer, e, loss_all, path, configs)

        if configs.if_lr_de==True:
            lr = scheduler.get_lr()[0]
        else: 
            lr = configs.lr

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, lr {:.8f}, best {:.6f}.'.format(e, loss_train, loss_val, lr, loss_val_best))
        if e%10==0:
            save_checkpoint(model, optimizer, e, loss_all, path_temp, configs)

        scheduler.step()



if __name__  == '__main__':
    parser = argparse.ArgumentParser('nn models for inverse design: cGAN')
    parser.add_argument('--model', type=str, default='forward_model')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension of x')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension of y')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for forward model')
    parser.add_argument('--if_lr_de',action='store_true', default='True', help='If decrease learning rate duing training')
    parser.add_argument('--lr_de', type=float, default=0.2, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=2000, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.2, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    parser.add_argument('--Num', type=int, default=1, help='Run multiple times and save ')
    args = parser.parse_args()

    main(args)