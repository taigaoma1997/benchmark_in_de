from utils import count_params, weights_init_normal

from datasets import SiliconColor, get_dataloaders
from torch.optim.lr_scheduler import StepLR
from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_new, cVAE_GSNN, cVAE_GSNN1, cVAE_Full, cVAE_tandem, cVAE_hybrid
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

# For cVAE or cVAE_hybrid model

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python cvae.py

def train(model, train_loader, optimizer, criterion, configs):

    # x: structure ; y: CIE 

    model.vae_model.train()
    model.forward_model.eval()

    loss_epoch = 0

    for x, y in train_loader:
        
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        recon_x, mu, logvar, x_hat,  y_pred = model(x, y)

        recon_loss = criterion(recon_x, x)
        replace_loss = criterion(x_hat, x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        pred_loss = criterion(y_pred, y)

        if configs.add_forward:
            # when added forward model as a critic on loss
            loss = recon_loss + replace_loss + KLD + configs.weight_forward*pred_loss

        else:
            loss = recon_loss + replace_loss + KLD

        loss.backward()
        optimizer.step()

        loss_epoch += loss*len(x)

    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch


def evaluate(model, val_loader, test_loader, optimizer, criterion, configs, test=False):

    # x: structure ; y: CIE 

    model.eval()

    dataloader = test_loader if test else val_loader



    with torch.no_grad():
        loss_epoch = 0
        loss_pred = 0

        
        for x, y in dataloader:

            x, y = x.to(DEVICE), y.to(DEVICE)

            recon_x, mu, logvar, x_hat,  y_pred = model(x, y)

            recon_loss = criterion(recon_x, x)
            replace_loss = criterion(x_hat, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = criterion(y_pred, y)

            if configs.add_forward:
                # when added forward model as a critic on loss
                loss = recon_loss + replace_loss + KLD + configs.weight_forward*pred_loss

            else:
                loss = recon_loss + replace_loss + KLD
            
            loss_epoch += loss*len(x)
            loss_pred += pred_loss*len(x)
        
        loss_epoch = loss_epoch / len(dataloader.dataset)
        loss_pred = loss_pred / len(dataloader.dataset)

    return loss_epoch, loss_pred


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
    
    train_loader, val_loader, test_loader = get_dataloaders('vae_hybrid', configs.batch_size)

    forward_model = MLP(4, 3).to(DEVICE)
    forward_model.load_state_dict(torch.load('./models/examples/forward_model_trained_training.pth')['model_state_dict'])
    forward_model_evaluate = MLP(configs.input_dim, configs.output_dim).to(DEVICE)
    forward_model_evaluate.load_state_dict(torch.load('./models/examples/forward_model_trained_evaluate_1.pth')['model_state_dict'])
    
    vae_model = cVAE_GSNN1(configs.input_dim, configs.latent_dim).to(DEVICE)
    model = cVAE_hybrid(forward_model, vae_model)

    # set up optimizer and criterion 

    #optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs.lr)
    

    scheduler = StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)

    criterion = nn.MSELoss()

    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
    
    # start training 
    path =  './models/trained/cvae_3_forward_'+str(configs.add_forward)+'_latent_'+str(configs.latent_dim)+'_lr_'+str(configs.lr)+'_STEP_'+ str(configs.if_lr_de)+'_epochs_'+ str(configs.epochs)+'_weight_forward_'+str(configs.weight_forward)+'_batch_size_'+ str(configs.batch_size)+'_trained_'+str(configs.Num)+'_.pth'
    path_temp =  './models/trained/cvae_3_forward_'+str(configs.add_forward)+'_latent_'+str(configs.latent_dim)+'_lr_'+str(configs.lr)+'_STEP_'+ str(configs.if_lr_de)+'_epochs_'+ str(configs.epochs)+'_weight_forward_'+str(configs.weight_forward)+'_batch_size_'+ str(configs.batch_size)+'_trained_'+str(configs.Num)+'_temp.pth'
    epochs = configs.epochs
    loss_all = np.zeros([3, configs.epochs])
    loss_val_best = 100
    
    for e in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion, configs)
        loss_train, temp = evaluate(model, train_loader, test_loader, optimizer, criterion, configs)
        loss_val, loss_pred = evaluate(model, val_loader, test_loader, optimizer, criterion, configs)
        loss_all[0,e] = loss_train
        loss_all[1,e] = loss_val
        loss_all[2,e] = loss_pred

        if loss_val_best >= loss_all[1,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[1,e]
            save_checkpoint(model, optimizer, e, loss_all, path, configs)
        
        lr = scheduler.get_lr()[0]

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, pred loss {:.6f},  lr {:.6f}, best {:.6f}.'.format(e, loss_train, loss_val,loss_pred, lr, loss_val_best))

        if e%10==0:
            save_checkpoint(model, optimizer, e, loss_all, path_temp, configs)


        scheduler.step()



if __name__  == '__main__':

    parser = argparse.ArgumentParser('nn models for inverse design: cGAN')
    parser.add_argument('--model', type=str, default='cVAE_GSNN')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension of x')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dim of y')
    
    parser.add_argument('--latent_dim', type=int, default=3, help='Dimension of latent variable')
    parser.add_argument('--add_forward', action='store_true', default='True', help='Connect cVAE to forward model or not')
    parser.add_argument('--weight_forward', type=float, default=10.0, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for forward model')
    parser.add_argument('--if_lr_de',action='store_true', default='True', help='If decrease learning rate duing training')    # to enable step lr, add argument '--if_lr_de' 
    parser.add_argument('--lr_de', type=float, default=0.2, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=2000, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    parser.add_argument('--Num', type=int, default=1, help='Run multiple times and save ')

    args = parser.parse_args()

    main(args)