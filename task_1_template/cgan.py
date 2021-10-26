import random

import sys
import torch
from torch import nn
import numpy as np

import argparse


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
#seed = 42
torch.manual_seed(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For cGAN model

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python cgan.py

def train(model, train_loader, optimizer_G, optimizer_D, criterion, prior):

    model.train()
    loss_epoch = 0
    g_loss_epoch = 0
    d_loss_epoch = 0

    for x, y in train_loader:

        batch_size = len(x)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Ground truth

        valid = torch.ones(batch_size, 1).to(DEVICE)
        fake = torch.zeros(batch_size, 1).to(DEVICE)

        # Train the generator

        optimizer_G.zero_grad()
        z = model.sample_noise(batch_size, prior).to(DEVICE)

        gen_y = model.generator(x, z)
        validity = model.discriminator(gen_y, x)
        g_loss = criterion(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # train the discriminator

        optimizer_D.zero_grad()
        # on real data

        real_pred = model.discriminator(y, x)
        d_loss_real = criterion(real_pred, valid)

        # on generated data
        gen_y = model.generator(x, z)
        fake_pred = model.discriminator(gen_y, x)
        d_loss_fake = criterion(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake)/2
        d_loss.backward()
        optimizer_D.step()

        g_loss_epoch += g_loss * batch_size
        d_loss_epoch += d_loss * batch_size

    g_loss_epoch, d_loss_epoch = g_loss_epoch / len(train_loader.dataset), d_loss_epoch / len(train_loader.dataset)
    print('generator loss {:.6f}, discriminator loss {:.6f}'.format(g_loss_epoch, d_loss_epoch))

    return g_loss_epoch + d_loss_epoch, g_loss_epoch, d_loss_epoch


def evaluate(model, val_loader, test_loader, optimizer_G, optimizer_D, forward_model, criterion, prior, test=False):

    model.eval()
    dataloader = test_loader if test else val_loader
    with torch.no_grad():
        loss_epoch = 0
        
        for x, y in dataloader:
            batch_size = len(x)
            x, y = x.to(DEVICE), y.to(DEVICE)

            valid = torch.ones(batch_size, 1).to(DEVICE)
            fake = torch.zeros(batch_size, 1).to(DEVICE)

            # loss on generator 

            z = model.sample_noise(batch_size, prior).to(DEVICE)
            gen_y = model.generator(x, z)

            
            validity = model.discriminator(gen_y, x)
            g_loss = criterion(validity, valid)
            
            # discriminator loss on real data 
            
            real_pred = model.discriminator(y, x)
            d_loss_real = criterion(real_pred, valid)
            
            # discriminator loss on fake data 

            fake_pred = model.discriminator(gen_y, x)
            d_loss_fake = criterion(fake_pred, fake)
            d_loss = (d_loss_real + d_loss_fake)/2

            loss_epoch += ((g_loss + d_loss) * batch_size)
        
        cie_raw, param_raw, cie_pred, param_pred = evaluate_gan_minmax_inverse(model, forward_model, val_loader.dataset, show=0)
        rmse_cie_raw = np.sqrt(np.sum(np.average(np.square((cie_raw - cie_pred)),axis=0)))
        loss_epoch = loss_epoch / len(dataloader.dataset)
    return loss_epoch, rmse_cie_raw


def save_checkpoint(model, optimizer_G, optimizer_D, epoch, loss_all, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'G_optimizer_state_dict': optimizer_G.state_dict(),
            'D_optimizer_state_dict': optimizer_D.state_dict(),
            'loss_all':loss_all,
            'configs':configs,
            'seed':seed,
        }, path)


def main(configs):
    
    train_loader, val_loader, test_loader = get_dataloaders('gan', configs.batch_size)

    model = cGAN(configs.input_dim, configs.output_dim, configs.noise_dim).to(DEVICE)
    model.apply(weights_init_normal)

    #model.load_state_dict(torch.load('./models/gan_trained_1.pth')['model_state_dict'])
    
    # set up optimizer and criterion 
    
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=configs.g_lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=configs.d_lr,betas=(configs.beta_1, configs.beta_2),  weight_decay=configs.weight_decay)
    
    scheduler_G = StepLR(optimizer_G, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
    scheduler_D = StepLR(optimizer_D, step_size=configs.epoch_lr_de, gamma=configs.lr_de)

    criterion = torch.nn.BCELoss()

    # load the forward_model for evaluate the RMSE on validation dataset

    forward_model = MLP(4, 3).to(DEVICE)
    forward_model.load_state_dict(torch.load('./models/examples/forward_model_trained_evaluate_1.pth')['model_state_dict'])
    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
    
    # start training 
    path =  './models/trained/cgan_3_batch_'+str(configs.batch_size)+'_epoch_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)+'_noise_'+str(configs.noise_dim)+'_g_'+str(configs.g_lr)+'_d_'+str(configs.d_lr)+'_STEP_'+ str(configs.if_lr_de) +str(configs.Num)+'_.pth'
    path_temp =  './models/trained/cgan_3_batch_'+str(configs.batch_size)+'_epoch_'+str(configs.epochs)+'_'+str(configs.epoch_lr_de)+'_noise_'+str(configs.noise_dim)+'_g_'+str(configs.g_lr)+'_d_'+str(configs.d_lr)+'_STEP_'+ str(configs.if_lr_de) +str(configs.Num)+'_temp.pth'
    

    epochs = configs.epochs
    loss_all = np.zeros([5, configs.epochs])
    loss_val_best = 100
    
    for e in range(epochs):

        loss_train, loss_train_g, loss_train_d = train(model, train_loader, optimizer_G, optimizer_D, criterion, configs.prior)
        loss_val, rmse_val = evaluate(model, val_loader, test_loader, optimizer_G, optimizer_D, forward_model, criterion, configs.prior)
        loss_all[0,e] = loss_train
        loss_all[1,e], loss_all[2,e] = loss_val, rmse_val
        loss_all[3,e], loss_all[4,e] = loss_train_g, loss_train_d

        if loss_val_best >= loss_all[2,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[2,e]
            save_checkpoint(model, optimizer_G, optimizer_D, e, loss_all, path, configs)

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, val RMSE {:.6f}, best {:.6f}'.format(e, loss_train, loss_val, rmse_val, loss_val_best))
        if e%10==0:
            save_checkpoint(model, optimizer_G, optimizer_D, e, loss_all, path_temp, configs)


        scheduler_D.step()
        scheduler_G.step()



if __name__  == '__main__':
    parser = argparse.ArgumentParser('nn models for inverse design: cGAN')
    parser.add_argument('--model', type=str, default='cGAN')
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension of condition for the generator')
    parser.add_argument('--output_dim', type=int, default=4, help='Output size of generator')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of dataset')
    parser.add_argument('--noise_dim', type=int, default=2, help='Dimension of noise')
    parser.add_argument('--prior', type=int, default=1, help='1 for (0,1) normal distribution, 0 for (0,1) uniform distribution')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--g_lr', type=float, default=5e-4, help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=5e-4, help='Learning rate for discriminator')
    parser.add_argument('--if_lr_de',action='store_true', default='False', help='If decrease learning rate duing training')
    parser.add_argument('--lr_de', type=float, default=0.2, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=4000, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    parser.add_argument('--Num', type=int, default=1, help='Run multiple times and save ')
    args = parser.parse_args()

    main(args)