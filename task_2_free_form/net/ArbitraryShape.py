# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-09-04 14:27:09
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-11 20:58:44

import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) *\
        torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) /
        (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel.requires_grad_(False)


class GeneratorNet(nn.Module):
    def __init__(self, noise_dim=100, ctrast_dim=58, d=32, kernel_size=5):
        super().__init__()
        self.noise_dim = noise_dim
        self.ctrast_dim = ctrast_dim
        self.gaussian_kernel = get_gaussian_kernel(kernel_size)
        self.pad = (kernel_size - 1) // 2
        self.deconv_block_ctrast = nn.Sequential(
            nn.ConvTranspose2d(self.ctrast_dim, d * 4, 4, 1, 0),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_noise = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, d * 4, 4, 1, 0),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.fc_block_1 = nn.Sequential(
            nn.Linear(64 * 64, 64 * 16),
            nn.BatchNorm1d(64 * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(64 * 16, 64 * 4),
            nn.BatchNorm1d(64 * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * 4, ctrast_dim),
            nn.BatchNorm1d(ctrast_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(ctrast_dim, 1),
            nn.Tanh()
        )
        self.short_cut = nn.Sequential(
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise_in, ctrast_in):
        noise = self.deconv_block_noise(noise_in.view(-1, self.noise_dim, 1, 1))
        ctrast = self.deconv_block_ctrast(ctrast_in.view(-1, self.ctrast_dim, 1, 1))
        net = torch.cat((noise, ctrast), 1)
        img = self.deconv_block_cat(net)
        # img = F.conv2d(img, self.gaussian_kernel, padding=self.pad)
        gap_in = self.fc_block_1(img.view(img.size(0), -1)) + self.short_cut(ctrast_in.view(ctrast_in.size(0), -1))
        gap = self.fc_block_2(gap_in)
        return (img + 1) / 2, (gap + 1) / 2


class SimulatorNet(nn.Module):
    def __init__(self, spec_dim=58, d=32):
        super().__init__()
        self.spec_dim = spec_dim
        self.conv_block_shape = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(1, 3 * d // 4, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_gap = nn.Sequential(
            # ------------------------------------------------------
            nn.ReplicationPad2d((63, 0, 63, 0)),
            nn.Conv2d(1, d // 4, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, 4, 2, 1),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, 4, 1, 0),
            nn.Sigmoid()
        )
        self.fc_block = nn.Sequential(
            nn.Linear(d * 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.2),
            nn.Linear(128, spec_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in):
        shape = self.conv_block_shape(shape_in)
        gap = self.conv_block_gap(gap_in.view(-1, 1, 1, 1))
        net = torch.cat((shape, gap), 1)
        spec = self.conv_block_cat(net)
        spec = self.fc_block(spec.view(spec.shape[0], -1))
        return (spec + 1) / 2

class SimulatorNet_new_linear(nn.Module):

    def __init__(self, img_size=64, gap_dim=1, spec_dim=58, d=16, thickness = [1,1], k_size = 3, k_pad = 1):

        super(SimulatorNet_new_linear, self).__init__()

        self.img_size = img_size
        self.gap_dim = gap_dim 
        self.spec_dim = spec_dim 
        self.d = d 
        self.thickness = thickness # for layers of gap, spec, img
        self.k_size = k_size
        self.k_pad = k_pad

        self.embed_gap = nn.Linear(self.gap_dim, img_size*img_size*self.thickness[0])
        self.embed_img = nn.Conv2d(1, self.thickness[0], kernel_size=1)
        
        self.net = nn.Sequential(

            nn.Conv2d(sum(self.thickness), d , self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d ),
            nn.LeakyReLU(0.2),
            # -------------------------------------------
            nn.Conv2d(d, d * 2, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 16),
            nn.LeakyReLU(0.2),
        )

        self.net_fc = nn.Sequential(
            nn.Linear(d * 16*4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, spec_dim),
            nn.Tanh()

        )


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in):

        embed_gap = self.embed_gap(gap_in)
        embed_gap = embed_gap.view(len(gap_in), -1, self.img_size, self.img_size)

        img = self.embed_img(shape_in)

        x = torch.cat([img, embed_gap], dim=1)

        y = self.net(x)

        y = torch.flatten(y, start_dim=1)

        spec = self.net_fc(y)


        return (spec + 1) / 2


class SimulatorNet_new(nn.Module):
    def __init__(self, spec_dim=58, d=32):
        super().__init__()
        self.spec_dim = spec_dim
        self.conv_block_shape = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(1, 3 * d // 4, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_gap = nn.Sequential(
            # ------------------------------------------------------
            nn.ReplicationPad2d((63, 0, 63, 0)),
            nn.Conv2d(1, d // 4, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(d, d * 2, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, 4, 1, 0),
            nn.LeakyReLU(0.2)
            #nn.Sigmoid()
        )
        self.fc_block = nn.Sequential(
            nn.Linear(d * 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.2),
            nn.Linear(128, spec_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in):
        shape = self.conv_block_shape(shape_in)
        gap = self.conv_block_gap(gap_in.view(-1, 1, 1, 1))
        net = torch.cat((shape, gap), 1)
        spec = self.conv_block_cat(net)
        spec = self.fc_block(spec.view(spec.shape[0], -1))
        return (spec + 1) / 2

class SimulatorNet_new_fc(nn.Module):
    def __init__(self, spec_dim=58, d=32):
        super().__init__()
        self.spec_dim = spec_dim
        self.conv_block_shape = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(1, 3 * d // 4, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_gap = nn.Sequential(
            # ------------------------------------------------------
            nn.ReplicationPad2d((63, 0, 63, 0)),
            nn.Conv2d(1, d // 4, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(d, d * 2, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, 4, 1, 0),
            nn.LeakyReLU(0.2)
            #nn.Sigmoid()
        )
        self.fc_block = nn.Sequential(
            nn.Linear(d * 16, spec_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in):
        shape = self.conv_block_shape(shape_in)
        gap = self.conv_block_gap(gap_in.view(-1, 1, 1, 1))
        net = torch.cat((shape, gap), 1)
        spec = self.conv_block_cat(net)
        spec = self.fc_block(spec.view(spec.shape[0], -1))
        return (spec + 1) / 2


class SimulatorNet_small(nn.Module):
    def __init__(self, spec_dim=58, d=16):
        super().__init__()
        self.spec_dim = spec_dim
        self.conv_block_shape = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(1, d-1, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_gap = nn.Sequential(
            # ------------------------------------------------------
            nn.ReplicationPad2d((31, 0, 31, 0))
        )
        self.conv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, 4, 2, 1),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(d * 8 * 16, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.2),
            nn.Linear(256, spec_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in):
        shape = self.conv_block_shape(shape_in)
        gap = self.conv_block_gap(gap_in.view(-1, 1, 1, 1))
        net = torch.cat((shape, gap), 1)
        spec = self.conv_block_cat(net)
        spec = torch.flatten(spec, 1)
        spec = self.fc_block(spec)
        return (spec + 1) / 2



class InverseNet(nn.Module):
    def __init__(self, spec_dim=58, d=32):
        super().__init__()
        self.spec_dim = spec_dim
        self.deconv_block_spec = nn.Sequential(
            nn.ConvTranspose2d(self.spec_dim, d * 8, 4, 1, 0),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.fc_block_1 = nn.Sequential(
            nn.Linear(64 * 64, 64 * 16),
            nn.BatchNorm1d(64 * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(64 * 16, 64 * 4),
            nn.BatchNorm1d(64 * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * 4, spec_dim),
            nn.BatchNorm1d(spec_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(spec_dim, 1),
            nn.Tanh()
        )
        self.short_cut = nn.Sequential(
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, spec_in):

        spec = self.deconv_block_spec(spec_in.view(-1, self.spec_dim, 1, 1))
        img = self.deconv_block_cat(spec)
        gap_in = self.fc_block_1(img.view(img.size(0), -1))
        gap = self.fc_block_2(gap_in)

        return (img + 1) / 2, (gap + 1) / 2




class InverseNet_new(nn.Module):

    def __init__(self, img_size=64, spec_dim=58, d=16, k_size = 3, k_pad = 1):
        super(InverseNet_new, self).__init__()

        self.img_size = img_size
        self.spec_dim = spec_dim 
        self.d = d 
        self.k_size = k_size
        self.k_pad = k_pad


        self.embed_spec = nn.Linear(spec_dim, d*16*4)

        self.net_conv = nn.Sequential(
            
            nn.ConvTranspose2d(d*16, d*8, kernel_size=k_size, stride = 2, padding=k_pad, output_padding=1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*8, d*4, kernel_size=k_size, stride = 2, padding=k_pad, output_padding=1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*4, d*2, kernel_size=k_size, stride = 2, padding=k_pad, output_padding=1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*2, d, kernel_size=k_size, stride = 2, padding=k_pad, output_padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
            # ---------------------------------------------
        )

        self.final_img_1 = nn.Sequential(
            nn.ConvTranspose2d(d, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.Conv2d(d, out_channels= 1, kernel_size= 3, padding= 1),
            nn.Tanh()
        )

        self.final_img_2 = nn.Sequential(
            nn.ConvTranspose2d(d, 1, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        )

        self.final_gap = nn.Sequential(
            nn.Linear(img_size*img_size, 1),
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, spec):
        # a surrogate model for inverse prediction of img and gap from spec
        result = self.embed_spec(spec)  # fc layer 
        result = result.view(-1, self.d*16, 2, 2)

        result = self.net_conv(result)
        img_hat = self.final_img_1(result) # 
        gap_hat = self.final_img_2(result)
        gap_hat = gap_hat.view(-1, self.img_size*self.img_size)
        gap_hat = self.final_gap(gap_hat)

        return (img_hat+1)/2, (gap_hat+1)/2


class TandemNet(nn.Module):

    def __init__(self, forward_model, inverse_model):

        super(TandemNet, self).__init__()

        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, spec):
        '''
        Pass the desired target spectrum to the tandem network.
        '''

        shape, gap = self.inverse_model(spec)
        spec_pred = self.forward_model(shape, gap)

        return spec_pred

    def pred(self, spec):
        # prediction of forward model

        shape, gap = self.inverse_model(spec)

        return shape, gap



class cVAE_hybrid(nn.Module):

    def __init__(self, forward_model, vae_model):
        
        super(cVAE_hybrid, self).__init__()
        self.forward_model = forward_model
        self.vae_model = vae_model

    def forward(self, img, gap, spec):
        #  the prediction is based on the VAE_GSNN model

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat = self.vae_model(img, gap, spec)

        spec_pred = self.forward_model(img_pred, gap_pred)

        return img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred



class cVAE_GSNN(nn.Module):

    def __init__(self, img_size=64, gap_dim=1, spec_dim=58, latent_dim=100, d=16, thickness = [1,1,1], k_size = 3, k_pad = 1):
        
        super(cVAE_GSNN, self).__init__()

        self.img_size = img_size
        self.gap_dim = gap_dim 
        self.spec_dim = spec_dim 
        self.latent_dim = latent_dim 
        self.d = d 
        self.thickness = [1,1,1] # for layers of gap, spec, img
        self.k_size = k_size
        self.k_pad = k_pad

        self.embed_gap = nn.Linear(self.gap_dim, img_size*img_size*self.thickness[0])
        self.embed_spec = nn.Linear(self.spec_dim, img_size*img_size*self.thickness[1])

        self.embed_img = nn.Conv2d(1, self.thickness[2], kernel_size=1)


        self.encoder = nn.Sequential(

            nn.Conv2d(sum(self.thickness), d , self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d ),
            nn.LeakyReLU(0.2),
            # -------------------------------------------
            nn.Conv2d(d, d * 2, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 16),
            nn.LeakyReLU(0.2)
        )

        self.mu_head = nn.Linear(d*16*4, self.latent_dim)
        self.logvar_head = nn.Linear(d*16*4, self.latent_dim)

        self.decoder_input = nn.Linear(latent_dim+spec_dim, d*16*4)

        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(d*16, d*8, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*8, d*4, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*4, d*2, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*2, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
            # ---------------------------------------------
        )

        self.final_img_1 = nn.Sequential(
            nn.ConvTranspose2d(d, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.Conv2d(d, out_channels= 1, kernel_size= k_size, padding= k_pad),
            nn.Tanh()
        )

        self.final_img_2 = nn.Sequential(
            nn.ConvTranspose2d(d, 1, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2)
        )

        self.final_gap = nn.Sequential(
            nn.Linear(img_size*img_size, 1),
            nn.Tanh()
        )

        # those are for the forward_net for inverse design:

        self.forward_embed_spec = nn.Linear(spec_dim, d*16*4)

        self.forward_net_conv = nn.Sequential(
            
            nn.ConvTranspose2d(d*16, d*8, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*8, d*4, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*4, d*2, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*2, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
            # ---------------------------------------------
        )

        self.forward_final_img_1 = nn.Sequential(
            nn.ConvTranspose2d(d, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.Conv2d(d, out_channels= 1, kernel_size= k_size, padding= k_pad),
            nn.Tanh()
        )

        self.forward_final_img_2 = nn.Sequential(
            nn.ConvTranspose2d(d, 1, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2)
        )

        self.forward_final_gap = nn.Sequential(
            nn.Linear(img_size*img_size, 1),
            nn.Tanh()
        )


    def encode(self, x):
        
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        return self.mu_head(result), self.logvar_head(result)
    
    def decode(self, z):
        
        result = self.decoder_input(z)  # fc layer 
        result = result.view(-1, self.d*16, 2, 2)

        result = self.decoder(result)
        img_pred = self.final_img_1(result) # 

        gap_pred = self.final_img_2(result)
        gap_pred = gap_pred.view(-1, self.img_size*self.img_size)

        gap_pred = self.final_gap(gap_pred)

        return (img_pred+1)/2, (gap_pred+1)/2


    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, img, gap, spec):
        # 
        embed_gap = self.embed_gap(gap)
        #embed_gap = embed_gap.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embed_gap = embed_gap.view(len(gap), -1, self.img_size, self.img_size)

        embed_spec = self.embed_spec(spec)
        #embed_spec = embed_spec.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embed_spec = embed_spec.view(len(gap), -1, self.img_size, self.img_size)
        
        img = self.embed_img(img)
        
        x = torch.cat([img, embed_spec, embed_gap], dim=1)
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        z = torch.cat([z, spec], dim=1)
        img_pred, gap_pred = self.decode(z)

        img_hat, gap_hat = self.forward_net(spec)
        return img_pred, gap_pred, mu, logvar, img_hat, gap_hat

    def inference(self, spec):
        # prediction using the spectrum 

        img_hat, gap_hat = self.forward_net(spec)

        embed_gap = self.embed_gap(gap_hat)
        #embed_gap = embed_gap.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embed_gap = embed_gap.view(len(spec), -1, self.img_size, self.img_size)

        embed_spec = self.embed_spec(spec)
        #embed_spec = embed_spec.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embed_spec = embed_spec.view(len(spec), -1, self.img_size, self.img_size)

        img_hat = self.embed_img(img_hat)

        x = torch.cat([img_hat, embed_spec, embed_gap], dim=1)
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        z = torch.cat([z, spec], dim=1)
        img_pred, gap_pred = self.decode(z)# return the inference of img and gap

        return img_pred, gap_pred, mu, logvar, img_hat, gap_hat

    def forward_net(self, spec):
        # a surrogate model for inverse prediction of img and gap from spec
        result = self.forward_embed_spec(spec)  # fc layer 
        result = result.view(-1, self.d*16, 2, 2)

        result = self.forward_net_conv(result)
        img_hat = self.forward_final_img_1(result) # 
        gap_hat = self.forward_final_img_2(result)
        gap_hat = gap_hat.view(-1, self.img_size*self.img_size)
        gap_hat = self.forward_final_gap(gap_hat)

        return (img_hat+1)/2, (gap_hat+1)/2




class Generator(nn.Module):

    def __init__(self, img_size=64, gap_dim=1, spec_dim=58, noise_dim=100, d=16, k_size = 3, k_pad = 1):

        super(Generator, self).__init__()

        self.img_size = img_size
        self.gap_dim = gap_dim 
        self.spec_dim = spec_dim 
        self.noise_dim = noise_dim 
        self.d = d 
        self.k_size = k_size
        self.k_pad = k_pad

        self.embed_input = nn.Linear(self.spec_dim+self.noise_dim, 4*16*d)

        self.net = nn.Sequential(

            nn.ConvTranspose2d(d*16, d*8, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*8, d*4, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*4, d*2, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.ConvTranspose2d(d*2, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
            # ---------------------------------------------
        )

        self.net_conv1 = nn.Sequential(
            nn.ConvTranspose2d(d, d, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            # ---------------------------------------------
            nn.Conv2d(d, out_channels= 1, kernel_size= k_size, padding= k_pad),
            nn.Tanh()
        )

        self.net_conv2 = nn.Sequential(
            nn.ConvTranspose2d(d, 1, kernel_size = k_size, stride = 2, padding = k_pad, output_padding = 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2)
        )

        self.net_fc = nn.Sequential(
            nn.Linear(self.img_size*self.img_size, 1),
            nn.Tanh()
        )


    def forward(self, spec, noise):
        # return generated spec and gap information 
        x = torch.cat([spec, noise], dim=1)
        x = self.embed_input(x)
        x = x.view(len(spec), self.d*16, 2, 2 )


        result = self.net(x)

        img_pred = self.net_conv1(result)

        gap_pred = self.net_conv2(result)
        gap_pred = gap_pred.view(-1, self.img_size*self.img_size)
        gap_pred = self.net_fc(gap_pred)

        return (img_pred+1)/2, (gap_pred+1)/2



class Discriminator(nn.Module):

    def __init__(self, img_size=64, gap_dim=1, spec_dim=58, noise_dim=100, d=16, thickness = [1,1,1], k_size = 3, k_pad = 1):

        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.gap_dim = gap_dim 
        self.spec_dim = spec_dim 
        self.noise_dim = noise_dim 
        self.d = d 
        self.thickness = thickness # for layers of gap, spec, img
        self.k_size = k_size
        self.k_pad = k_pad

        self.embed_gap = nn.Linear(self.gap_dim, img_size*img_size*self.thickness[0])
        self.embed_spec = nn.Linear(self.spec_dim, img_size*img_size*self.thickness[1])

        self.embed_img = nn.Conv2d(1, self.thickness[2], kernel_size=1)


        self.net = nn.Sequential(

            nn.Conv2d(sum(self.thickness), d , self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d ),
            nn.LeakyReLU(0.2),
            # -------------------------------------------
            nn.Conv2d(d, d * 2, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, self.k_size, 1, self.k_pad),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(d * 16),
            nn.LeakyReLU(0.2)
        )

        self.net_fc = nn.Sequential(
            nn.Linear(16*d*4, 1),
            nn.Sigmoid()
        )

    
    def forward(self, img, gap, spec):
        # 
        embed_gap = self.embed_gap(gap)
        #embed_gap = embed_gap.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embed_gap = embed_gap.view(len(gap), -1, self.img_size, self.img_size)

        embed_spec = self.embed_spec(spec)
        #embed_spec = embed_spec.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embed_spec = embed_spec.view(len(gap), -1, self.img_size, self.img_size)
        
        img = self.embed_img(img)
        
        x = torch.cat([img, embed_spec, embed_gap], dim=1)

        y = self.net(x)

        y = torch.flatten(y, start_dim=1)

        validity = self.net_fc(y)

        return validity

class cGAN(nn.Module):
    
    def __init__(self, img_size=64, gap_dim=1, spec_dim=58, noise_dim=100, d=16, thickness = [1,1,1], k_size = 3, k_pad = 1):

        super(cGAN, self).__init__()
        self.Generator = Generator(img_size, gap_dim, spec_dim, noise_dim, d, k_size, k_pad)
        self.Discriminator = Discriminator(img_size, gap_dim, spec_dim, noise_dim, d, thickness, k_size, k_pad)

        self.noise_dim = noise_dim


    
    def forward(self, spec, noise):

        img_fake, gap_fake = self.Generator(sepc, noise)
        validity = self.Discriminator(img_fake, gap_fake, spec)

        return validity


    def sample_noise(self, batch_size, prior=1):

        if prior == 1:
            z = torch.tensor(np.random.normal(0, 1, (batch_size, self.noise_dim))).float()
        else:
            z = torch.tensor(np.random.uniform(0, 1, (batch_size, self.noise_dim))).float()
        return z





















































if __name__ == '__main__':
    import torchsummary

    if torch.cuda.is_available():
        generator = GeneratorNet().cuda()
    else:
        generator = GeneratorNet()

    torchsummary.summary(generator, [tuple([100]), tuple([58])])

    # if torch.cuda.is_available():
    #     simulator = SimulatorNet().cuda()
    # else:
    #     simulator = SimulatorNet()

    # torchsummary.summary(simulator, [(1, 64, 64), tuple([1])])
