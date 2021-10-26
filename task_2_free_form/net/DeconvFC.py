# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-05 12:45:28
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-31 20:28:38

import os
import sys
import numpy as np
import torch.nn as nn
import torch


class GeneratorNet(nn.Module):
    def __init__(self, in_num=29, out_num=3, d=64):
        super().__init__()
        self.in_num = in_num
        self.deconv_block = nn.Sequential(
            # ------------------------------------------------------
            nn.ConvTranspose1d(in_num, d * 8, 4, 1, 0),
            nn.BatchNorm1d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm1d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d * 2, d, 4, 2, 1),
            nn.BatchNorm1d(d),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d, 1, 4, 2, 1),
            nn.Tanh()
        )

        self.short_cut = nn.Sequential(
            nn.Conv1d(in_num, 64, 1, bias=False)
        )

        self.fc_block = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(64, 16 * d),
            nn.BatchNorm1d(16 * d),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(16 * d, d),
            nn.BatchNorm1d(d),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(d, out_num),
            nn.ReLU6()
        )

    def forward(self, x):
        x = x.view(-1, self.in_num, 1)
        xs = self.short_cut(x)
        net = self.deconv_block(x)
        net = net.view(net.size(0), -1)
        net = net + xs.squeeze(2)
        net = self.fc_block(net) / 6

        return net


if __name__ == '__main__':
    import torchsummary

    if torch.cuda.is_available():
        generator = GeneratorNet().cuda()
    else:
        generator = GeneratorNet()

    torchsummary.summary(generator, tuple([29]))
