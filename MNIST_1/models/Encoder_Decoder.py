import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, C, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.C = C

        self.conv1 = nn.Conv2d(C, 32, 4, 2, 1)  # 14 x 14
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 7 x 7
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 3 x 3
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 512, 3)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, self.output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, H, W = x.size()
        h = x.view(-1, self.C, H, W)
        h = self.bn1(self.conv1(h))
        h0 = h.view(-1, 32*14*14)
        h = self.act(h)
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return h0, z


class ConvDecoder(nn.Module):
    def __init__(self, C, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 3, 1, 0)  # 3 x 3
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1, 1)  # 7 x 7
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 14 x 14
        self.bn4 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, C, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        mu_img = self.conv_final(h)
        return mu_img
