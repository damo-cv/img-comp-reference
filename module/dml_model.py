"""Hyper Model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from .ops import Upsample


class HyperEncoder(nn.Module):
    def __init__(self, in_channel=128, out_channel=128, channel=128):
        super(HyperEncoder, self).__init__()
        self.channel = channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encoder = self.build_encoder()
 
    def build_encoder(self):
        return torch.nn.Sequential(
            nn.Conv2d(self.in_channel, self.channel, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel, self.channel, 5, stride=2, padding=2, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel, self.out_channel, 5, stride=2, padding=2, padding_mode='zeros')
        )

    def forward(self, x):
        return self.encoder(x)

class HyperDecoder(nn.Module):
    def __init__(self, in_channel=128, out_channel=128*2, channel=128):
        super(HyperDecoder, self).__init__()
        self.channel = channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.decoder = self.build_decoder()

    def build_decoder(self):
        return torch.nn.Sequential(
            Upsample(self.in_channel, self.channel, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Upsample(self.channel, self.channel, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel, self.out_channel, 1, stride=1, padding=0, padding_mode='zeros')
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

