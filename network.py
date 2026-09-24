import torch
import numpy as np
# import bilateral_ops
import convolutional_ops
import utils

import pyexr
import os
import sys
import math
from torch.nn import functional as F

from transformer_time_test import TransformerBlock

def reparametrize(conv1, conv3, conv5, has_identity_branch=False):
    reparam_kernel = conv5.weight + torch.nn.functional.pad(conv3.weight, [1, 1, 1, 1]) + torch.nn.functional.pad(conv1.weight, [2, 2, 2, 2])
    if has_identity_branch:
        identity_conv_weights = torch.eye(reparam_kernel.shape[0], device=conv1.weight.device).reshape((reparam_kernel.shape[0], reparam_kernel.shape[0], 1, 1))
        identity_conv_weights = torch.nn.functional.pad(identity_conv_weights, [2, 2, 2, 2])
        reparam_kernel += identity_conv_weights

    conv = torch.nn.Conv2d(in_channels=reparam_kernel.shape[1], out_channels=reparam_kernel.shape[0], kernel_size=5, stride=1, padding=2, bias=False)
    conv.weight.data = reparam_kernel.detach().clone()
    return conv



class WeightCNN(torch.nn.Module):
    def __init__(self, device=None, pretrained_model=None, state_dict=None, deployed=False):
        super().__init__()

        self.device = device
        self.deployed = deployed

        # CNN 3 (weight prediction)
        # if not deployed:
        self.conv_layer1_1 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_layer1_3 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_layer1_5 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
        self.lrelu_1 = torch.nn.LeakyReLU(0.1)

        self.conv_layer2_1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_layer2_3 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_layer2_5 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.conv_layer_1 = None
        self.conv_layer_2 = None



    def forward(self, x_input):
        # two tone mapped irradiance maps concatenated
        if not self.deployed:
            x1_1 = self.conv_layer1_1(x_input)
            x1_3 = self.conv_layer1_3(x_input)
            x1_5 = self.conv_layer1_5(x_input)
            x1_out = self.lrelu_1(x1_1 + x1_3 + x1_5)

            x2_1 = self.conv_layer2_1(x1_out)
            x2_3 = self.conv_layer2_3(x1_out)
            x2_5 = self.conv_layer2_5(x1_out)
            x6_out = self.sigmoid(x2_1 + x2_3 + x2_5)

            
        else:
            x1 = self.conv_layer_1(x_input)
            x1_out = self.lrelu_1(x1)

            x2 = self.conv_layer_2(x1_out)
            x6_out =  self.sigmoid(x2)

            
        return x6_out