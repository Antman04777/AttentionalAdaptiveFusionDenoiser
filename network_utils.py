import torch
import numpy as np
# import bilateral_ops
# import convolutional_ops
import utils

import pyexr
import os
import sys
import math

import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
import torch.nn as nn

import statistics


def norm(norm_type, out_ch):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(out_ch, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(out_ch, affine=False)
    else:
        raise NotImplementedError('Normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for lrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('Activation layer [{:s}] is not found'.format(act_type))
    return layer


def conv_block(in_ch, out_ch, kernel_size, stride=1, dilation=1, padding=0, padding_mode='zeros', norm_type=None,
               act_type='relu', groups=1, inplace=True):
    c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding,
                  padding_mode=padding_mode, groups=groups)
    n = norm(norm_type, out_ch) if norm_type else None
    a = act(act_type, inplace) if act_type else None
    return sequential(c, n, a)

def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class AFGSA(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, bias=False):
        super(AFGSA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)

        self.conv_map = conv_block(ch*2, ch, kernel_size=1, act_type='relu')
        self.q_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, noisy, aux):
        n_aux = self.conv_map(torch.cat([noisy, aux], dim=1))
        b, c, h, w, block, halo, heads = *noisy.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0, 'feature map dimensions must be divisible by the block size'

        q = self.q_conv(n_aux)
        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q *= self.head_ch ** -0.5  # b*#blocks, flattened_query, c

        k = self.k_conv(n_aux)
        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = self.v_conv(noisy)
        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.q_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.k_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.v_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class TransformerBlock(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, checkpoint=True):
        super(TransformerBlock, self).__init__()
        self.checkpoint = checkpoint
        self.attention = AFGSA(ch, block_size=block_size, halo_size=halo_size, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            conv_block(ch, ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu'),
            conv_block(ch, ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu')
        )

    def forward(self, x):
        if self.checkpoint:
            noisy = x[:, 0] + checkpoint(self.attention, x[:, 0], x[:, 1])
        else:
            noisy = x[:, 0] + self.attention(x[:, 0], x[:, 1])
        noisy = noisy + self.feed_forward(noisy)
        return (noisy, x[:, 1])
    
class RepBlock(torch.nn.Module):
    def __init__(self, device=None, in_ch=3,out_ch=1, is_input=False,pretrained_model=None, state_dict=None, deployed=False):
        super().__init__()

        self.device = device
        self.deployed = deployed
        self.is_input = is_input

        # CNN 3 (weight prediction)
        # if not deployed:
        self.conv_layer1_1 = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_layer1_3 = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_layer1_5 = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.lrelu_1 = torch.nn.LeakyReLU(0.1)

        self.conv_layer2_1 = torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_layer2_3 = torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_layer2_5 = torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.lrelu_2 = torch.nn.LeakyReLU(0.1)

        self.conv_layer_1 = None
        self.conv_layer_2 = None


    def forward(self, x_input):
        
        if not self.deployed:
            if self.is_input:
                x1_1 = self.conv_layer1_1(x_input)
                x1_3 = self.conv_layer1_3(x_input)
                x1_5 = self.conv_layer1_5(x_input)
                x1_out = self.lrelu_1(x1_1 + x1_3 + x1_5)
            else:
                x1_1 = self.conv_layer1_1(x_input)
                x1_3 = self.conv_layer1_3(x_input)
                x1_5 = self.conv_layer1_5(x_input)
                x1_out = self.lrelu_1(x1_1 + x1_3 + x1_5 + x_input) 

            x2_1 = self.conv_layer2_1(x1_out)
            x2_3 = self.conv_layer2_3(x1_out)
            x2_5 = self.conv_layer2_5(x1_out)
            x2_out = self.lrelu_2(x2_1 + x2_3 + x2_5 + x1_out)

            
        else:
            x1 = self.conv_layer_1(x_input)
            x1_out = self.lrelu_1(x1)

            x2 = self.conv_layer_2(x1_out)
            x2_out =  self.lrelu_2(x2)

            
        return x2_out

class WeightCNN(torch.nn.Module):
    def __init__(self, device=None, pretrained_model=None, state_dict=None, deployed=False):
        super().__init__()

        self.device = device
        self.deployed = deployed

        # CNN 3 (weight prediction)
        # if not deployed:
        self.conv_layer1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.rep1 = RepBlock(device, in_ch=16, out_ch=16).to(device)

        self.conv_layer2 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x_input):
        # two tone mapped irradiance maps concatenated
        #if not self.deployed:

        x1 = self.conv_layer1(x_input)
        x2 = self.rep1(x1)
        x3 = self.conv_layer2(x2)
        x3 = self.sigmoid(x3)

            
        return x3

class FeatureExtractor(torch.nn.Module):
    def __init__(self,device,num_input_channels=16):
        super().__init__()
        self.num_input_channels=num_input_channels
        self.device=device

        self.rep_noisy = RepBlock(device,in_ch=6,out_ch=64,is_input=True)
        self.rep_aux = RepBlock(device,in_ch=10,out_ch=64,is_input=True)

    def forward(self, x_input):

        noisy_ft = self.rep_noisy(x_input[:,0:6])
        aux_ft = self.rep_aux(x_input[:,6:16])

        return noisy_ft,aux_ft

class KMP(torch.nn.Module):
    def __init__(self,device,num_input_channels=16):
        super().__init__()
        self.device=device
        self.num_input_channels=num_input_channels
        self.transformer=TransformerBlock(ch=64,checkpoint=False)

        # self.conv_layer_6_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, bias=False)
        self.conv_layer_6 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv_layer_7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_6 = torch.nn.BatchNorm2d(128)
        self.relu_6 = torch.nn.ReLU()

        self.conv_layer_new_8 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_7 = torch.nn.BatchNorm2d(64)
        self.relu_new_7 = torch.nn.ReLU()

        self.conv_layer_8 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_8 = torch.nn.BatchNorm2d(64)
        self.relu_7 = torch.nn.ReLU()

        self.conv_layer_9_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False)

        self.conv_layer_10 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_9 = torch.nn.BatchNorm2d(64)
        self.relu_8 = torch.nn.ReLU()

        self.conv_layer_11 = torch.nn.Conv2d(in_channels=64, out_channels=49, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_10 = torch.nn.BatchNorm2d(49)
        self.relu_9 = torch.nn.ReLU()

    def forward(self, noisy_ft, aux_ft):

        inputs = torch.stack((noisy_ft,aux_ft), dim=1)
        x1,_ = self.transformer(inputs)

        x2 = self.conv_layer_6(x1)

        x3 = self.conv_layer_7(x2)
        x3 = self.bn_6(x3)
        x3 = self.relu_6(x3)

        x4 = self.conv_layer_new_8(x3)
        x4 = self.bn_7(x4) 
        x4 = self.relu_new_7(x4) 

        x5 = self.conv_layer_8(x4) 
        x5 = self.bn_8(x5) 
        x5 = self.relu_7(x5)
        kNonTA = x5[:,0:49] 

        x6 = self.conv_layer_9_t(x5) 

        x7 = self.conv_layer_10(x6) 
        x7 = self.bn_9(x7) 
        x7 = self.relu_8(x7) 

        x8 = self.conv_layer_11(x7) 
        x8 = self.bn_10(x8) 
        x8 = self.relu_9(x8)
        kTA = x8[:, 0:49]

        return kNonTA,kTA