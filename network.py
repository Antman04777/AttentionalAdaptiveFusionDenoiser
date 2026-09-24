import torch
import numpy as np
# import bilateral_ops
# import convolutional_ops
import utils

import pyexr
import os
import sys
import math
from torch.nn import functional as F

#from memory_profiler import profile

def reparametrize(conv1, conv3, conv5, has_identity_branch=False):
    reparam_kernel = conv5.weight + torch.nn.functional.pad(conv3.weight, [1, 1, 1, 1]) + torch.nn.functional.pad(conv1.weight, [2, 2, 2, 2])
    if has_identity_branch:
        identity_conv_weights = torch.eye(reparam_kernel.shape[0], device=conv1.weight.device).reshape((reparam_kernel.shape[0], reparam_kernel.shape[0], 1, 1))
        identity_conv_weights = torch.nn.functional.pad(identity_conv_weights, [2, 2, 2, 2])
        reparam_kernel += identity_conv_weights

    conv = torch.nn.Conv2d(in_channels=reparam_kernel.shape[1], out_channels=reparam_kernel.shape[0], kernel_size=5, stride=1, padding=2, bias=False)
    conv.weight.data = reparam_kernel.detach().clone()
    return conv


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

class SqueezeExcite(torch.nn.Module):
    def __init__(self, device, r=3, num_input_channels=3):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.r = r
        self.channel_compressed = int(num_input_channels / r)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fully_connect = torch.nn.Sequential(
            torch.nn.Linear(num_input_channels, self.channel_compressed,bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(self.channel_compressed, num_input_channels, bias = False),
            torch.nn.Sigmoid()
        )

    def forward(self, x_in):
        b, c, _, _ = x_in.shape
        embedding = self.avg_pool(x_in).view(b, c)
        channel_weights = self.fully_connect(embedding).view(b, c, 1, 1)
        x_out = x_in * channel_weights.expand_as(x_in)
        return x_out


class AdaptiveFusion(torch.nn.Module):
    def __init__(self, device, num_input_channels=16):
        super().__init__()

        self.device = device

        self.num_input_channels = num_input_channels

        self.rep1_noisy = RepBlock(device, in_ch=self.num_input_channels, out_ch=32, is_input=True).to(device)
        self.squeeze_excite_layer1_noisy = SqueezeExcite(device, r=4, num_input_channels=32).to(device)

        self.rep1_aux = RepBlock(device, in_ch=self.num_input_channels, out_ch=32, is_input=True).to(device)
        self.squeeze_excite_layer1_aux = SqueezeExcite(device, r=4, num_input_channels=32).to(device)

        self.conv_layer_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_2 = torch.nn.BatchNorm2d(64)
        self.relu_2 = torch.nn.ReLU()

        self.rep2 = RepBlock(device, in_ch=64, out_ch=64).to(device)

        self.conv_layer_3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_3 = torch.nn.BatchNorm2d(128)
        self.relu_3 = torch.nn.ReLU()

        self.rep3 = RepBlock(device, in_ch=128, out_ch=128).to(device)

        #for decoder
        self.squeeze_excite_layer2 = SqueezeExcite(device, r=4, num_input_channels=128).to(device)

        self.conv_layer_4_t = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, bias=False)

        self.conv_layer_5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_4 = torch.nn.BatchNorm2d(128)
        self.relu_4 = torch.nn.ReLU()

        self.conv_layer_6 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_5 = torch.nn.BatchNorm2d(64)
        self.relu_5 = torch.nn.ReLU()

        self.conv_layer_7_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False)

        self.conv_layer_8 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_6 = torch.nn.BatchNorm2d(64)
        self.relu_6 = torch.nn.ReLU()

        self.conv_layer_9 = torch.nn.Conv2d(in_channels=128, out_channels=49, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_7 = torch.nn.BatchNorm2d(49)
        self.relu_7 = torch.nn.ReLU()

    def forward(self, x_input):

        x_1_noisy = self.rep1_noisy(x_input)
        x_1_noisy = self.squeeze_excite_layer1_noisy(x_1_noisy)  #res connection out

        x_1_aux = self.rep1_aux(x_input)
        x_1_aux = self.squeeze_excite_layer1_aux(x_1_aux)

        x_embedded = torch.cat((x_1_noisy, x_1_aux), axis=1)

        x_2 = self.conv_layer_2(x_embedded)
        x_2 = self.bn_2(x_2)
        x_2 = self.relu_2(x_2)

        x_3 = self.rep2(x_2)

        x_4 = self.conv_layer_3(x_3)
        x_4 = self.bn_3(x_4)
        x_4 = self.relu_3(x_4) # residual connection output

        x_5 = self.rep3(x_4)
        x_5 = self.squeeze_excite_layer2(x_5)

        x_6 = self.conv_layer_4_t(x_5)

        x_7 = self.conv_layer_5(x_6)
        x_7 = self.bn_4(x_7)
        x_7 = self.relu_4(x_7)

        x_8 = torch.cat((x_2, x_7), axis=1)
        x_8 = self.conv_layer_6(x_8)
        x_8 = self.bn_5(x_8)
        x_8 = self.relu_5(x_8)
        k_non_TA = x_8[:,0:49] #sliced first 49 channels for non ta kernel map

        x_9 = self.conv_layer_7_t(x_8)

        x_10 = self.conv_layer_8(x_9)
        x_10 = self.bn_6(x_10)
        x_10 = self.relu_6(x_10)

        x_11_in = torch.cat((x_embedded, x_10), axis=1)
        x_11 = self.conv_layer_9(x_11_in)
        x_11 = self.bn_7(x_11)
        x_11 = self.relu_7(x_11)
        k_TA = x_11

        return k_non_TA, k_TA

class WeightCNN(torch.nn.Module):
    def __init__(self, device=None, pretrained_model=None, state_dict=None, deployed=False):
        super().__init__()

        self.device = device
        self.deployed = deployed

        # CNN 3 (weight prediction)
        # if not deployed:
        self.conv_layer1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.rep1 = RepBlock(device, in_ch=16, out_ch=16).to(device)
        #self.conv_layer1_1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv_layer1_3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv_layer1_5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        #self.lrelu_1 = torch.nn.LeakyReLU(0.1)

        #self.conv_layer2_1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv_layer2_3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv_layer2_5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        #self.lrelu_2 = torch.nn.LeakyReLU(0.1)

        self.conv_layer2 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        #self.conv_layer_1 = None
        #self.conv_layer_2 = None
        #self.conv_layer_3 = None
        #self.conv_layer_4 = None
        #self.conv_layer_5 = None
        #self.conv_layer_6 = None


    def forward(self, x_input):
        # two tone mapped irradiance maps concatenated
        #if not self.deployed:

        x1 = self.conv_layer1(x_input)
        x2 = self.rep1(x1)
        x3 = self.conv_layer2(x2)
        x3 = self.sigmoid(x3)
            #x1_1 = self.conv_layer1_1(x_input)
            #x1_3 = self.conv_layer1_3(x_input)
            #x1_5 = self.conv_layer1_5(x_input)
            #x1_out = self.lrelu_1(x1_1 + x1_3 + x1_5)
#
            #x2_1 = self.conv_layer2_1(x1_out)
            #x2_3 = self.conv_layer2_3(x1_out)
            #x2_5 = self.conv_layer2_5(x1_out)
            #x6_out = self.sigmoid(x2_1 + x2_3 + x2_5)

            
        #else:
        #    x1 = self.conv_layer_1(x_input)
        #    x1_out = self.lrelu_1(x1)
#
        #    x2 = self.conv_layer_2(x1_out)
#
        #    x6_out =  self.sigmoid(x2)

            
        return x3


class RepVGG(torch.nn.Module):
    def __init__(self, device, deployed=False, pretrained_model=None, state_dict_preload=None, ss_non_ta=16, sr_non_ta=16, ss_ta=4, sr_ta=4):
        super().__init__()
        # define convolutional layers
        # 4 output channels (weight map 1 (1 channel), weight map 2 (1 channel), guide 1 (1 channel), guide 2 (1 channel))
        # same repvgg structure as WSKP (for now...)
        # inputs: 1-spp TA irr map (3 channel), 1-spp non-TA irr map (3 channel), albedo (3 channel), normals (3 channel), positions (3 channel), depth (1 channel) (16 total input channels)

        self.device = device
        self.deployed = deployed
        self.pretrained_model = pretrained_model

        self.base_depth = 14
        self.num_input_channels = 13
        self.num_output_channels = 1

        self.ss_non_ta = ss_non_ta
        self.sr_non_ta = sr_non_ta
        self.ss_ta = ss_ta
        self.sr_ta = sr_ta

        #debug
        self.x_final_out = None
        self.guide1 = None
        self.guide2 = None
        self.weight_map1 = None
        self.grid1 = None
        self.grid2 = None
        self.irradiance_non_TA_denoised = None
        self.irradiance_TA_denoised = None
        self.irradiance_final = None
        self.final = None

        self.adaptive_fusion_conv = AdaptiveFusion(device).to(device)
        self.weight_cnn = WeightCNN(device, pretrained_model, state_dict=state_dict_preload, deployed=deployed).to(device)

        self.downsampler = torch.nn.AvgPool2d(2, stride = 2)
        self.upsampler = torch.nn.Upsample(scale_factor=2)

        torch.autograd.set_detect_anomaly(True)

    def deploy(self):
        self.deployed=True
        self.adaptive_fusion_conv.rep1_noisy.deployed=True
        self.adaptive_fusion_conv.rep1_aux.deployed=True
        self.adaptive_fusion_conv.rep2.deployed=True
        self.adaptive_fusion_conv.rep3.deployed=True
        self.weight_cnn.rep1.deployed=True
        self.weight_cnn.rep1.deployed=True

    def init_deployed_conv_layers(self):
        self.weight_cnn.rep1.conv_layer_1 = reparametrize(self.weight_cnn.rep1.conv_layer1_1, self.weight_cnn.rep1.conv_layer1_3, self.weight_cnn.rep1.conv_layer1_5)
        self.weight_cnn.rep1.conv_layer_2 = reparametrize(self.weight_cnn.rep1.conv_layer2_1, self.weight_cnn.rep1.conv_layer2_3, self.weight_cnn.rep1.conv_layer2_5, has_identity_branch=False)

        self.adaptive_fusion_conv.rep1_aux.conv_layer_1 = reparametrize(self.adaptive_fusion_conv.rep1_aux.conv_layer1_1, self.adaptive_fusion_conv.rep1_aux.conv_layer1_3, self.adaptive_fusion_conv.rep1_aux.conv_layer1_5)
        self.adaptive_fusion_conv.rep1_aux.conv_layer_2 = reparametrize(self.adaptive_fusion_conv.rep1_aux.conv_layer2_1, self.adaptive_fusion_conv.rep1_aux.conv_layer2_3, self.adaptive_fusion_conv.rep1_aux.conv_layer2_5, has_identity_branch=True)
        
        self.adaptive_fusion_conv.rep1_noisy.conv_layer_1 = reparametrize(self.adaptive_fusion_conv.rep1_noisy.conv_layer1_1, self.adaptive_fusion_conv.rep1_noisy.conv_layer1_3, self.adaptive_fusion_conv.rep1_noisy.conv_layer1_5)
        self.adaptive_fusion_conv.rep1_noisy.conv_layer_2 = reparametrize(self.adaptive_fusion_conv.rep1_noisy.conv_layer2_1, self.adaptive_fusion_conv.rep1_noisy.conv_layer2_3, self.adaptive_fusion_conv.rep1_noisy.conv_layer2_5, has_identity_branch=True)

        self.adaptive_fusion_conv.rep2.conv_layer_1 = reparametrize(self.adaptive_fusion_conv.rep2.conv_layer1_1, self.adaptive_fusion_conv.rep2.conv_layer1_3, self.adaptive_fusion_conv.rep2.conv_layer1_5, has_identity_branch=True)
        self.adaptive_fusion_conv.rep2.conv_layer_2 = reparametrize(self.adaptive_fusion_conv.rep2.conv_layer2_1, self.adaptive_fusion_conv.rep2.conv_layer2_3, self.adaptive_fusion_conv.rep2.conv_layer2_5, has_identity_branch=True)

        self.adaptive_fusion_conv.rep3.conv_layer_1 = reparametrize(self.adaptive_fusion_conv.rep3.conv_layer1_1, self.adaptive_fusion_conv.rep3.conv_layer1_3, self.adaptive_fusion_conv.rep3.conv_layer1_5, has_identity_branch=True)
        self.adaptive_fusion_conv.rep3.conv_layer_2 = reparametrize(self.adaptive_fusion_conv.rep3.conv_layer2_1, self.adaptive_fusion_conv.rep3.conv_layer2_3, self.adaptive_fusion_conv.rep3.conv_layer2_5, has_identity_branch=True)        
    # @profile
    def forward(self, raw_inputs):
        x_irradiance_non_TA = raw_inputs[:, 0:3]
        x_irradiance_TA = raw_inputs[:, 3:6]
        x_albedo = raw_inputs[:, 6:9]

        x_tone_mapped_non_TA_frame = utils.tone_mapping(x_irradiance_non_TA * x_albedo)
        x_tone_mapped_TA_frame = utils.tone_mapping(x_irradiance_TA * x_albedo)


        # x_tone_mapped_TA_frame_copy = x_tone_mapped_TA_frame

        # alpha, one_minus_alpha = self.attention_layer(x_tone_mapped_non_TA_frame, x_tone_mapped_TA_frame, x_tone_mapped_TA_frame_copy)

        # x_input = torch.cat((x_tone_mapped_non_TA_frame, x_tone_mapped_TA_frame, raw_inputs[:, 6:self.num_input_channels]), axis=1)
        x_input = torch.cat((x_tone_mapped_non_TA_frame, x_tone_mapped_TA_frame, raw_inputs[:, 6:self.adaptive_fusion_conv.num_input_channels]), axis=1)

        kernel_non_TA, kernel_TA = self.adaptive_fusion_conv(x_input)
        #y_TA = y_TA[:, :3, :, :]
        #y_TA = y_TA.permute((0, 2, 3, 1))

        # x_albedo = x_albedo.permute((0, 2, 3, 1))
        # alpha = alpha.permute((0, 2, 3, 1))
        #return y_TA

        # x_input_non_TA = torch.cat((x_tone_mapped_non_TA_frame, raw_inputs[:, 6:16]), axis=1)
        # x_input_TA = torch.cat((x_tone_mapped_TA_frame, raw_inputs[:, 6:16]), axis=1)

        downsampled_irr_non_TA = self.downsampler(x_irradiance_non_TA)

        # if self.deployed:
        B,_,H_ta,W_ta = x_irradiance_TA.shape
        _,_,H_non_ta,W_non_ta = downsampled_irr_non_TA.shape
        kernels_num_channels=49
        kernels_size = 7

        TA_denoised = torch.empty((B, 3, H_ta, W_ta), dtype=x_irradiance_TA.dtype, device=self.device)
        Non_TA_denoised = torch.empty((B, 3, H_non_ta, W_non_ta), dtype=x_irradiance_non_TA.dtype, device=self.device)

        for i in range(3):
            TA_irr_channel = x_irradiance_TA[:, i:(i+1)]
            non_TA_irr_channel = downsampled_irr_non_TA[:, i:(i+1)]

            #irradiance_TA denoising
            TA_unfolded = torch.nn.functional.unfold(TA_irr_channel, kernel_size=kernels_size, padding=3, stride=1)
            kernel_TA_unfolded_format = kernel_TA.reshape(B, kernels_num_channels, H_ta * W_ta)
            TA_input_kernel_multipliaction = kernel_TA_unfolded_format * TA_unfolded
            TA_out_flat = TA_input_kernel_multipliaction.sum(dim=1, keepdim=True)
            TA_out = TA_out_flat.reshape(B, 1, H_ta, W_ta)
            TA_norm = kernel_TA.sum(dim=1, keepdim=True)
            TA_out = TA_out / (TA_norm + 0.000001)

            TA_denoised[:, i:(i+1)] = TA_out

            #irradiance_non_TA denoising
            non_TA_unfolded = torch.nn.functional.unfold(non_TA_irr_channel, kernel_size=kernels_size, padding=3, stride=1)
            kernel_non_TA_unfolded_format = kernel_non_TA.reshape(B, kernels_num_channels, H_non_ta * W_non_ta)
            Non_TA_input_kernel_multipliaction = kernel_non_TA_unfolded_format * non_TA_unfolded
            Non_TA_out_flat = Non_TA_input_kernel_multipliaction.sum(dim=1, keepdim=True)
            Non_TA_out = Non_TA_out_flat.reshape(B, 1, H_non_ta, W_non_ta)
            Non_TA_norm = kernel_non_TA.sum(dim=1, keepdim=True)
            Non_TA_out = Non_TA_out / (Non_TA_norm + 0.000001)

            Non_TA_denoised[:, i:(i+1)] = Non_TA_out
        # else:
            # custom cuda kernels
            # downsampled_irr_non_TA = torch.nn.functional.pad(downsampled_irr_non_TA, (3, 3, 3, 3))
            # x_irradiance_TA = torch.nn.functional.pad(x_irradiance_TA, (3, 3, 3, 3))
            # kernel_non_TA = torch.nn.functional.pad(kernel_non_TA, (3, 3, 3, 3))
            # kernel_TA = torch.nn.functional.pad(kernel_TA, (3, 3, 3, 3))
            # Non_TA_denoised = convolutional_ops.apply_kernel_maps_to_irradiance_maps(downsampled_irr_non_TA, kernel_non_TA)[:, :, 3:-3, 3:-3]
            # TA_denoised = convolutional_ops.apply_kernel_maps_to_irradiance_maps(x_irradiance_TA, kernel_TA)[:, :, 3:-3, 3:-3]

        Non_TA_denoised = self.upsampler(Non_TA_denoised)

        TA_denoised_tone_mapped = utils.tone_mapping(TA_denoised)
        Non_TA_denoised_tone_mapped = utils.tone_mapping(Non_TA_denoised)

        weight_input = torch.cat((Non_TA_denoised_tone_mapped, TA_denoised_tone_mapped), axis=1)

        weight = self.weight_cnn(weight_input)

        # print("min = ", torch.min(weight))
        # print("max = ", torch.max(weight))

        downsampled_TA_denoised = self.downsampler(TA_denoised)
        upsampled_TA_denoised = self.upsampler(downsampled_TA_denoised)

        # final = TA_denoised - weight * upsampled_TA_denoised + (weight * Non_TA_denoised)
        # final = TA_denoised - weight * upsampled_TA_denoised + (weight * Non_TA_denoised)
        # final = torch.clamp(TA_denoised - weight, min=0.0) * upsampled_TA_denoised + (weight * Non_TA_denoised)
        final_irradiance = torch.clamp(TA_denoised - weight * upsampled_TA_denoised, min=0.0) + (weight * Non_TA_denoised)
        final = x_albedo * final_irradiance
        # final = torch.clamp(TA_denoised - upsampled_TA_denoised, min=0.0) + torch.clamp(Non_TA_denoised, min=0.0)


        return final.permute((0, 2, 3, 1))
        # return TA_denoised.permute((0, 2, 3, 1))
        # return kernel_TA[:, 0:3].permute((0, 2, 3, 1))
        


'''
class RepVGG(torch.nn.Module):
    def __init__(self, device, deployed=False, pretrained_model=None, ss_non_ta=16, sr_non_ta=16, ss_ta=4, sr_ta=4):
        super().__init__()
        # define convolutional layers
        # 4 output channels (weight map 1 (1 channel), weight map 2 (1 channel), guide 1 (1 channel), guide 2 (1 channel))
        # same repvgg structure as WSKP (for now...)
        # inputs: 1-spp TA irr map (3 channel), 1-spp non-TA irr map (3 channel), albedo (3 channel), normals (3 channel), positions (3 channel), depth (1 channel) (16 total input channels)

        self.device = device
        self.deployed = deployed
        self.pretrained_model = pretrained_model

        self.base_depth = 14
        self.num_input_channels = 13
        self.num_output_channels = 1

        self.ss_non_ta = ss_non_ta
        self.sr_non_ta = sr_non_ta
        self.ss_ta = ss_ta
        self.sr_ta = sr_ta

        #debug
        self.x_final_out = None
        self.guide1 = None
        self.guide2 = None
        self.weight_map1 = None
        self.grid1 = None
        self.grid2 = None
        self.irradiance_non_TA_denoised = None
        self.irradiance_TA_denoised = None
        self.irradiance_final = None
        self.final = None

        if not deployed:
            # CNN 1
            self.conv_layer1_non_TA_1 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer1_non_TA_3 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer1_non_TA_5 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu1 = torch.nn.LeakyReLU(0.1)

            self.conv_layer2_non_TA_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer2_non_TA_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer2_non_TA_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu2 = torch.nn.LeakyReLU(0.1)

            self.conv_layer3_non_TA_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_input_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer3_non_TA_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer3_non_TA_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_input_channels, kernel_size=5, stride=1, padding=2, bias=False)
            # self.lrelu3 = torch.nn.LeakyReLU(0.1)
            self.sigmoid_non_TA = torch.nn.Sigmoid()

            # CNN 2
            self.conv_layer1_TA_1 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer1_TA_3 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer1_TA_5 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu4 = torch.nn.LeakyReLU(0.1)

            self.conv_layer2_TA_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer2_TA_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer2_TA_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu5 = torch.nn.LeakyReLU(0.1)

            self.conv_layer3_TA_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_output_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer3_TA_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_output_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer3_TA_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_output_channels, kernel_size=5, stride=1, padding=2, bias=False)
            self.sigmoid_TA = torch.nn.Sigmoid()

            # CNN 3 (weight prediction)
            self.conv_layer1_weight_1 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer1_weight_3 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer1_weight_5 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu6 = torch.nn.LeakyReLU(0.1)

            self.conv_layer2_weight_1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer2_weight_3 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer2_weight_5 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)
            self.sigmoid_weight = torch.nn.Sigmoid()

            self.downsampler = torch.nn.AvgPool2d(2, stride=2)
            self.upsampler = torch.nn.Upsample(size=(128, 128))

        else:
            self.rep_non_TA_conv_layer1 = reparametrize(pretrained_model.conv_layer1_non_TA_1, pretrained_model.conv_layer1_non_TA_3, pretrained_model.conv_layer1_non_TA_5)
            self.lrelu1 = torch.nn.LeakyReLU(0.1)

            self.rep_non_TA_conv_layer2 = reparametrize(pretrained_model.conv_layer2_non_TA_1, pretrained_model.conv_layer2_non_TA_3, pretrained_model.conv_layer2_non_TA_5, has_identity_branch=True)
            self.lrelu2 = torch.nn.LeakyReLU(0.1)

            self.rep_non_TA_conv_layer3 = reparametrize(pretrained_model.conv_layer3_non_TA_1, pretrained_model.conv_layer3_non_TA_3, pretrained_model.conv_layer3_non_TA_5)
            self.sigmoid_non_TA = torch.nn.Sigmoid()


            self.rep_TA_conv_layer1 = reparametrize(pretrained_model.conv_layer1_TA_1, pretrained_model.conv_layer1_TA_3, pretrained_model.conv_layer1_TA_5)
            self.lrelu4 = torch.nn.LeakyReLU(0.1)

            self.rep_TA_conv_layer2 = reparametrize(pretrained_model.conv_layer2_TA_1, pretrained_model.conv_layer2_TA_3, pretrained_model.conv_layer2_TA_5, has_identity_branch=True)
            self.lrelu5 = torch.nn.LeakyReLU(0.1)

            self.rep_TA_conv_layer3 = reparametrize(pretrained_model.conv_layer3_TA_1, pretrained_model.conv_layer3_TA_3, pretrained_model.conv_layer3_TA_5)
            self.sigmoid_TA = torch.nn.Sigmoid()


            self.rep_weight_conv_layer1 = reparametrize(pretrained_model.conv_layer1_weight_1, pretrained_model.conv_layer1_weight_3, pretrained_model.conv_layer1_weight_5)
            self.lrelu6 = torch.nn.LeakyReLU(0.1)

            self.rep_weight_conv_layer2 = reparametrize(pretrained_model.conv_layer2_weight_1, pretrained_model.conv_layer2_weight_3, pretrained_model.conv_layer2_weight_5)
            self.sigmoid_weight = torch.nn.Sigmoid()

            self.downsampler = torch.nn.AvgPool2d(2, stride=2)
            self.upsampler = torch.nn.Upsample(size=(720, 1280))

    # @profile
    def forward(self, raw_inputs):
        x_irradiance_non_TA = raw_inputs[:, 0:3]
        x_irradiance_TA = raw_inputs[:, 3:6]
        x_albedo = raw_inputs[:, 6:9]

        x_tone_mapped_non_TA_frame = utils.tone_mapping(x_irradiance_non_TA * x_albedo)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 2)
        # pyexr.write(os.path.join('x_tone_mapped_non_TA_frame_normal.exr'), x_tone_mapped_non_TA_frame[0, :].cpu().numpy())
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(2, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 2)
        # pyexr.write(os.path.join('x_tone_mapped_non_TA_frame_downsampled_padded.exr'), x_tone_mapped_non_TA_frame[0, :].cpu().numpy())
        # sys.exit(1)

        # x_tone_mapped_non_TA_frame = self.downsampler(x_tone_mapped_non_TA_frame)
        x_tone_mapped_TA_frame = utils.tone_mapping(x_irradiance_TA * x_albedo)

        x_input_non_TA = torch.cat((x_tone_mapped_non_TA_frame, raw_inputs[:, 6:16]), axis=1)
        x_input_TA = torch.cat((x_tone_mapped_TA_frame, raw_inputs[:, 6:16]), axis=1)

        if not self.deployed:
            # CNN 1 (non TA)
            x1_non_TA_1 = self.conv_layer1_non_TA_1(x_input_non_TA)
            x1_non_TA_3 = self.conv_layer1_non_TA_3(x_input_non_TA)
            x1_non_TA_5 = self.conv_layer1_non_TA_5(x_input_non_TA)
            x1_non_TA_out = self.lrelu1(x1_non_TA_1 + x1_non_TA_3 + x1_non_TA_5)

            x2_non_TA_1 = self.conv_layer2_non_TA_1(x1_non_TA_out)
            x2_non_TA_3 = self.conv_layer2_non_TA_3(x1_non_TA_out)
            x2_non_TA_5 = self.conv_layer2_non_TA_5(x1_non_TA_out)
            x2_non_TA_out = self.lrelu2(x2_non_TA_1 + x2_non_TA_3 + x2_non_TA_5 + x1_non_TA_out)

            x3_non_TA_1 = self.conv_layer3_non_TA_1(x2_non_TA_out)
            x3_non_TA_3 = self.conv_layer3_non_TA_3(x2_non_TA_out)
            x3_non_TA_5 = self.conv_layer3_non_TA_5(x2_non_TA_out)
            x_final_non_TA_out = self.sigmoid_non_TA(x3_non_TA_1 + x3_non_TA_3 + x3_non_TA_5)

            x_final_non_TA_out = x_final_non_TA_out.transpose(1, 3)
            x_final_non_TA_out = x_final_non_TA_out.transpose(1, 2)

            # CNN 2 (TA)
            x1_TA_1 = self.conv_layer1_TA_1(x_input_TA)
            x1_TA_3 = self.conv_layer1_TA_3(x_input_TA)
            x1_TA_5 = self.conv_layer1_TA_5(x_input_TA)
            x1_TA_out = self.lrelu4(x1_TA_1 + x1_TA_3 + x1_TA_5)

            x2_TA_1 = self.conv_layer2_TA_1(x1_TA_out)
            x2_TA_3 = self.conv_layer2_TA_3(x1_TA_out)
            x2_TA_5 = self.conv_layer2_TA_5(x1_TA_out)
            x2_TA_out = self.lrelu5(x2_TA_1 + x2_TA_3 + x2_TA_5 + x1_TA_out)

            x3_TA_1 = self.conv_layer3_TA_1(x2_TA_out)
            x3_TA_3 = self.conv_layer3_TA_3(x2_TA_out)
            x3_TA_5 = self.conv_layer3_TA_5(x2_TA_out)
            self.x_final_TA_out = self.sigmoid_TA(x3_TA_1 + x3_TA_3 + x3_TA_5)

            self.x_final_TA_out = self.x_final_TA_out.transpose(1, 3)
            self.x_final_TA_out = self.x_final_TA_out.transpose(1, 2)

            self.guide1 = x_final_non_TA_out[:, :, :, 0] * 255
            self.guide2 = self.x_final_TA_out[:, :, :, 0] * 255

            # self.weight_map1 = torch.unsqueeze(self.x_final_out[:, :, :, 2], dim=3)
            # self.weight_map1 = torch.tile(self.weight_map1, (1, 1, 1, 3))
            # weight_map2 = 1.0 - self.weight_map1

            attributes_grid_1 = torch.zeros((self.ss_non_ta, self.sr_non_ta), dtype=torch.int32, device=self.device)
            attributes_grid_2 = torch.zeros((self.ss_ta, self.sr_non_ta), dtype=torch.int32, device=self.device)

            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 3)
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 2)

            x_irradiance_TA = x_irradiance_TA.transpose(1, 3)
            x_irradiance_TA = x_irradiance_TA.transpose(1, 2)

            x_albedo = x_albedo.transpose(1, 3)
            x_albedo = x_albedo.transpose(1, 2)

            self.grid1 = bilateral_ops.creation_wrapper(x_irradiance_non_TA, self.guide1, attributes_grid_1)
            self.grid2 = bilateral_ops.creation_wrapper(x_irradiance_TA, self.guide2, attributes_grid_2)

            self.irradiance_non_TA_denoised = bilateral_ops.slicing_wrapper(self.grid1, self.guide1, attributes_grid_1)
            self.irradiance_TA_denoised = bilateral_ops.slicing_wrapper(self.grid2, self.guide2, attributes_grid_2)

            # CNN 3 (weight)
            irr_non_TA_tone_mapped = utils.tone_mapping(self.irradiance_non_TA_denoised)
            irr_TA_tone_mapped = utils.tone_mapping(self.irradiance_TA_denoised)
            x_input_weight = torch.cat((irr_non_TA_tone_mapped, irr_TA_tone_mapped), axis=3)
            x_input_weight = x_input_weight.transpose(1, 3)
            x_input_weight = x_input_weight.transpose(2, 3)

            x1_weight_1 = self.conv_layer1_weight_1(x_input_weight)
            x1_weight_3 = self.conv_layer1_weight_3(x_input_weight)
            x1_weight_5 = self.conv_layer1_weight_5(x_input_weight)
            x1_weight_out = self.lrelu6(x1_weight_1 + x1_weight_3 + x1_weight_5)

            x2_weight_1 = self.conv_layer2_weight_1(x1_weight_out)
            x2_weight_3 = self.conv_layer2_weight_3(x1_weight_out)
            x2_weight_5 = self.conv_layer2_weight_5(x1_weight_out)
            x2_final_weight_out = self.sigmoid_weight(x2_weight_1 + x2_weight_3 + x2_weight_5)
            x2_final_weight_out = x2_final_weight_out.transpose(1, 3)
            x2_final_weight_out = x2_final_weight_out.transpose(1, 2)

            weight_map = torch.unsqueeze(x2_final_weight_out[:, :, :, 0], dim=3)
            weight_map = torch.tile(weight_map, (1, 1, 1, 3))

            ### down sample and upsampling ###
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(2, 3)

            irradiance_non_TA_denoised_du = self.downsampler(self.irradiance_non_TA_denoised)
            irradiance_non_TA_denoised_du = self.upsampler(irradiance_non_TA_denoised_du)

            irradiance_non_TA_denoised_du = irradiance_non_TA_denoised_du.transpose(1, 3)
            irradiance_non_TA_denoised_du = irradiance_non_TA_denoised_du.transpose(1, 2)

            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 2)


            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 3)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(2, 3)

            irradiance_TA_denoised_du = self.downsampler(self.irradiance_TA_denoised)
            irradiance_TA_denoised_du = self.upsampler(irradiance_TA_denoised_du)

            irradiance_TA_denoised_du = irradiance_TA_denoised_du.transpose(1, 3)
            irradiance_TA_denoised_du = irradiance_TA_denoised_du.transpose(1, 2)

            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 3)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 2)
            ##############



            # self.irradiance_final = self.weight_map1 * self.irradiance_non_TA_denoised + weight_map2 * self.irradiance_TA_denoised
            # self.irradiance_final = self.irradiance_TA_denoised - (self.weight_map1 * du_irradiance_TA_denoised) + weight_map2 * self.irradiance_non_TA_denoised
            # pyexr.write(os.path.join('irradiance_final.exr'), self.irradiance_final[0, :].detach().cpu().numpy())

            self.irradiance_final = self.irradiance_TA_denoised - (weight_map * irradiance_TA_denoised_du) + (weight_map * irradiance_non_TA_denoised_du)

            self.final = x_albedo * self.irradiance_final
            # pyexr.write(os.path.join('final.exr'), self.final[0, :].detach().cpu().numpy())
            # sys.exit(1)

            return self.final

        else:
            # CNN 1
            x1_non_TA = self.rep_non_TA_conv_layer1(x_input_non_TA)
            x1_non_TA_out = self.lrelu1(x1_non_TA)

            x2_non_TA = self.rep_non_TA_conv_layer2(x1_non_TA_out)
            x2_non_TA_out = self.lrelu2(x2_non_TA)

            x3_non_TA = self.rep_non_TA_conv_layer3(x2_non_TA_out)
            x_final_non_TA_out = self.sigmoid_non_TA(x3_non_TA)

            x_final_non_TA_out = x_final_non_TA_out.transpose(1, 3)
            x_final_non_TA_out = x_final_non_TA_out.transpose(1, 2)

            # CNN 2
            x1_TA = self.rep_TA_conv_layer1(x_input_TA)
            x1_TA_out = self.lrelu4(x1_TA)

            x2_TA = self.rep_TA_conv_layer2(x1_TA_out)
            x2_TA_out = self.lrelu5(x2_TA)

            x3_TA = self.rep_TA_conv_layer3(x2_TA_out)
            x_final_TA_out = self.sigmoid_TA(x3_TA)

            x_final_TA_out = x_final_TA_out.transpose(1, 3)
            x_final_TA_out = x_final_TA_out.transpose(1, 2)

            self.guide1 = x_final_non_TA_out[:, :, :, 0] * 255
            self.guide2 = x_final_TA_out[:, :, :, 0] * 255

            attributes_grid_1 = torch.zeros((self.ss_non_ta, self.sr_non_ta), dtype=torch.int32, device=self.device)
            attributes_grid_2 = torch.zeros((self.ss_ta, self.sr_ta), dtype=torch.int32, device=self.device)

            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 3)
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 2)

            x_irradiance_TA = x_irradiance_TA.transpose(1, 3)
            x_irradiance_TA = x_irradiance_TA.transpose(1, 2)

            x_albedo = x_albedo.transpose(1, 3)
            x_albedo = x_albedo.transpose(1, 2)

            self.grid1 = bilateral_ops.creation_wrapper(x_irradiance_non_TA, self.guide1, attributes_grid_1)
            self.grid2 = bilateral_ops.creation_wrapper(x_irradiance_TA, self.guide2, attributes_grid_2)

            self.irradiance_non_TA_denoised = bilateral_ops.slicing_wrapper(self.grid1, self.guide1, attributes_grid_1)
            self.irradiance_TA_denoised = bilateral_ops.slicing_wrapper(self.grid2, self.guide2, attributes_grid_2)

            irr_non_TA_tone_mapped = utils.tone_mapping(self.irradiance_non_TA_denoised)
            irr_TA_tone_mapped = utils.tone_mapping(self.irradiance_TA_denoised)

            
            x_input_weight = torch.cat((irr_non_TA_tone_mapped, irr_TA_tone_mapped), axis=3)
            x_input_weight = x_input_weight.transpose(1, 3)
            x_input_weight = x_input_weight.transpose(2, 3)

            x1_weight = self.rep_weight_conv_layer1(x_input_weight)
            x1_weight_out = self.lrelu6(x1_weight)

            x2_weight = self.rep_weight_conv_layer2(x1_weight_out)
            x2_final_weight_out = self.sigmoid_weight(x2_weight)
            x2_final_weight_out = x2_final_weight_out.transpose(1, 3)
            x2_final_weight_out = x2_final_weight_out.transpose(1, 2)

            weight_map = torch.unsqueeze(x2_final_weight_out[:, :, :, 0], dim=3)
            weight_map = torch.tile(weight_map, (1, 1, 1, 3))

            ### down sample and upsampling ###
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(2, 3)

            irradiance_non_TA_denoised_du = self.downsampler(self.irradiance_non_TA_denoised)
            irradiance_non_TA_denoised_du = self.upsampler(irradiance_non_TA_denoised_du)

            irradiance_non_TA_denoised_du = irradiance_non_TA_denoised_du.transpose(1, 3)
            irradiance_non_TA_denoised_du = irradiance_non_TA_denoised_du.transpose(1, 2)

            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 2)


            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 3)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(2, 3)

            irradiance_TA_denoised_du = self.downsampler(self.irradiance_TA_denoised)
            irradiance_TA_denoised_du = self.upsampler(irradiance_TA_denoised_du)

            irradiance_TA_denoised_du = irradiance_TA_denoised_du.transpose(1, 3)
            irradiance_TA_denoised_du = irradiance_TA_denoised_du.transpose(1, 2)

            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 3)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 2)
            ##############

            self.irradiance_final = self.irradiance_TA_denoised - (weight_map * irradiance_TA_denoised_du) + (weight_map * irradiance_non_TA_denoised_du)

            self.final = x_albedo * self.irradiance_final

            return self.final
'''

'''
class RepVGG(torch.nn.Module):
    def __init__(self, device, deployed=False, pretrained_model=None, ss_non_ta=16, sr_non_ta=16, ss_ta=4, sr_ta=4):
        super().__init__()
        # define convolutional layers
        # 4 output channels (weight map 1 (1 channel), weight map 2 (1 channel), guide 1 (1 channel), guide 2 (1 channel))
        # same repvgg structure as WSKP (for now...)
        # inputs: 1-spp TA irr map (3 channel), 1-spp non-TA irr map (3 channel), albedo (3 channel), normals (3 channel), positions (3 channel), depth (1 channel) (16 total input channels)

        self.device = device
        self.deployed = deployed
        self.pretrained_model = pretrained_model

        self.base_depth = 14
        self.num_input_channels = 16
        self.num_output_channels = 3

        self.ss_non_ta = ss_non_ta
        self.sr_non_ta = sr_non_ta
        self.ss_ta = ss_ta
        self.sr_ta = sr_ta

        self.downsampler = torch.nn.AvgPool2d(2, stride=2)
        self.upsampler = torch.nn.Upsample(size=(128, 128), mode='bilinear')

        #debug
        self.x_final_out = None
        self.guide1 = None
        self.guide2 = None
        self.weight_map1 = None
        self.grid1 = None
        self.grid2 = None
        self.irradiance_non_TA_denoised = None
        self.irradiance_TA_denoised = None
        self.irradiance_final = None
        self.final = None

        if not deployed:
            self.conv_layer1_1 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer1_3 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer1_5 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu1 = torch.nn.LeakyReLU(0.1)

            self.conv_layer2_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer2_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer2_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu2 = torch.nn.LeakyReLU(0.1)

            self.conv_layer3_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer3_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer3_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu3 = torch.nn.LeakyReLU(0.1)

            self.conv_layer4_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer4_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer4_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu4 = torch.nn.LeakyReLU(0.1)

            self.conv_layer5_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layer5_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layer5_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.lrelu5 = torch.nn.LeakyReLU(0.1)

            self.conv_layerf_1 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_output_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_layerf_3 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_output_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_layerf_5 = torch.nn.Conv2d(in_channels=self.base_depth, out_channels=self.num_output_channels, kernel_size=5, stride=1, padding=2, bias=False)
            self.sigmoid = torch.nn.Sigmoid()

        else:
            self.rep_conv_layer1 = reparametrize(pretrained_model.conv_layer1_1, pretrained_model.conv_layer1_3, pretrained_model.conv_layer1_5)
            self.lrelu1 = torch.nn.LeakyReLU(0.1)

            self.rep_conv_layer2 = reparametrize(pretrained_model.conv_layer2_1, pretrained_model.conv_layer2_3, pretrained_model.conv_layer2_5, has_identity_branch=True)
            self.lrelu2 = torch.nn.LeakyReLU(0.1)

            self.rep_conv_layer3 = reparametrize(pretrained_model.conv_layer3_1, pretrained_model.conv_layer3_3, pretrained_model.conv_layer3_5, has_identity_branch=True)
            self.lrelu3 = torch.nn.LeakyReLU(0.1)

            self.rep_conv_layer4 = reparametrize(pretrained_model.conv_layer4_1, pretrained_model.conv_layer4_3, pretrained_model.conv_layer4_5, has_identity_branch=True)
            self.lrelu4 = torch.nn.LeakyReLU(0.1)

            self.rep_conv_layer5 = reparametrize(pretrained_model.conv_layer5_1, pretrained_model.conv_layer5_3, pretrained_model.conv_layer5_5, has_identity_branch=True)
            self.lrelu5 = torch.nn.LeakyReLU(0.1)

            self.rep_conv_layerf = reparametrize(pretrained_model.conv_layerf_1, pretrained_model.conv_layerf_3, pretrained_model.conv_layerf_5)
            self.sigmoid = torch.nn.Sigmoid()

    # @profile
    def forward(self, raw_inputs):
        x_irradiance_non_TA = raw_inputs[:, 0:3]
        x_irradiance_TA = raw_inputs[:, 3:6]
        x_albedo = raw_inputs[:, 6:9]

        x_tone_mapped_non_TA_frame = utils.tone_mapping(x_irradiance_non_TA * x_albedo)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 2)
        # pyexr.write(os.path.join('x_tone_mapped_non_TA_frame_normal.exr'), x_tone_mapped_non_TA_frame[0, :].cpu().numpy())
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(2, 3)
        x_tone_mapped_non_TA_frame = self.downsampler(x_tone_mapped_non_TA_frame)
        x_tone_mapped_non_TA_frame = torch.nn.functional.pad(x_tone_mapped_non_TA_frame, pad=(32, 32, 32, 32))
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 3)
        # x_tone_mapped_non_TA_frame = x_tone_mapped_non_TA_frame.transpose(1, 2)
        # pyexr.write(os.path.join('x_tone_mapped_non_TA_frame_downsampled_padded.exr'), x_tone_mapped_non_TA_frame[0, :].cpu().numpy())
        # sys.exit(1)

        x_tone_mapped_TA_frame = utils.tone_mapping(x_irradiance_TA * x_albedo)

        x_input = torch.cat((x_tone_mapped_non_TA_frame, x_tone_mapped_TA_frame, raw_inputs[:, 6:self.num_input_channels]), axis=1)

        if not self.deployed:
            x1_1 = self.conv_layer1_1(x_input)
            x1_3 = self.conv_layer1_3(x_input)
            x1_5 = self.conv_layer1_5(x_input)
            x1_out = self.lrelu1(x1_1 + x1_3 + x1_5)

            x2_1 = self.conv_layer2_1(x1_out)
            x2_3 = self.conv_layer2_3(x1_out)
            x2_5 = self.conv_layer2_5(x1_out)
            x2_out = self.lrelu2(x2_1 + x2_3 + x2_5 + x1_out)

            x3_1 = self.conv_layer3_1(x2_out)
            x3_3 = self.conv_layer3_3(x2_out)
            x3_5 = self.conv_layer3_5(x2_out)
            x3_out = self.lrelu3(x3_1 + x3_3 + x3_5 + x2_out)

            x4_1 = self.conv_layer4_1(x3_out)
            x4_3 = self.conv_layer4_3(x3_out)
            x4_5 = self.conv_layer4_5(x3_out)
            x4_out = self.lrelu4(x4_1 + x4_3 + x4_5 + x3_out)

            x5_1 = self.conv_layer5_1(x4_out)
            x5_3 = self.conv_layer5_3(x4_out)
            x5_5 = self.conv_layer5_5(x4_out)
            x5_out = self.lrelu5(x5_1 + x5_3 + x5_5 + x4_out)

            xf_1 = self.conv_layerf_1(x5_out)
            xf_3 = self.conv_layerf_3(x5_out)
            xf_5 = self.conv_layerf_5(x5_out)

            self.x_final_out = self.sigmoid(xf_1 + xf_3 + xf_5)

            self.x_final_out = self.x_final_out.transpose(1, 3)
            self.x_final_out = self.x_final_out.transpose(1, 2)

            self.guide1 = self.x_final_out[:, :, :, 0] * 255
            self.guide2 = self.x_final_out[:, :, :, 1] * 255

            self.weight_map1 = torch.unsqueeze(self.x_final_out[:, :, :, 2], dim=3)
            self.weight_map1 = torch.tile(self.weight_map1, (1, 1, 1, 3))
            weight_map2 = 1.0 - self.weight_map1

            attributes_grid_1 = torch.zeros((self.ss_non_ta, self.sr_non_ta), dtype=torch.int32, device=self.device)
            attributes_grid_2 = torch.zeros((self.ss_ta, self.sr_non_ta), dtype=torch.int32, device=self.device)

            x_irradiance_TA = x_irradiance_TA.transpose(1, 3)
            x_irradiance_TA = x_irradiance_TA.transpose(1, 2)

            x_albedo = x_albedo.transpose(1, 3)
            x_albedo = x_albedo.transpose(1, 2)

            ### DOWNSAMPLING ###
            x_irradiance_non_TA = self.downsampler(x_irradiance_non_TA)
            x_irradiance_non_TA = torch.nn.functional.pad(x_irradiance_non_TA, pad=(32, 32, 32, 32))
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 3)
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 2)
            ####################

            self.grid1 = bilateral_ops.creation_wrapper(x_irradiance_non_TA, self.guide1, attributes_grid_1)
            self.grid2 = bilateral_ops.creation_wrapper(x_irradiance_TA, self.guide2, attributes_grid_2)

            self.irradiance_non_TA_denoised = bilateral_ops.slicing_wrapper(self.grid1, self.guide1, attributes_grid_1)
            self.irradiance_TA_denoised = bilateral_ops.slicing_wrapper(self.grid2, self.guide2, attributes_grid_2)

            ### Cutting & upsampling ###
            height = self.irradiance_non_TA_denoised.shape[1]
            width = self.irradiance_non_TA_denoised.shape[2]
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(2, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised[:, :, 32 : height - 32, 32 : width - 32]
            self.irradiance_non_TA_denoised = self.upsampler(self.irradiance_non_TA_denoised)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 3)
            self.irradiance_non_TA_denoised = self.irradiance_non_TA_denoised.transpose(1, 2)
            ##############

            ### Disney Kernel Prediction Equation ###
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 3)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(2, 3)
            du_irradiance_TA_denoised = self.downsampler(self.irradiance_TA_denoised)
            du_irradiance_TA_denoised = self.upsampler(du_irradiance_TA_denoised)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 3)
            self.irradiance_TA_denoised = self.irradiance_TA_denoised.transpose(1, 2)
            du_irradiance_TA_denoised = du_irradiance_TA_denoised.transpose(1, 3)
            du_irradiance_TA_denoised = du_irradiance_TA_denoised.transpose(1, 2)
            ###

            # self.irradiance_final = self.weight_map1 * self.irradiance_non_TA_denoised + weight_map2 * self.irradiance_TA_denoised
            self.irradiance_final = self.irradiance_TA_denoised - (self.weight_map1 * du_irradiance_TA_denoised) + weight_map2 * self.irradiance_non_TA_denoised
            # pyexr.write(os.path.join('irradiance_final.exr'), self.irradiance_final[0, :].detach().cpu().numpy())

            self.final = x_albedo * self.irradiance_final
            # pyexr.write(os.path.join('final.exr'), self.final[0, :].detach().cpu().numpy())
            # sys.exit(1)

            return self.final

        else:
            x1 = self.rep_conv_layer1(x_input)
            x1_out = self.lrelu1(x1)

            x2 = self.rep_conv_layer2(x1_out)
            x2_out = self.lrelu2(x2)

            x3 = self.rep_conv_layer3(x2_out)
            x3_out = self.lrelu3(x3)

            x4 = self.rep_conv_layer4(x3_out)
            x4_out = self.lrelu4(x4)

            x5 = self.rep_conv_layer5(x4_out)
            x5_out = self.lrelu5(x5)

            xf = self.rep_conv_layerf(x5_out)

            self.x_final_out = self.sigmoid(xf)
            self.x_final_out = self.x_final_out.transpose(1, 3)
            self.x_final_out = self.x_final_out.transpose(1, 2)

            self.guide1 = self.x_final_out[:, :, :, 0] * 255
            self.guide2 = self.x_final_out[:, :, :, 1] * 255

            self.weight_map1 = torch.unsqueeze(self.x_final_out[:, :, :, 2], dim=3)
            self.weight_map1 = torch.tile(self.weight_map1, (1, 1, 1, 3))
            weight_map2 = 1.0 - self.weight_map1

            attributes_grid_1 = torch.zeros((self.ss_non_ta, self.sr_non_ta), dtype=torch.int32, device=self.device)
            attributes_grid_2 = torch.zeros((self.ss_ta, self.sr_ta), dtype=torch.int32, device=self.device)

            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 3)
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 2)

            x_irradiance_TA = x_irradiance_TA.transpose(1, 3)
            x_irradiance_TA = x_irradiance_TA.transpose(1, 2)

            x_albedo = x_albedo.transpose(1, 3)
            x_albedo = x_albedo.transpose(1, 2)

            self.grid1 = bilateral_ops.creation_wrapper(x_irradiance_non_TA, self.guide1, attributes_grid_1)
            self.grid2 = bilateral_ops.creation_wrapper(x_irradiance_TA, self.guide2, attributes_grid_2)

            self.irradiance_non_TA_denoised = bilateral_ops.slicing_wrapper(self.grid1, self.guide1, attributes_grid_1)
            self.irradiance_TA_denoised = bilateral_ops.slicing_wrapper(self.grid2, self.guide2, attributes_grid_2)

            self.irradiance_final = self.weight_map1 * self.irradiance_non_TA_denoised + weight_map2 * self.irradiance_TA_denoised

            self.final = x_albedo * self.irradiance_final

            return self.final
'''