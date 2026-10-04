import torch
# import bilateral_ops
# import convolutional_ops
import utils
from torchvision.ops import Conv2dNormActivation
from typing import Dict


class AdaptiveFusion(torch.nn.Module):
    def __init__(self, device, num_input_channels=16):
        super().__init__()

        
        self.device = device

        self.num_input_channels = num_input_channels

        #self.attention_layer = AttentionModule(device, r=4, num_input_channels=64).to(device)

        self.conv_layer_1 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_1 = torch.nn.ReLU()

        self.conv_layer_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_2 = torch.nn.ReLU()

        self.conv_layer_3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu_3 = torch.nn.ReLU()

        self.conv_layer_4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu_4 = torch.nn.ReLU()

        self.conv_layer_5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_5 = torch.nn.ReLU()

        self.conv_layer_6_t = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, bias=False)

        self.conv_layer_7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_6 = torch.nn.ReLU()

        self.conv_layer_8 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_7 = torch.nn.ReLU()

        self.conv_layer_9_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False)

        self.conv_layer_10 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_8 = torch.nn.ReLU()

        self.conv_layer_11 = torch.nn.Conv2d(in_channels=96, out_channels=49, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_9 = torch.nn.ReLU()

    def forward(self, x_input):

        x_1 = self.conv_layer_1(x_input)
        x_1 = self.relu_1(x_1)

        x_2 = self.conv_layer_2(x_1)
        x_2 = self.relu_2(x_2) # residual connection output

        x_3 = self.conv_layer_3(x_2)
        x_3 = self.relu_3(x_3) # residual connection output

        x_4 = self.conv_layer_4(x_3)
        x_4 = self.relu_4(x_4)

        x_5 = self.conv_layer_5(x_4)
        x_5 = self.relu_5(x_5)

        x_6 = self.conv_layer_6_t(x_5)

        x_7 = self.conv_layer_7(x_6)
        x_7 = self.relu_6(x_7)

        # x_8_in = torch.add_(x_3, x_8_new)

        #alpha, one_minus_alpha = self.attention_layer(x_3, x_8_new)

        #x_8_in = alpha * x_3 + one_minus_alpha * x_8_new

        x_8_in = torch.cat((x_3, x_7), axis=1)
        x_8 = self.conv_layer_8(x_8_in)
        x_8 = self.relu_7(x_8)
        k_non_TA = x_8[:,0:49]              #sliced first 49 channels for non ta kernel map

        x_9 = self.conv_layer_9_t(x_8)

        x_10 = self.conv_layer_10(x_9)

        # x_11_in = torch.add_(x_2, x_10)
        x_11_in = torch.cat((x_2, x_10), axis=1)
        x_11 = self.conv_layer_11(x_11_in)
        x_11 = self.relu_9(x_11)
        k_TA = x_11

        return k_non_TA, k_TA


class WeightCNN(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.device = device

        self.conv_layer1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)

        self.rep1 = RepVGGBlock(in_channels=16, out_channels=16).to(device)
        self.rep2 = RepVGGBlock(in_channels=16, out_channels=16).to(device)

        self.conv_layer2 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def deploy(self):
        self.rep1 = self.rep1.to_fast().to(self.device)
        self.rep2 = self.rep2.to_fast().to(self.device)


    def forward(self, x_input):
        # two tone mapped irradiance maps concatenated

        x1 = self.conv_layer1(x_input)
        x2 = self.rep1(x1)
        x2 = self.rep2(x2)
        x3 = self.conv_layer2(x2)
        x3 = self.sigmoid(x3)

        return x3

class RepVGG(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.num_input_channels = 13

        self.adaptive_fusion_conv = AdaptiveFusion(device).to(device)
        self.weight_cnn = WeightCNN(device).to(device)

        self.downsampler = torch.nn.AvgPool2d(2, stride=2)
        self.upsampler = torch.nn.Upsample(scale_factor=2)

        torch.autograd.set_detect_anomaly(True)

    def deploy(self):
        self.adaptive_fusion_conv.deploy()
        self.weight_cnn.deploy()

    def forward(self, raw_inputs):
        x_irradiance_non_TA = raw_inputs[:, 0:3]
        x_irradiance_TA = raw_inputs[:, 3:6]
        x_albedo = raw_inputs[:, 6:9]

        x_tone_mapped_non_TA_frame = utils.tone_mapping(x_irradiance_non_TA)
        x_tone_mapped_TA_frame = utils.tone_mapping(x_irradiance_TA)


        x_input = torch.cat((x_tone_mapped_non_TA_frame, x_tone_mapped_TA_frame, raw_inputs[:, 6:self.adaptive_fusion_conv.num_input_channels]), axis=1)

        kernel_non_TA, kernel_TA = self.adaptive_fusion_conv(x_input)

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
            TA_out = TA_out / (TA_norm + 0.0001)

            TA_denoised[:, i:(i+1)] = TA_out

            #irradiance_non_TA denoising
            non_TA_unfolded = torch.nn.functional.unfold(non_TA_irr_channel, kernel_size=kernels_size, padding=3, stride=1)
            kernel_non_TA_unfolded_format = kernel_non_TA.reshape(B, kernels_num_channels, H_non_ta * W_non_ta)
            Non_TA_input_kernel_multipliaction = kernel_non_TA_unfolded_format * non_TA_unfolded
            Non_TA_out_flat = Non_TA_input_kernel_multipliaction.sum(dim=1, keepdim=True)
            Non_TA_out = Non_TA_out_flat.reshape(B, 1, H_non_ta, W_non_ta)
            Non_TA_norm = kernel_non_TA.sum(dim=1, keepdim=True)
            Non_TA_out = Non_TA_out / (Non_TA_norm + 0.0001)

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

        downsampled_TA_denoised = self.downsampler(TA_denoised)
        upsampled_TA_denoised = self.upsampler(downsampled_TA_denoised)

        final_irradiance = torch.clamp(TA_denoised - weight * upsampled_TA_denoised, min=0.0) + (weight * Non_TA_denoised)
        final = x_albedo * final_irradiance

        return final.permute((0, 2, 3, 1))