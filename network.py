import torch
# import bilateral_ops
# import convolutional_ops
import utils
from torchvision.ops import Conv2dNormActivation
from typing import Dict

class RepVGGFastBlock(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.relu = torch.nn.ReLU(inplace=True)

class RepVGGBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
            activation_layer=None,
            # the original model may also have groups > 1
        )

        self.shortcut = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            activation_layer=None,
        )

        self.identity = (
            torch.nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        )

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        res = x  # <- 2x memory
        x = self.block(x)
        x += self.shortcut(res)
        if self.identity:
            x += self.identity(res)
        x = self.relu(x)  # <- 1x memory
        return x

    def to_fast(self) -> RepVGGFastBlock:
        fused_conv_state_dict = get_fused_conv_state_dict_from_block(self)
        fast_block = RepVGGFastBlock(
            self.block[0].in_channels,
            self.block[0].out_channels,
            stride=self.block[0].stride,
        )

        fast_block.conv.load_state_dict(fused_conv_state_dict)

        return fast_block

def get_fused_bn_to_conv_state_dict(
    conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d
) -> Dict[str, torch.Tensor]:
    
    bn_mean, bn_var, bn_gamma, bn_beta = (
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
    )
    bn_std = (bn_var + bn.eps).sqrt()

    conv_weight = torch.nn.Parameter((bn_gamma / bn_std).reshape(-1, 1, 1, 1) * conv.weight)

    conv_bias = torch.nn.Parameter(bn_beta - bn_mean * bn_gamma / bn_std)

    return {"weight": conv_weight, "bias": conv_bias}

def get_fused_conv_state_dict_from_block(block: RepVGGBlock) -> Dict[str, torch.Tensor]:
    fused_block_conv_state_dict = get_fused_bn_to_conv_state_dict(
        block.block[0], block.block[1]
    )

    if block.shortcut:
        # fuse the 1x1 shortcut
        conv_1x1_state_dict = get_fused_bn_to_conv_state_dict(
            block.shortcut[0], block.shortcut[1]
        )
        # we pad the 1x1 to a 3x3
        conv_1x1_state_dict["weight"] = torch.nn.functional.pad(
            conv_1x1_state_dict["weight"], [1, 1, 1, 1]
        )
        fused_block_conv_state_dict["weight"] += conv_1x1_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += conv_1x1_state_dict["bias"]
    if block.identity:
        # create our identity 3x3 conv kernel
        identify_conv = torch.nn.Conv2d(
            block.block[0].in_channels,
            block.block[0].in_channels,
            kernel_size=3,
            bias=True,
            padding=1,
        ).to(block.block[0].weight.device)
        # set them to zero!
        identify_conv.weight.zero_()
        # set the middle element to zero for the right channel
        in_channels = identify_conv.in_channels
        for i in range(identify_conv.in_channels):
            identify_conv.weight[i, i % in_channels, 1, 1] = 1
        # fuse the 3x3 identity
        identity_state_dict = get_fused_bn_to_conv_state_dict(
            identify_conv, block.identity
        )
        fused_block_conv_state_dict["weight"] += identity_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += identity_state_dict["bias"]

    fused_conv_state_dict = {
        k: torch.nn.Parameter(v) for k, v in fused_block_conv_state_dict.items()
    }

    return fused_conv_state_dict

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


class AFF(torch.nn.Module):
    def __init__(self,device,r=4,num_input_channels=3):
        super().__init__()

        self.num_input_channels=num_input_channels
        self.compressed_channels = int(num_input_channels/r)
        #self.global_context = SqueezeExcite(device,4,num_input_channels)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_g1 = torch.nn.Conv1d(num_input_channels, self.compressed_channels, 1, bias=False)
        self.bn_g1 = torch.nn.BatchNorm1d(self.compressed_channels)
        self.relu_g1 = torch.nn.ReLU() 
        self.conv_g2 = torch.nn.Conv1d(self.compressed_channels, 64, 1, bias=False)
        self.bn_g2 = torch.nn.BatchNorm1d(64)
        
        self.conv_l1 = torch.nn.Conv1d(num_input_channels, self.compressed_channels, 1, bias=False)
        self.bn_l1 = torch.nn.BatchNorm1d(self.compressed_channels)
        self.relu_l1 = torch.nn.ReLU() 
        self.conv_l2 = torch.nn.Conv1d(self.compressed_channels, 64, 1, bias=False)
        self.bn_l2 = torch.nn.BatchNorm1d(64)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):

        x_in = x + y

        #x_global = self.global_context(x_in)
#
        #b_in, c_in, h_in, w_in = x_in.shape
        #x_in = x_in.reshape(b_in, c_in, h_in * w_in)
        #x_local = self.conv1(x_in)
        #x_local = self.relu1(x_local)
        #x_local = self.conv2(x_local)
        #x_local = x_local.reshape(b_in, c_in, h_in, w_in)
#
        #x_context = x_global + x_local
        #x_context = self.sigmoid(x_context)

        b_in, c_in, h_in, w_in = x_in.shape

        #global
        x_g = self.avg_pool(x_in)
        x_g = x_g.reshape(b_in, c_in, 1)
        x_g = self.conv_g1(x_g)
        x_g = self.bn_g1(x_g)
        x_g = self.relu_g1(x_g)
        x_g = self.conv_g2(x_g)
        x_g = self.bn_g2(x_g)

        #local
        x_in = x_in.reshape(b_in, c_in, h_in * w_in)
        x_l = self.conv_l1(x_in)
        x_l = self.bn_l1(x_l)
        x_l = self.relu_l1(x_l)
        x_l = self.conv_l2(x_l)
        x_l = self.bn_l2(x_l)

        x_context = x_g + x_l
        x_context = self.sigmoid(x_context)
        x_context = x_context.reshape(b_in, c_in, h_in, w_in)

        x_out = (x_context * x) + ((1-x_context) * y)

        return x_out
    
class AdaptiveFusion(torch.nn.Module):
    def __init__(self, device, num_input_channels=16):
        super().__init__()

        self.device = device

        self.num_input_channels = num_input_channels

        # self.rep1 = RepBlock(device, in_ch=self.num_input_channels, out_ch=32, is_input=True, bn=True).to(device)
        self.rep1_noisy = RepVGGBlock(in_channels=6, out_channels=32).to(device)
        self.rep2_noisy = RepVGGBlock(in_channels=32, out_channels=32).to(device)
        self.squeeze_excite_layer1_noisy = SqueezeExcite(device, r=4, num_input_channels=32).to(device)

        self.rep1_aux = RepVGGBlock(in_channels=10, out_channels=32).to(device)
        self.rep2_aux = RepVGGBlock(in_channels=32, out_channels=32).to(device)
        self.squeeze_excite_layer1_aux = SqueezeExcite(device, r=4, num_input_channels=32).to(device)

        self.conv_layer_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_2 = torch.nn.BatchNorm2d(64)
        self.relu_2 = torch.nn.ReLU()

        # self.rep2 = RepBlock(device, in_ch=64, out_ch=64, bn=True).to(device)
        self.rep3 = RepVGGBlock(in_channels=64, out_channels=64).to(device)
        self.rep4 = RepVGGBlock(in_channels=64, out_channels=64).to(device)

        self.conv_layer_3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_3 = torch.nn.BatchNorm2d(128)
        self.relu_3 = torch.nn.ReLU()
        # self.drop1 = torch.nn.Dropout2d(0.15)

        # self.rep3 = RepBlock(device, in_ch=128, out_ch=128, bn=True).to(device)
        self.rep5 = RepVGGBlock(in_channels=128, out_channels=128).to(device)
        self.rep6 = RepVGGBlock(in_channels=128, out_channels=128).to(device)

        #for decoder
        self.squeeze_excite_layer2 = SqueezeExcite(device, r=4, num_input_channels=128).to(device)

        self.conv_layer_4_t = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, bias=False)

        self.conv_layer_5 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_4 = torch.nn.BatchNorm2d(64)
        self.relu_4 = torch.nn.ReLU()

        self.aff_1 = AFF(device,r=4,num_input_channels=64).to(device)
        self.aff_2 = AFF(device,r=4,num_input_channels=64).to(device)

        self.conv_layer_6 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_5 = torch.nn.BatchNorm2d(64)
        self.relu_5 = torch.nn.ReLU()
        # self.drop2 = torch.nn.Dropout2d(0.15)

        self.conv_layer_7_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False)

        self.conv_layer_8 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_6 = torch.nn.BatchNorm2d(64)
        self.relu_6 = torch.nn.ReLU()

        self.conv_layer_9 = torch.nn.Conv2d(in_channels=64, out_channels=49, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_7 = torch.nn.BatchNorm2d(49)
        self.relu_7 = torch.nn.ReLU()

    def deploy(self):
        self.rep1_noisy = self.rep1_noisy.to_fast().to(self.device)
        self.rep2_noisy = self.rep2_noisy.to_fast().to(self.device)
        self.rep1_aux = self.rep1_aux.to_fast().to(self.device)
        self.rep2_aux = self.rep2_aux.to_fast().to(self.device)
        self.rep3 = self.rep3.to_fast().to(self.device)
        self.rep4 = self.rep4.to_fast().to(self.device)
        self.rep5 = self.rep5.to_fast().to(self.device)
        self.rep6 = self.rep6.to_fast().to(self.device)

    def forward(self, x_input):
        x_input_noisy = x_input[:, 0:6]
        x_input_aux = x_input[:, 6:16]

        x_1_noisy = self.rep1_noisy(x_input_noisy)
        x_1_noisy = self.rep2_noisy(x_1_noisy)
        x_1_noisy = self.squeeze_excite_layer1_noisy(x_1_noisy)

        x_1_aux = self.rep1_aux(x_input_aux)
        x_1_aux = self.rep2_aux(x_1_aux)
        x_1_aux = self.squeeze_excite_layer1_aux(x_1_aux)

        x_embedded = torch.cat((x_1_noisy, x_1_aux), axis=1)  #res connection out

        x_2 = self.conv_layer_2(x_embedded)
        x_2 = self.bn_2(x_2)
        x_2 = self.relu_2(x_2)

        x_3 = self.rep3(x_2)
        x_3 = self.rep4(x_3)

        x_4 = self.conv_layer_3(x_3)
        x_4 = self.bn_3(x_4)
        x_4 = self.relu_3(x_4)
        # x_4 = self.drop1(x_4) # residual connection output

        x_5 = self.rep5(x_4)
        x_5 = self.rep6(x_5)
        x_5 = self.squeeze_excite_layer2(x_5)

        x_6 = self.conv_layer_4_t(x_5)

        x_7 = self.conv_layer_5(x_6)
        x_7 = self.bn_4(x_7)
        x_7 = self.relu_4(x_7)
        # x_7 = self.drop2(x_7) 

        x_8 = self.aff_1(x_2, x_7)           #order of entries matter
        #x_8 = torch.cat((x_2, x_7), axis=1)
        x_8 = self.conv_layer_6(x_8)
        x_8 = self.bn_5(x_8)
        x_8 = self.relu_5(x_8)
        k_non_TA = x_8[:,0:49]              #sliced first 49 channels for non ta kernel map

        x_9 = self.conv_layer_7_t(x_8)

        x_10 = self.conv_layer_8(x_9)
        x_10 = self.bn_6(x_10)
        x_10 = self.relu_6(x_10)

        # x_11_in = torch.cat((x_embedded, x_10), axis=1)
        x_11_in = self.aff_2(x_embedded, x_10)
        x_11 = self.conv_layer_9(x_11_in)
        x_11 = self.bn_7(x_11)
        x_11 = self.relu_7(x_11)
        k_TA = x_11

        return k_non_TA, k_TA


class WeightCNN(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.device = device

        self.conv_layer1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)

        self.rep1 = RepVGGBlock(in_channels=16, out_channels=16).to(device)
        self.rep2 = RepVGGBlock(in_channels=16, out_channels=16).to(device)
        self.rep3 = RepVGGBlock(in_channels=16, out_channels=16).to(device)
        self.rep4 = RepVGGBlock(in_channels=16, out_channels=16).to(device)

        self.conv_layer2 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def deploy(self):
        self.rep1 = self.rep1.to_fast().to(self.device)
        self.rep2 = self.rep2.to_fast().to(self.device)
        self.rep3 = self.rep3.to_fast().to(self.device)
        self.rep4 = self.rep4.to_fast().to(self.device)


    def forward(self, x_input):
        # two tone mapped irradiance maps concatenated

        x1 = self.conv_layer1(x_input)
        x2 = self.rep1(x1)
        x2 = self.rep2(x2)
        x2 = self.rep3(x2)
        x2 = self.rep4(x2)
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
        self.upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        torch.autograd.set_detect_anomaly(True)

    def deploy(self):
        self.adaptive_fusion_conv.deploy()
        self.weight_cnn.deploy()

    # @profile
    def forward(self, raw_inputs):
        x_irradiance_non_TA = raw_inputs[:, 0:3]
        x_irradiance_TA = raw_inputs[:, 3:6]
        x_albedo = raw_inputs[:, 6:9]

        #x_tone_mapped_non_TA_frame = utils.tone_mapping(x_irradiance_non_TA * x_albedo)
        x_tone_mapped_non_TA_frame = utils.tone_mapping(x_irradiance_non_TA)
        #x_tone_mapped_TA_frame = utils.tone_mapping(x_irradiance_TA * x_albedo)
        x_tone_mapped_TA_frame = utils.tone_mapping(x_irradiance_TA)


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

        kernel_TA = torch.exp(kernel_TA)
        kernel_TA = kernel_TA / kernel_TA.sum(dim=1, keepdim=True)

        kernel_non_TA = torch.exp(kernel_non_TA)
        kernel_non_TA = kernel_non_TA / kernel_non_TA.sum(dim=1, keepdim=True)

        for i in range(3):
            TA_irr_channel = x_irradiance_TA[:, i:(i+1)]
            non_TA_irr_channel = downsampled_irr_non_TA[:, i:(i+1)]

            #irradiance_TA denoising
            TA_unfolded = torch.nn.functional.unfold(TA_irr_channel, kernel_size=kernels_size, padding=3, stride=1)
            kernel_TA_unfolded_format = kernel_TA.reshape(B, kernels_num_channels, H_ta * W_ta)
            TA_input_kernel_multipliaction = kernel_TA_unfolded_format * TA_unfolded
            TA_out_flat = TA_input_kernel_multipliaction.sum(dim=1, keepdim=True)
            TA_out = TA_out_flat.reshape(B, 1, H_ta, W_ta)

            TA_denoised[:, i:(i+1)] = TA_out

            #irradiance_non_TA denoising
            non_TA_unfolded = torch.nn.functional.unfold(non_TA_irr_channel, kernel_size=kernels_size, padding=3, stride=1)
            kernel_non_TA_unfolded_format = kernel_non_TA.reshape(B, kernels_num_channels, H_non_ta * W_non_ta)
            Non_TA_input_kernel_multipliaction = kernel_non_TA_unfolded_format * non_TA_unfolded
            Non_TA_out_flat = Non_TA_input_kernel_multipliaction.sum(dim=1, keepdim=True)
            Non_TA_out = Non_TA_out_flat.reshape(B, 1, H_non_ta, W_non_ta)

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
        final_irradiance = TA_denoised - weight * upsampled_TA_denoised + (weight * Non_TA_denoised)
        final = x_albedo * final_irradiance
        # final = torch.clamp(TA_denoised - upsampled_TA_denoised, min=0.0) + torch.clamp(Non_TA_denoised, min=0.0)


        return final.permute((0, 2, 3, 1))
        # return TA_denoised.permute((0, 2, 3, 1))
        # return kernel_TA[:, 0:3].permute((0, 2, 3, 1))