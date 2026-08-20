import torch
import numpy as np
import bilateral_ops

from memory_profiler import profile

def reparametrize(conv1, conv3, conv5, has_identity_branch=False):
    reparam_kernel = conv5.weight + torch.nn.functional.pad(conv3.weight, [1, 1, 1, 1]) + torch.nn.functional.pad(conv1.weight, [2, 2, 2, 2])
    if has_identity_branch:
        identity_conv_weights = torch.eye(reparam_kernel.shape[0], device=conv1.weight.device).reshape((reparam_kernel.shape[0], reparam_kernel.shape[1], 1, 1))
        identity_conv_weights = torch.nn.functional.pad(identity_conv_weights, [2, 2, 2, 2])
        reparam_kernel += identity_conv_weights

    conv = torch.nn.Conv2d(in_channels=reparam_kernel.shape[1], out_channels=reparam_kernel.shape[0], kernel_size=5, stride=1, padding=2, bias=False)
    conv.weight.data = reparam_kernel.detach().cpu().clone()
    return conv

# def tone_mapping(self, input_image):
        # with tf.name_scope("tone_mapping"):
            # tone_mapped_color = tf.clip_by_value(
                # tf.math.pow(tf.math.maximum(0., input_image), 0.454545), 0., 1.)
            # return tone_mapped_color

def tone_map(input_image):
    if isinstance(input_image, np.ndarray):
        tone_mapped_image = np.clip(np.pow(np.maximum(input_image, 0.0), 0.454545), 0.0, 1.0)
        return tone_mapped_image
    elif isinstance(input_image, torch.Tensor):
        tone_mapped_image = torch.pow(torch.clamp(input_image, min=0.0, max=1.0), 0.454545)
        return tone_mapped_image

class RepVGG(torch.nn.Module):
    def __init__(self, device, deployed=False, pretrained_model=None):
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

        if not deployed:
            self.conv_layer1_1 = torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False) # bias false?
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
            self.rep_conv_layer1 = reparametrize(pretrained_model.conv_layer1_1, pretrained_model.conv_layer1_3, pretrained_model.conv_layer1_5, has_identity_branch=True)
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

        #print("Albedo shape: ", x_albedo.shape)
        #print("x_irradiance_non_TA shape: ", x_irradiance_non_TA.shape)
        #print("x_irradiance_TA shape: ", x_irradiance_TA.shape)

        x_tone_mapped_non_TA_frame = tone_map(x_irradiance_non_TA * x_albedo)
        x_tone_mapped_TA_frame = tone_map(x_irradiance_TA * x_albedo)

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

            x_final_out = self.sigmoid(xf_1 + xf_3 + xf_5)
            #
            #x_final_out = x_final_out.transpose((0, 2, 3, 1))
            #

            x_final_out = x_final_out.transpose(1, 3)
            x_final_out = x_final_out.transpose(1, 2)

            #guide1 = x_final_out[:, 0, :, :] * 255
            #guide2 = x_final_out[:, 1, :, :] * 255

            guide1 = x_final_out[:, :, :, 0] * 255
            guide2 = x_final_out[:, :, :, 1] * 255

            # weight_map1 = x_final_out[:, 2, :, :]
            weight_map1 = torch.unsqueeze(x_final_out[:, :, :, 2], dim=3)
            weight_map1 = torch.tile(weight_map1, (1, 1, 1, 3))
            weight_map2 = 1.0 - weight_map1

            ss = 4
            sr = 4

            attributes = torch.zeros((ss, sr), dtype=torch.int32, device=self.device)
            # x_irradiance_non_TA = x_irradiance_non_TA.transpose((0, 2, 3, 1))
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 3)
            x_irradiance_non_TA = x_irradiance_non_TA.transpose(1, 2)

            # x_irradiance_TA = x_irradiance_TA.transpose((0, 2, 3, 1))
            x_irradiance_TA = x_irradiance_TA.transpose(1, 3)
            x_irradiance_TA = x_irradiance_TA.transpose(1, 2)

            # x_albedo = x_albedo.transpose((0, 2, 3, 1))
            x_albedo = x_albedo.transpose(1, 3)
            x_albedo = x_albedo.transpose(1, 2)

            grid1 = bilateral_ops.creation_wrapper(x_irradiance_non_TA, guide1, attributes)
            grid2 = bilateral_ops.creation_wrapper(x_irradiance_TA, guide2, attributes)

            irradiance_non_TA_denoised = bilateral_ops.slicing_wrapper(grid1, guide1, attributes)
            irradiance_TA_denoised = bilateral_ops.slicing_wrapper(grid2, guide2, attributes)

            #print("guide1: ", guide1.shape)
            #print("guide2: ", guide2.shape)
            #print("weight_map1: ", weight_map1.shape)
            #print("weight_map2: ", weight_map2.shape)
            #print("grid1: ", grid1.shape)
            #print("grid2: ", grid2.shape)
            #print("irradiance_non_TA_denoised: ", irradiance_non_TA_denoised.shape)
            #print("irradiance_TA_denoised: ", irradiance_TA_denoised.shape)

            irradiance_final = weight_map1 * irradiance_non_TA_denoised + weight_map2 * irradiance_TA_denoised
            # irradiance_final = weight_map1 * x_irradiance_non_TA + weight_map2 * x_irradiance_TA

            final = x_albedo * irradiance_final

            return final