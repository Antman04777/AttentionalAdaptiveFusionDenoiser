from dataloader import RawDataReader, BMFRFinalDataset, DatasetPreloader
from network import RepVGG
import pyexr
import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import utils
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import argparse
import sys
import json

import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from train import train
from torch.utils.tensorboard import SummaryWriter

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
            noisy = x[0] + checkpoint(self.attention, x[0], x[1])
        else:
            noisy = x[0] + self.attention(x[0], x[1])
        noisy = noisy + self.feed_forward(noisy)
        return (noisy, x[1])

def test(model, device, dataloader, save_dir=''):
    model.eval()
    SSIMs = []
    PSNRs = []
    SMAPEs = []
    RMSEs = []

    criterion = utils.SMAPEloss().to(device)
    criterion_RMSE = torch.nn.MSELoss()

    # with torch.no_grad():
    with torch.inference_mode():
        for test_img_idx, (inputs, references) in enumerate(dataloader):
            # inputs = inputs.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True).half()
            # references = references.to(device, non_blocking = True)
            references = references.to(device, non_blocking = True).half()

            #
            outputs = model(inputs).detach()
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
                # outputs = model(inputs).detach()
            #

            SMAPE = float(criterion(outputs,references.permute((0, 2, 3, 1))).item())
            RMSE = criterion_RMSE(outputs, references.permute((0, 2, 3, 1)))
            RMSE = torch.sqrt(RMSE)
            RMSE = float(RMSE.item())
#
            output = outputs.cpu().numpy()[0]
            reference = references.cpu().numpy()[0].transpose((1, 2, 0))
#
            SSIM, PSNR = utils.compute_metrics(reference, output)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
            SMAPEs.append(SMAPE)
            RMSEs.append(RMSE)

            pyexr.write(os.path.join(save_dir, str(test_img_idx)+".exr"), output)

    # irr_non_ta_denoised = model.irradiance_non_TA_denoised.cpu().numpy()[0]
    # irr_ta_denoised = model.irradiance_TA_denoised.cpu().numpy()[0]
    # irr_final = model.irradiance_final.cpu().numpy()[0]
    # pyexr.write(os.path.join(save_dir, 'irr_non_ta_denoised.exr'), irr_non_ta_denoised)
    # pyexr.write(os.path.join(save_dir, 'irr_ta_denoised.exr'), irr_ta_denoised)
    # pyexr.write(os.path.join(save_dir, 'irr_final.exr'), irr_final)

    print("Test: ")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    SMAPE_mean = np.mean(SMAPEs)
    RMSE_mean = np.mean(RMSEs)
    print("mean SSIM: ", SSIM_mean)
    print("mean PSNR: ", PSNR_mean)
    print("mean SMAPE: ", SMAPE_mean)
    print("mean RMSE: ", RMSE_mean)
    SSIMs.append(SSIM_mean)
    PSNRs.append(PSNR_mean)
    np.savetxt(os.path.join(save_dir, 'ssim.txt'), SSIMs, fmt="%s")
    np.savetxt(os.path.join(save_dir, 'psnr.txt'), PSNRs, fmt="%s")

    return SSIM_mean, PSNR_mean     

def time_inference(model, device, dataloader, steps=60):
    model.eval()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    cache_tensor = torch.empty(int(6 * (1024 ** 2)), dtype=torch.int8, device=device) # for the V100 GPU

    # dummy_tensor = torch.zeros((5, 16, 720, 1920)).to(device).half()
    # warm_up_reps = 2

    # with torch.no_grad():
    with torch.inference_mode():
        # for i in range(warm_up_reps):
            # _ = model(dummy_tensor)
        for test_img_idx, (inputs, references) in enumerate(dataloader):
            cache_tensor.zero_() # perform cache flush for more accurate measurement

            # inputs = inputs.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True).half()
            # references = references.to(device, non_blocking = True)
            references = references.to(device, non_blocking = True).half()

            start_events[test_img_idx].record()
            # outputs = model(inputs).detach()
            _ = model(inputs)
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
                # outputs = model(inputs).detach()
            end_events[test_img_idx].record()
            torch.cuda.synchronize()

        # torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        average_inference_time = statistics.mean(times)

        return average_inference_time


if __name__ == "__main__":

    is_train=True


    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)


    if(is_train):
        parser = argparse.ArgumentParser(prog='train')
        # parser.add_argument('--resume', type=str, help='<path_to_model.pt>')
        # parser.add_argument('--resume', action=argparse.BooleanOptionalAction, help='<path_to_model.pt>')
        parser.add_argument('--resume', action='store_true', help='<path_to_model.pt>')
        parser.add_argument('--parameters', type=str, help='<path_to_parameters.json>')

        args = parser.parse_args()
        # resuming_training = True if args.resume is not None else False
        resuming_training = args.resume
        # if resuming_training:
            # saved_model_path = args.resume

        if args.parameters is None:
            parser.print_help()
            sys.exit(1)

        if not os.path.exists(args.parameters):
            parser.print_help()
            sys.exit(1)

        parameters_path = args.parameters

        #experiment = "official-lr4-epoch-400-batchsize-20-ssta-4-srta-4-ssnta-16-srnta-16-l1_loss"
        #current_testing_scene_name = "sponza"
        #learning_rate = 1e-4  #og 1e-3
        #epochs = 400
        #epoch_validate = 25
        #batch_size = 20     #og 40

        experiment, current_testing_scene_name, learning_rate, loss_function, \
        epochs, epochs_validate, batch_size, previously_elapsed_epochs, \
        ss_non_ta, sr_non_ta, ss_ta, sr_ta = utils.load_train_parameters_from_json(parameters_path)

        raw_data = RawDataReader(is_training=True, hold_out_scene=current_testing_scene_name)
        dataset_train = BMFRFinalDataset(raw_data, use_train=True)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

        dataset_val = BMFRFinalDataset(raw_data, use_val=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        tensorboard_saving_dir = os.path.join("runs", experiment, current_testing_scene_name)
        model_saving_dir = os.path.join("checkpoints", experiment, current_testing_scene_name)

        for dir in [tensorboard_saving_dir, model_saving_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        model = TransformerBlock(ch=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        lower_bound = 0
        upper_bound = epochs

        if resuming_training:
            # checkpoint = torch.load(args.resume, device)
            checkpoint = torch.load(os.path.join(model_saving_dir, 'model.pt'))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            writer = SummaryWriter(log_dir=tensorboard_saving_dir, purge_step=previously_elapsed_epochs+1)
            lower_bound = previously_elapsed_epochs + 1
        else:
            writer = SummaryWriter(tensorboard_saving_dir)

        # model = torch.compile(model)
        utils.record_training_parameters(learning_rate, epochs, epochs_validate, batch_size, current_testing_scene_name, experiment, device_name, model.ss_ta, model.sr_ta, model.ss_non_ta, model.sr_non_ta, loss_function, model_saving_dir)

        for epoch in range(lower_bound, upper_bound):
            print(f'Epoch: {epoch}')
            
            train(model, device, dataloader_train, optimizer, epoch, writer, loss_func=loss_function)
            
            if (epoch + 1) % epochs_validate == 0:
                #ssim_val, psnr_val = validate(model, device, dataloader_val, loss_func=loss_function)
                #writer.add_scalar("SSIM-val", ssim_val, epoch)
                #writer.add_scalar("PSNR-val", psnr_val, epoch)
                
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_saving_dir, "model.pt"))

            with open(parameters_path, 'r+') as f:
                parameters_json = json.load(f)
                parameters_json['previously_elapsed_epochs'] += 1
                f.seek(0)
                json.dump(parameters_json, f, indent=4)
                f.truncate()
            
        writer.flush()
        writer.close()

    else:
        rawdata = RawDataReader(is_training=False, hold_out_scene="san-miguel")
        dataset = BMFRFinalDataset(rawdata, use_test=True)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

        parser = argparse.ArgumentParser(prog='train')
        parser.add_argument('--parameters', type=str, help='<path_to_parameters.json>')

        args = parser.parse_args()

        if args.parameters is None:
            parser.print_help()
            sys.exit(1)

        if not os.path.exists(args.parameters):
            parser.print_help()
            sys.exit(1)

        parameters_path = args.parameters

        experiment, test_scene, ss_non_ta, sr_non_ta, ss_ta, sr_ta = utils.load_test_parameters_from_json(parameters_path)

        rawdata = RawDataReader(is_training=False, hold_out_scene=test_scene)
        dataset = BMFRFinalDataset(rawdata, use_test=True)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

        test_saving_dir = os.path.join("results", experiment, test_scene)
        model_saving_dir = os.path.join("checkpoints", experiment, test_scene)
        for dir in [test_saving_dir, model_saving_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        checkpoint = torch.load(os.path.join(model_saving_dir, 'model.pt'), device)

        deployed_model = TransformerBlock(ch=32).to(device)


        deployed_model.load_state_dict(checkpoint['model_state_dict'])
        #deployed_model.deployed = True
        #deployed_model.init_deployed_conv_layers()
        deployed_model = deployed_model.half()
        deployed_model = torch.compile(deployed_model) # torch compile is in our favour now? lol

        SSIM_mean, PSNR_mean = test(deployed_model, device, test_dataloader, test_saving_dir)

        timing_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
        average_inference_time = time_inference(deployed_model, device, timing_dataloader, steps=rawdata.num_frames_per_scene)
        print(f'Average inference time = {average_inference_time}')

        utils.record_test_results(test_scene, experiment, device_name, ss_ta, sr_ta, ss_non_ta, sr_non_ta, PSNR_mean, SSIM_mean, average_inference_time, test_saving_dir)


