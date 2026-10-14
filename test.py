import network
import utils
import torch
import numpy as np
import pyexr
import os
import dataloader
import statistics
import argparse
import sys

def test(model, device, dataloader, save_dir=''):
    model.eval()
    SSIMs = []
    PSNRs = []
    SMAPEs = []
    RMSEs = []

    with torch.inference_mode():
        for test_img_idx, (inputs, references) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)

            with torch.autocast(device_type="cuda"):
                outputs = model(inputs).detach()

            output = outputs.cpu().numpy()[0]
            reference = references.cpu().numpy()[0].transpose((1, 2, 0))

            SSIM, PSNR, SMAPE, RMSE = utils.compute_metrics(reference, output)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
            SMAPEs.append(SMAPE)
            RMSEs.append(RMSE)

            pyexr.write(os.path.join(save_dir, str(test_img_idx)+".exr"), output)

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

    with torch.inference_mode():
        for test_img_idx, (inputs, references) in enumerate(dataloader):
            cache_tensor.zero_() # perform cache flush for more accurate measurement

            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)

            start_events[test_img_idx].record()

            with torch.autocast(device_type="cuda"):
                _ = model(inputs)

            end_events[test_img_idx].record()
            torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        average_inference_time = statistics.mean(times)

        return average_inference_time

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == '__main__':
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    torch.set_float32_matmul_precision('high')

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

    experiment, test_scene = utils.load_test_parameters_from_json(parameters_path)

    rawdata = dataloader.RawDataReader(is_training=False, hold_out_scene=test_scene)
    dataset = dataloader.BMFRFinalDataset(rawdata, use_test=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

    test_saving_dir = os.path.join("results", experiment, test_scene)
    model_saving_dir = os.path.join("checkpoints", experiment, test_scene)
    for dir in [test_saving_dir, model_saving_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    checkpoint = torch.load(os.path.join(model_saving_dir, 'model.pt'), device)

    deployed_model = network.AttentionalAFNet(device).to(device)
    deployed_model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        deployed_model.deploy()

    deployed_model = torch.compile(deployed_model)

    SSIM_mean, PSNR_mean = test(deployed_model, device, test_dataloader, test_saving_dir)

    timing_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
    average_inference_time = time_inference(deployed_model, device, timing_dataloader, steps=rawdata.num_frames_per_scene)
    print(f'Average inference time = {average_inference_time}')

    utils.record_test_results(test_scene, experiment, device_name, PSNR_mean, SSIM_mean, average_inference_time, test_saving_dir)