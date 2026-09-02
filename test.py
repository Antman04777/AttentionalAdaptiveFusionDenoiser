import network
import utils
import torch
import numpy as np
import pyexr
import os
import dataloader


def test(model, device, dataloader, save_dir=''):
    model.eval()
    SSIMs = []
    PSNRs = []

    with torch.no_grad():
        for test_img_idx,(inputs, references) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)
            outputs = model(inputs).detach()
            # references = references.transpose(1, 3)
            # references = references.transpose(1, 2)

            output = outputs.cpu().numpy()[0]
            reference = references.cpu().numpy()[0].transpose((1, 2, 0))

            SSIM, PSNR = utils.compute_metrics(reference, output)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)   

            pyexr.write(os.path.join(save_dir, str(test_img_idx)+".exr"), output)

    print("Test: ")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    print("mean SSIM: ", SSIM_mean)
    print("mean PSNR: ", PSNR_mean)
    SSIMs.append(SSIM_mean)
    PSNRs.append(PSNR_mean)
    np.savetxt(os.path.join(save_dir, 'ssim.txt'), SSIMs, fmt="%s")
    np.savetxt(os.path.join(save_dir, 'psnr.txt'), PSNRs, fmt="%s")       

if __name__ == '__main__':

    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda")

    test_scene = "sponza"

    rawData = dataloader.RawDataReader(is_training=False, hold_out_scene=test_scene)
    dataset = dataloader.BMFRFinalDataset(rawData,use_test=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,num_workers=0,pin_memory=False)

    experiment = 'test-init'
    test_saving_dir = os.path.join("results",experiment, test_scene)
    for dir in [test_saving_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    pretrained_model = network.RepVGG(device).to(device)
    checkpoint = torch.load("checkpoints/test-init/sponza/model.pt", device)
    pretrained_model.load_state_dict(checkpoint["model_state_dict"])

    deployed_model = network.RepVGG(device,deployed=True,pretrained_model=pretrained_model)
    test(deployed_model,device,test_dataloader,test_saving_dir)


