import dataloader
import pyexr
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def tone_mapping(image):
    if isinstance(image, np.ndarray):
        return np.clip(np.power(np.maximum(image, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(image, torch.Tensor):
        return torch.pow(torch.clamp(image, min=0.0, max=1.0), 0.454545)


def compute_metrics(reference_image, test_output_image):
    reference_image = tone_mapping(reference_image)
    test_output_image = tone_mapping(test_output_image)

    SSIM = structural_similarity(reference_image, test_output_image, channel_axis = 2)
    PSNR = peak_signal_noise_ratio(reference_image, test_output_image)

    return SSIM, PSNR

class SMAPEloss(nn.module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps=eps

    def forward(self, outputs, references):
        denominator = outputs
        loss = torch.mean(torch.abs(outputs - references) / (denominator.abs() + references.abs() + self.eps))
        return loss
    
def train(model, device, dataloader, optimizer, epoch, writer):
    model.train()
    losses=[]
    criterion = SMAPEloss.to(device)

    #Each iteration acts on a different batch
    for (inputs, references) in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device, non_blockking=True)
        references = references.to(device, non_blocking = True)
        outputs = model(inputs)
        loss = criterion(outputs,references)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    writer.add_scalar("Loss/total_train", np.mean(losses), epoch)
    print(np.mean(losses))


def inference(model, device, dataloader, dataset, save_dir=""):
    model.eval()
    SSIMs = []
    PSNRs = []

    with torch.no_grad:
        for image_index, (inputs, references) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)
            outputs = model(inputs).detach()

            if dataset.use_val:
                output = outputs.cpu().numpy()
                reference = references.cpu().numpy()

                for i in range(output.shape[0]):
                    if np.sum(reference[i]) == 0.0:
                        continue
                    SSIM, PSNR = compute_metrics(reference[i].transpose((1,2,0)), output[i].transpose((1,2,0)))
                    SSIMs.append(SSIM)
                    PSNRs.append(PSNR)

            elif dataset.use_test:
                output = outputs.cpu().numpy()[0].transpose((1,2,0)) #batch size is 1
                reference = references.cpu().numpy()[0].transpose((1,2,0))
                SSIM, PSNR = compute_metrics(reference[i].transpose((1,2,0)), output[i].transpose((1,2,0)))
                SSIMs.append(SSIM)
                PSNRs.append(PSNR)

            if dataset.use_test:
                pyexr.write(os.path.join(save_dir, str(image_index)+".exr"), output)

    if dataset.use_val:
        print("Validation: ")
    elif dataset.use_test:
        print("Test: ")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    print("mean SSIM: ", SSIM_mean)
    print("mean PSNR: ", PSNR_mean)
    SSIMs.append(SSIM_mean)
    PSNRs.append(PSNR_mean)
    if dataset.use_test:
        np.savetxt(os.path.join(save_dir, "SSIM.txt"), SSIMs, fmt = "%s")
        np.savetxt(os.path.join(save_dir, "SSIM.txt"), SSIMs, fmt = "%s")

    return SSIM_mean, PSNR_mean

if __name__ == "__main__":

    torch.cuda.set_device(1)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda")

    learning_rate = 1e-3 # BMFR
    epochs = 500
    epoch_test = 25
    batch_size = 64

    raw_data = dataloader.RawDataReader()
    dataset_train = dataloader.BMFRFinalDataset(raw_data, use_train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_val = dataloader.BMFRFinalDataset(raw_data, use_val=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_test = dataloader.BMFRFinalDataset(raw_data, use_test=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

 #   timestamp = "open-source-test"
    current_testing_scene_name = "sponza"
    tensorboard_saving_path = os.path.join("runs",  current_testing_scene_name)    #teimstep was used here
    test_saving_root = os.path.join("results", current_testing_scene_name) # and here
    model_saving_path = os.path.join("checkpoints", current_testing_scene_name)   #and here
    for folder in [tensorboard_saving_path, model_saving_path, test_saving_root]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    model = net.repGuidePredictionNet(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    writer = SummaryWriter(tensorboard_saving_path)


    for epoch in range(epochs):
        # Train
        train(model, device, dataloader_train, optimizer, epoch, writer)
        # Evaluation
        if (epoch+1) % epoch_test == 0:
            _SSIM_val, _PSNR_val = inference(model, device, dataloader_val, dataset_val)
            writer.add_scalar("SSIM-val", _SSIM_val, epoch)
            writer.add_scalar("PSNR-val", _PSNR_val, epoch)

            if epoch > epochs * 0.8:
                _SSIM_test, _PSNR_test = inference(model, device, dataloader_test, dataset_test, test_saving_root)
                writer.add_scalar("SSIM-test", _SSIM_test, epoch)
                writer.add_scalar("PSNR-test", _PSNR_test, epoch)                
            
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_saving_path, "model.pt"))

    writer.flush()
    writer.close()