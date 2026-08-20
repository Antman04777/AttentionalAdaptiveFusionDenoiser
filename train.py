import dataloader
from network import RepVGG
# import pyexr
import os
import numpy as np

import torch
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

class SMAPEloss(torch.nn.Module):
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

    preloader = dataloader.DatasetPreloader(loader=dataloader, device=device)

    #Each iteration acts on a different batch
    # for (inputs, references) in dataloader:
    for (inputs, references) in preloader:
        optimizer.zero_grad()
        # optimizer.zero_grad(set_to_none=True)

        references = references.transpose(1, 3)
        references = references.transpose(1, 2)
        inputs = inputs.to(device, non_blockking=True)
        references = references.to(device, non_blocking = True)

        outputs = model(inputs)

        loss = criterion(outputs,references)
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    writer.add_scalar("Loss/total_train", mean_loss, epoch)
    print(mean_loss)


def validate(model, device, dataloader):
    model.eval()
    SSIMs = []
    PSNRs = []

    with torch.no_grad:
        for (inputs, references) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)
            outputs = model(inputs).detach()

            output = outputs.cpu().numpy()
            reference = references.cpu().numpy()

            for i in range(output.shape[0]):
                if np.sum(reference[i]) == 0.0:
                    continue
                SSIM, PSNR = compute_metrics(reference[i].transpose((1,2,0)), output[i])
                SSIMs.append(SSIM)
                PSNRs.append(PSNR)

    print("Validation: ")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    print("mean SSIM: ", SSIM_mean)
    print("mean PSNR: ", PSNR_mean)

    return SSIM_mean, PSNR_mean

if __name__ == "__main__":

    torch.cuda.set_device(1)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda")

    learning_rate = 1e-3
    epochs = 300
    epoch_validate = 25
    batch_size = 40

    raw_data = dataloader.RawDataReader()

    dataset_train = dataloader.BMFRFinalDataset(raw_data, use_train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)

    dataset_val = dataloader.BMFRFinalDataset(raw_data, use_val=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    experiment = "test-init"
    current_testing_scene_name = "sponza"
    tensorboard_saving_dir = os.path.join("runs", experiment, current_testing_scene_name)
    test_saving_dir = os.path.join("results", experiment, current_testing_scene_name)
    model_saving_dir = os.path.join("checkpoints", experiment, current_testing_scene_name)

    for dir in [tensorboard_saving_dir, model_saving_dir, test_saving_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    model = RepVGG(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    writer = SummaryWriter(tensorboard_saving_dir)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        
        train(model, device, dataloader_train, optimizer, epoch, writer)
        
        if (epoch + 1) % epoch_validate == 0:
            ssim_val, psnr_val = validate(model, device, dataloader_val, dataset_val)
            writer.add_scalar("SSIM-val", ssim_val, epoch)
            writer.add_scalar("PSNR-val", psnr_val, epoch)
            
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_saving_dir, "model.pt"))

    writer.flush()
    writer.close()