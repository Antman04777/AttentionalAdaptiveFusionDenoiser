from dataloader import RawDataReader, BMFRFinalDataset, DatasetPreloader
from network import RepVGG
# import pyexr
import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

    
def train(model, device, dataloader, optimizer, epoch, writer):
    model.train()
    losses=[]
    #SMAPE
    #criterion = utils.SMAPEloss().to(device)
    #l1_loss
    criterion = utils.l1_loss().to(device)

    preloader = DatasetPreloader(loader=dataloader, device=device)

    #Each iteration acts on a different batch
    # for (inputs, references) in dataloader:
    for (inputs, references) in preloader:
        optimizer.zero_grad()
        # optimizer.zero_grad(set_to_none=True)

        references = references.transpose(1, 3)
        references = references.transpose(1, 2)
        inputs = inputs.to(device, non_blocking=True)
        references = references.to(device, non_blocking=True)

        outputs = model(inputs)

        loss = criterion(outputs,references)
        loss.backward()

        #loss used by NBG
        #l1_loss = torch.mean(torch.abs(outputs-references))
        #l1_loss.backward()
        #losses.append(l1_loss.item())

        # print(model.final.grad) # gradients are of order 10^-5 - 10^-6
        # print(model.irradiance_final.grad) # gradients are of order 10^-6 - 10^-7
        # print(model.irradiance_TA_denoised.grad) # gradients are of order 10^-6 - 10^-7
        # print(model.irradiance_non_TA_denoised.grad) # gradients are of order 10^-6 - 10^-8
        # print(model.grid2.grad) # gradients are 0
        # print(model.grid1.grad) # gradients are 0
        # print(model.guide1.grad)
        #print(model.x_final_out.grad)

        optimizer.step()

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    writer.add_scalar("Loss/total_train", mean_loss, epoch)
    print(mean_loss)


def validate(model, device, dataloader):
    model.eval()
    SSIMs = []
    PSNRs = []
    val_losses = []
    #SMAPE
    #criterion = utils.SMAPEloss().to(device)
    #l1_loss
    criterion = utils.l1_loss().to(device)

    with torch.no_grad():
        for (inputs, references) in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)
            outputs = model(inputs).detach()
            references = references.transpose(1, 3)
            references = references.transpose(1, 2)

            loss = criterion(outputs,references)
            val_losses.append(loss.item())

            #loss used by NBG
            #l1_loss = torch.mean(torch.abs(outputs-references))
            #l1_loss.backward()
            #val_losses.append(l1_loss.item())

            output = outputs.cpu().numpy()
            reference = references.cpu().numpy()

            for i in range(output.shape[0]):
                if np.sum(reference[i]) == 0.0:
                    continue
                SSIM, PSNR = utils.compute_metrics(reference[i], output[i])
#                SSIM, PSNR = compute_metrics(reference[i].transpose((1,2,0)), output[i])
                SSIMs.append(SSIM)
                PSNRs.append(PSNR)


    print("Validation: ")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    print("mean SSIM: ", SSIM_mean)
    print("mean PSNR: ", PSNR_mean)

    mean_val_loss = np.mean(val_losses)
    writer.add_scalar("Loss/total_valiadtion", mean_val_loss, epoch)

    return SSIM_mean, PSNR_mean

if __name__ == "__main__":

    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    learning_rate = 1e-4  #og 1e-3
    epochs = 400
    epoch_validate = 25
    batch_size = 20     #og 40

    current_testing_scene_name = "sponza"

    raw_data = RawDataReader(is_training=True, hold_out_scene=current_testing_scene_name)
    dataset_train = BMFRFinalDataset(raw_data, use_train=True)
    # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True) // exceeded 120gb quota
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dataset_val = BMFRFinalDataset(raw_data, use_val=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    experiment = "official-lr4-epoch-400-batchsize-20-ssta-4-srta-4-ssnta-16-srnta-16-l1_loss"
    tensorboard_saving_dir = os.path.join("runs", experiment, current_testing_scene_name)
    model_saving_dir = os.path.join("checkpoints", experiment, current_testing_scene_name)

    for dir in [tensorboard_saving_dir, model_saving_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    model = RepVGG(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    writer = SummaryWriter(tensorboard_saving_dir)

    utils.record_training_parameters(learning_rate, epochs, epoch_validate, batch_size, current_testing_scene_name, experiment, device_name, model.ss_ta, model.sr_ta, model.ss_non_ta, model.sr_non_ta, model_saving_dir)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        
        train(model, device, dataloader_train, optimizer, epoch, writer)
        
        if (epoch + 1) % epoch_validate == 0:
            ssim_val, psnr_val = validate(model, device, dataloader_val)
            writer.add_scalar("SSIM-val", ssim_val, epoch)
            writer.add_scalar("PSNR-val", psnr_val, epoch)
            
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_saving_dir, "model.pt"))

    writer.flush()
    writer.close()