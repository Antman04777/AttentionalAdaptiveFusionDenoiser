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

import argparse
import sys
import json
    
def train(model, device, dataloader, optimizer, epoch, writer, loss_func='smape'):
    model.train()
    losses=[]

    criterion = None
    match loss_func:
        case 'smape':
            criterion = utils.SMAPEloss().to(device)
        case 'l1':
            criterion = utils.l1_loss().to(device) #loss used by NBG
        case _:
            print('Invalid loss function, defaulting to SMAPE loss')
            criterion = utils.SMAPEloss().to(device)

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

        optimizer.step()

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    writer.add_scalar("Loss/total_train", mean_loss, epoch)
    print(mean_loss)


def validate(model, device, dataloader, loss_func='smape'):
    model.eval()
    SSIMs = []
    PSNRs = []
    val_losses = []

    criterion = None
    match loss_func:
        case 'smape':
            criterion = utils.SMAPEloss().to(device)
        case 'l1':
            criterion = utils.l1_loss().to(device) # loss used by NBG
        case _:
            print('Invalid loss function, defaulting to SMAPE loss')
            criterion = utils.SMAPEloss().to(device)

    with torch.no_grad():
        for (inputs, references) in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            references = references.to(device, non_blocking = True)
            outputs = model(inputs).detach()
            references = references.transpose(1, 3)
            references = references.transpose(1, 2)

            loss = criterion(outputs,references)
            val_losses.append(loss.item())

            output = outputs.cpu().numpy()
            reference = references.cpu().numpy()

            for i in range(output.shape[0]):
                if np.sum(reference[i]) == 0.0:
                    continue
                SSIM, PSNR = utils.compute_metrics(reference[i], output[i])
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

    model = RepVGG(device, ss_non_ta=ss_non_ta, sr_non_ta=sr_non_ta, ss_ta=ss_ta, sr_ta=sr_ta).to(device)
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
            ssim_val, psnr_val = validate(model, device, dataloader_val, loss_func=loss_function)
            writer.add_scalar("SSIM-val", ssim_val, epoch)
            writer.add_scalar("PSNR-val", psnr_val, epoch)
            
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_saving_dir, "model.pt"))

        with open(parameters_path, 'r+') as f:
            parameters_json = json.load(f)
            parameters_json['previously_elapsed_epochs'] += 1
            f.seek(0)
            json.dump(parameters_json, f, indent=4)
            f.truncate()
        
    writer.flush()
    writer.close()