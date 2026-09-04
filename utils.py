import numpy as np
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import os

def tone_mapping(image):
    if isinstance(image, np.ndarray):
        return np.clip(np.power(np.maximum(image, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(image, torch.Tensor):
        return torch.pow(torch.clamp(image, min=0.0, max=1.0), 0.454545)


def compute_metrics(reference_image, test_output_image):
    reference_image = tone_mapping(reference_image)
    test_output_image = tone_mapping(test_output_image)

    max_1 = np.max(reference_image)
    max_2 = np.max(test_output_image)
    upper = max(max_1, max_2)

    min_1 = np.min(reference_image)
    min_2 = np.min(test_output_image)
    lower = min(min_1, min_2)

    SSIM = structural_similarity(reference_image, test_output_image, channel_axis = 2, data_range=(upper - lower))
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

def record_train_parameters(lr, epochs, val_epoch_interval, batch_size, hold_off_scene, experiment_name, device_name, ss_ta, sr_ta, ss_non_ta, sr_non_ta, save_dir):
    parameters = f'''Learning Rate: {lr}
        Total Epochs: {epochs}
        Validation Frequency (Epochs): {val_epoch_interval}
        Batch Size: {batch_size}
        Hold Off Scene: {hold_off_scene}
        Experiment Name: {experiment_name}
        GPU Device Name: {device_name}
        SS TA: {ss_ta}
        SR TA: {sr_ta}
        SS Non-TA: {ss_non_ta}
        SR Non-TA: {sr_non_ta}'''

    parameters = parameters.replace('\t', '').replace('    ', '')

    file_save_path = os.path.join(save_dir, 'training_parameters.txt')

    with open(file_save_path, 'w') as f:
        f.write(parameters)