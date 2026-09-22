from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
import numpy as np
import pyexr
import os
from PIL import Image
import math

def tone_mapping(image):
    if isinstance(image, np.ndarray):
        return np.clip(np.power(np.maximum(image, 0.0), 0.454545), 0.0, 1.0)
    
def SMAPE_LOSS(outputs, references):
    denominator = outputs
    loss = np.mean(np.abs(outputs - references) / (np.abs(denominator) + np.abs(references) + 0.01))
    return loss

def compute_metrics(reference_image, test_output_image):
    # reference_image = tone_mapping(reference_image)
    # test_output_image = tone_mapping(test_output_image)

    max_1 = np.max(reference_image)
    max_2 = np.max(test_output_image)
    # upper = max(max_1, max_2)

    min_1 = np.min(reference_image)
    min_2 = np.min(test_output_image)
    # lower = min(min_1, min_2)

    # SSIM = structural_similarity(reference_image, test_output_image, channel_axis = 2, data_range=(upper - lower))
    SSIM = structural_similarity(reference_image, test_output_image, channel_axis = -1, data_range=reference_image.max() - reference_image.min(), gaussian_weights=True, K1=0.01, K2=0.03, sigma=1.5)
    # SSIM = structural_similarity(reference_image, test_output_image, channel_axis = 2)
    PSNR = peak_signal_noise_ratio(reference_image, test_output_image, data_range=reference_image.max() - reference_image.min())

    SMAPE = SMAPE_LOSS(test_output_image, reference_image)

    return SSIM, PSNR, SMAPE

total_SSIM = 0
total_PSNR = 0
total_SMAPE = 0
method = 'ONND'
for i in range(1, 61):
    if i < 10:
        ref = Image.open(os.path.join('dataset', 'sponza', 'Reference', f'Sponza-Reference-00{i}.png'))
        out = Image.open(os.path.join('dataset', 'sponza', method, f'Sponza-{method}-00{i}.png'))
    else:
        ref = Image.open(os.path.join('dataset', 'sponza', 'Reference', f'Sponza-Reference-0{i}.png'))
        out = Image.open(os.path.join('dataset', 'sponza', method, f'Sponza-{method}-0{i}.png'))

    ref = np.asarray(ref)
    out = np.asarray(out)

    SSIM, PSNR, SMAPE = compute_metrics(ref, out)
    total_SSIM += SSIM
    total_PSNR += PSNR
    total_SMAPE += SMAPE

total_SSIM /= 60
total_PSNR /= 60
total_SMAPE /= 60

print(f'SSIM = {total_SSIM}, PSNR = {total_PSNR}, SMAPE = {total_SMAPE}')

