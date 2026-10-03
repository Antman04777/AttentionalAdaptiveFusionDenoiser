import torch
import torchvision.transforms.functional as tf
import numpy as np
import os
import pyexr

if __name__ == '__main__':
    ref_location = os.path.join('dataset', 'san-miguel', 'inputs', 'reference30.exr')
    ref_image = pyexr.read(ref_location)

    ref_image_tensor = tf.to_tensor(ref_image.astype(np.float32))
    print(ref_image_tensor.shape)

    ref_image_freq = torch.fft.rfft2(ref_image_tensor)

    ref_image_back = torch.fft.irfft2(ref_image_freq)

    ref_image_freq_np = ref_image_back.cpu().numpy()

    print(ref_image_freq_np.shape)

    pyexr.write(os.path.join('reference30_freq.exr'), ref_image_freq_np.transpose((1, 2, 0)))