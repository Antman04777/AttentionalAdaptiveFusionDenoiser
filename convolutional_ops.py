import torch

torch.ops.load_library("build/libconvolutional_ops.so")

print("libconvolutional_ops.so loaded")

apply_kernel_maps_to_irradiance_maps_launcher = torch.ops.my_ops.apply_kernel_maps_to_irradiance_maps_launcher