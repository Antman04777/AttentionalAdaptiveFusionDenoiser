import torch

torch.ops.load_library("build/libconvolutional_ops.so")

print("libconvolutional_ops.so loaded")

# apply_kernel_maps_to_irradiance_maps_launcher = torch.ops.my_ops.apply_kernel_maps_to_irradiance_maps_wrapper

@torch.library.custom_op("convolution_ops::apply_kernel_maps_to_irradiance_maps_wrapper", mutates_args=())
def apply_kernel_maps_to_irradiance_maps(input_irradiance: torch.Tensor, kernel_maps: torch.Tensor) -> torch.Tensor:
    return torch.ops.my_ops.apply_kernel_maps_to_irradiance_maps_wrapper(input_irradiance, kernel_maps)

@apply_kernel_maps_to_irradiance_maps.register_fake
def _(x, y):
    B, C, H, W = x.shape
    a = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
    return torch.empty_like(a)