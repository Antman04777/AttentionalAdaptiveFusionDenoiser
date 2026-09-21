#include "convolution_functions.h"

#include <torch/torch.h>

static torch::Tensor apply_kernel_maps_to_irradiance_maps_wrapper
(
    torch::Tensor input_irradiance,
    torch::Tensor albedo,
    torch::Tensor kernel_maps
)
{
    if (input_irradiance.dim() != 4) throw std::invalid_argument("Error: Input irradiance dimension should be 4: {batch, channel, height, width}.");
    if (albedo.dim() != 4) throw std::invalid_argument("Error: Input albedo dimension should be 4: {batch, channel, height, width}.");
    if (kernel_maps.dim() != 4) throw std::invalid_argument("Error: Kernel maps dimension should be 4: {batch, channel, height, width}.");

    torch::Device device = input_irradiance.device();
    torch::Tensor output = torch::empty({batch_size, grid_height, grid_width, grid_depth, channel_size}, torch::kHalf);
    output = output.to(device);

    apply_kernel_maps_to_irradiance_maps_launcher
    (
        input_irradiance,
        albedo,
        kernel_maps,
        output
    );

    return output;
}

TORCH_LIBRARY (my_ops, m) {
    m.def("apply_kernel_maps_to_irradiance_maps_wrapper", &apply_kernel_maps_to_irradiance_maps_wrapper);
}