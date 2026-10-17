#include "convolutional_ops.h"

#include <torch/torch.h>

static torch::Tensor apply_kernel_maps_to_irradiance_maps_wrapper
(
    torch::Tensor input_irradiance,
    torch::Tensor kernel_maps
)
{
    if (input_irradiance.dim() != 4) throw std::invalid_argument("Error: Input irradiance dimension should be 4: {batch, channel, height, width}.");
    if (kernel_maps.dim() != 4) throw std::invalid_argument("Error: Kernel maps dimension should be 4: {batch, channel, height, width}.");

    const auto batch_size = input_irradiance.size(0);
    const auto channels = input_irradiance.size(1);
    const auto channels_km = kernel_maps.size(1);
    const auto height = input_irradiance.size(2);
    const auto width = input_irradiance.size(3);
    const auto type = input_irradiance.scalar_type();

    torch::Device device = input_irradiance.device();
    torch::Tensor output = torch::empty({batch_size, channels, height, width}, type);
    output = output.to(device);

    apply_kernel_maps_to_irradiance_maps_launcher
    (
        input_irradiance,
        kernel_maps,
        output
    );

    return output;
}

TORCH_LIBRARY (my_ops, m) {
    m.def("apply_kernel_maps_to_irradiance_maps_wrapper", &apply_kernel_maps_to_irradiance_maps_wrapper);
}