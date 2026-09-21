#pragma once

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#endif

#include <torch/script.h>

void apply_kernel_maps_to_irradiance_maps_launcher
(
    torch::Tensor input_irradiance,
    torch::Tensor kernel_maps,
    torch::Tensor output
);