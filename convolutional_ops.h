#pragma once

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#endif

#include <torch/script.h>

void apply_kernel_maps_to_irradiance_maps_launcher
(
    const at::Half *input_irradiance,
	const at::Half *kernel_maps,
	at::Half *output,
    const int batch_size,
    const int channels,
    const int channels_km,
    const int height,
    const int width,
    const torch::Dtype type
);