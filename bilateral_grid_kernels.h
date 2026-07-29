#pragma once

#ifndef TORCH_ROCM
#include <ATen/cuda/CUDAContext.h>
#else
#include <ATen/hip/HIPContext.h>
#endif

#include <torch/script.h>

void create_grid_kernel_launcher
(
    const float* input_image,
    const float* input_guide, 
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    float* const out_color,
    torch::Device& device,
    const torch::Dtype& type
);

void create_grid_grad_kernel_launcher
(
    const float* input_image,
	const float* input_guide,
    const float* grad,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* const image_grad,
	float* const guide_grad,
    torch::Device& device,
    const torch::Dtype& type
);
