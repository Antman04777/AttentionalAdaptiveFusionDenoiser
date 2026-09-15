#pragma once

#if defined(USE_ROCM) || defined(TORCH_USE_HIP_DSA)
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#endif

#include <torch/script.h>

#define TRAINING
// #undef TRAINING

#ifdef TRAINING
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
    float* out_color,
    torch::Device& device,
    const torch::Dtype& type
);
#else
void create_grid_kernel_launcher
(
    const at::Half* input_image,
    const at::Half* input_guide, 
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    at::Half* out_color,
    torch::Device& device,
    const torch::Dtype& type
);
#endif
// void create_grid_grad_kernel_launcher
// (
//     const float* input_image,
// 	const float* input_guide,
//     const float* backprop,
// 	const int batch_size,
// 	const int image_height,
// 	const int image_width,
// 	const int channel_size,
// 	const int grid_height, 
// 	const int grid_width, 
// 	const int grid_depth,
// 	const int sr,
// 	float* const image_grad,
// 	float* const guide_grad,
//     torch::Device& device,
//     const torch::Dtype& type
// );

void create_grid_grad_kernel_launcher
(
    const float* input_image,
	const float* input_guide,
    const float* backprop,
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

#ifdef TRAINING
void slice_grid_kernel_launcher
(
    const float* input_grid,
    const float* input_guide,
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    float* const output_image,
    torch::Device& device,
    const torch::Dtype& type
);
#else
void slice_grid_kernel_launcher
(
    const at::Half* input_grid,
    const at::Half* input_guide,
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    at::Half* const output_image,
    torch::Device& device,
    const torch::Dtype& type
);
#endif

void slice_grid_grad_kernel_launcher
(
	const float* input_grid,
    const float* input_guide,
	const float* backprop,
    const int batch_size,
    const int image_height,
    const int image_width,
    const int channel_size,
    const int grid_height,
    const int grid_width,
    const int grid_depth,
    const int sr,
    float* const grid_grad,
	float* const guide_grad,
    torch::Device& device,
    const torch::Dtype& type
);