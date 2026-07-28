#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

torch::Tensor create_grid
(
    const float* input_image,
    const float* input_guide, 
    const int batch_size,
    const float image_height 
    const float image_width,
    const int channel_size,
    const float grid_height,
    const float grid_width,
    const float grid_depth,
    float* const out_color);

torch::Tensor create_grid_grad
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
	float* const guide_grad
);