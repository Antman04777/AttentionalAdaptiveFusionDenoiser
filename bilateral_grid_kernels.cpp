#include <torch/torch.h>
#include <torch/script.h>
#include "bilateral_grid_kernels.h"

static torch::Tensor create_grid_wrapper(torch::Tensor input_image, torch::Tensor input_guide, torch::Tensor input_attrs){

    if (input_image.dim() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_flat_image = input_image.flatten<float>();

    if (input_guide.dim() != 3)
      throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide_flat = input_guide.flatten<float>();

    if (input_attrs.dim() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");
    const int ss = input_attrs.shape[0];
    const int sr = input_attrs.shape[1];

    const int batch_size = input_image.shape[0];
    const int image_height = input_image.shape[1];
    const int image_width = input_image.shape[2];
    const int channel_size = input_image.shape[3];

    const int grid_height = image_height / ss;
    const int grid_width = image_width / ss;
    const int grid_depth = 256 / sr;


}