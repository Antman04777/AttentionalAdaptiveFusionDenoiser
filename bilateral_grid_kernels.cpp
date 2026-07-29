#include <torch/torch.h>
#include <torch/script.h>
#include "bilateral_grid_kernels.h"
#include <tuple>

static torch::Tensor create_grid_wrapper(torch::Tensor& input_image, torch::Tensor& input_guide, torch::Tensor& input_attrs)
{
    if (input_image.dim() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_flat_image = input_image.flatten();

    if (input_guide.dim() != 3)
      throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide_flat = input_guide.flatten();

    if (input_attrs.dim() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");
    const int ss = input_attrs.size(0);
    const int sr = input_attrs.size(1);

    const int batch_size = input_image.size(0);
    const int image_height = input_image.size(1);
    const int image_width = input_image.size(2);
    const int channel_size = input_image.size(3);

    const int grid_height = image_height / ss;
    const int grid_width = image_width / ss;
    const int grid_depth = 256 / sr;

    torch::Tensor output_color_tensor = torch::empty({batch_size, grid_height, grid_width, grid_depth, channel_size}, torch::kFloat);
    auto output_color = output_color_tensor.flatten();

    torch::Device device = input_image.device();
    const auto& scalar_type = input_flat_image.scalar_type();

    create_grid_kernel_launcher(
    input_flat_image.const_data_ptr<float>(),
    input_guide_flat.const_data_ptr<float>(), 
    batch_size,
    image_height,
    image_width,
    channel_size,
    grid_height,
    grid_width,
    grid_depth,
    sr,
    output_color.data_ptr<float>(),
    device,
    scalar_type);

    return output_color;

//    torch::Tensor image_grad_tensor = torch::empty((batch_size, image_height, image_width, channel_size), dtype=torch::kFloat)
//    torch::Tensor guide_grad_tensor = torch::empty({batch_size, image_height, image_width}, dtype = torch::kFloat)
//
//    auto image_grad = image_grad_tensor.flatten<float>()
//    auto guide_grad = guide_grad_tensor.flatten<float>()

}

static std::tuple<torch::Tensor,torch::Tensor> create_grid_grad_wrapper(torch::Tensor& input_image, torch::Tensor& input_guide, torch::Tensor& input_attr, torch::Tensor& grad){
    if (input_image.dim() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_flat_image = input_image.flatten();

    if (input_guide.dim() != 3)
      throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide_flat = input_guide.flatten();

    if (input_attrs.dim() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");
    const int ss = input_attrs.size(0);
    const int sr = input_attrs.size(1);

    const int batch_size = input_image.size(0);
    const int image_height = input_image.size(1);
    const int image_width = input_image.size(2);
    const int channel_size = input_image.size(3);

    const int grid_height = image_height / ss;
    const int grid_width = image_width / ss;
    const int grid_depth = 256 / sr;

    torch::Tensor image_grad_tensor = torch::empty({batch_size, image_height, image_width,channel_size}, torch::kFloat);
    auto image_grad = image_grad_tensor.flatten();   
    
    torch::Tensor guide_grad_tensor = torch::empty({batch_size, image_height, image_width}, torch::kFloat);
    auto guide_grad = guide_grad_tensor.flatten();

    auto backprop = grad.flatten();

    torch::Device device = input_image.device();
    const auto& scalar_type = input_flat_image.scalar_type();

    create_bi_grid_grad_kernel_launcher(
      input_flat_image.const_data_ptr<float>(), 
      input_guide_flat.const_data_ptr<float>(), 
      backprop.const_data_ptr<float>(),
      batch_size,
      image_height,
      image_width,
      channel_size,
      grid_height, 
      grid_width, 
      grid_depth, 
      sr,
      image_grad.data_ptr<float>(),
      guide_grad.data_ptr<float>(),
      device,
      scalar_type
    );

    return std::forward_as_tuple(image_grad,guide_grad)
}

TORCH_LIBRARY (my_ops, m){
    m.def("create_grid_wrapper", &create_grid_wrapper);
    m.def("create_grid_grad_wrapper", &create_grid_grad);
}