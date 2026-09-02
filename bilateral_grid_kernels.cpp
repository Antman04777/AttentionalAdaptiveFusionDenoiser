#include <torch/torch.h>
#include <torch/script.h>
#include "bilateral_grid_kernels.h"
#include <tuple>

static torch::Tensor create_grid_wrapper(torch::Tensor input_image, torch::Tensor input_guide, torch::Tensor input_attrs)
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

    torch::Device device = input_image.device();

    torch::Tensor output_color_tensor = torch::empty({batch_size, grid_height, grid_width, grid_depth, channel_size}, torch::kFloat);
    output_color_tensor = output_color_tensor.to(device);
    auto output_color = output_color_tensor.flatten();

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

    // std::cout << "output color: " << output_color[0] << '\n';

    // std::cout << output_color_tensor << '\n';

    // return output_color;
    return output_color_tensor;

//    torch::Tensor image_grad_tensor = torch::empty((batch_size, image_height, image_width, channel_size), dtype=torch::kFloat)
//    torch::Tensor guide_grad_tensor = torch::empty({batch_size, image_height, image_width}, dtype = torch::kFloat)
//
//    auto image_grad = image_grad_tensor.flatten<float>()
//    auto guide_grad = guide_grad_tensor.flatten<float>()

}

static std::tuple<torch::Tensor, torch::Tensor> create_grid_grad_wrapper(torch::Tensor input_image, torch::Tensor input_guide, torch::Tensor input_attrs, torch::Tensor grad){
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

    torch::Device device = input_image.device();

    torch::Tensor image_grad_tensor = torch::empty({batch_size, image_height, image_width, channel_size}, torch::kFloat);
    // torch::Tensor image_grad_tensor = torch::zeros_like(input_image);
    image_grad_tensor = image_grad_tensor.to(device);
    auto image_grad = image_grad_tensor.flatten();
    
    torch::Tensor guide_grad_tensor = torch::empty({batch_size, image_height, image_width}, torch::kFloat);
    // torch::Tensor guide_grad_tensor = torch::zeros_like(input_guide);
    guide_grad_tensor = guide_grad_tensor.to(device);
    auto guide_grad = guide_grad_tensor.flatten();

    auto backprop = grad.flatten();

    const auto& scalar_type = input_flat_image.scalar_type();

    create_grid_grad_kernel_launcher(
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

    // return std::forward_as_tuple(image_grad,guide_grad);
    return std::make_tuple(image_grad_tensor,guide_grad_tensor);
}

static torch::Tensor slice_grid_wrapper(torch::Tensor input_grid, torch::Tensor input_guide, torch::Tensor input_attrs)
{
    if (input_grid.dim() != 5) throw std::invalid_argument("Error: Input grid dimension should be 5: {batch, grid_height, grid_width, grid_depth, channel}.");
    if (input_guide.dim() != 3) throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    if (input_attrs.dim() != 2) throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");

    auto input_flat_grid = input_grid.flatten();
    auto input_flat_guide = input_guide.flatten();
    
    const int ss = input_attrs.size(0);
    const int sr = input_attrs.size(1);

    const int batch_size = input_grid.size(0);
    const int grid_height = input_grid.size(1);
    const int grid_width = input_grid.size(2);
    const int grid_depth = input_grid.size(3);
    const int channel_size = input_grid.size(4);

    const int image_height = input_guide.size(1);
    const int image_width = input_guide.size(2);

    torch::Device device = input_grid.device();

    torch::Tensor output_image_tensor = torch::empty({batch_size, image_height, image_width, channel_size}, torch::kFloat);
    output_image_tensor = output_image_tensor.to(device);
    auto output_image = output_image_tensor.flatten();

    const auto& scalar_type = input_flat_grid.scalar_type();

    slice_grid_kernel_launcher
    (
        input_flat_grid.const_data_ptr<float>(),
        input_flat_guide.const_data_ptr<float>(),
        batch_size,
        image_height,
        image_width,
        channel_size,
        grid_height,
        grid_width,
        grid_depth,
        sr,
        output_image.data_ptr<float>(),
        device,
        scalar_type
    );

    return output_image_tensor;
}

static std::tuple<torch::Tensor, torch::Tensor> slice_grid_grad_wrapper(torch::Tensor input_grid, torch::Tensor input_guide, torch::Tensor input_attrs, torch::Tensor grad)
{
    if (input_grid.dim() != 5) throw std::invalid_argument("Error: Input grid dimension should be 5: {batch, grid_height, grid_width, grid_depth, channel}.");
    if (input_guide.dim() != 3) throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    if (input_attrs.dim() != 2) throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");

    auto input_flat_grid = input_grid.flatten();
    auto input_flat_guide = input_guide.flatten();

    const int ss = input_attrs.size(0);
    const int sr = input_attrs.size(1);

    const int batch_size = input_grid.size(0);
    const int grid_height = input_grid.size(1);
    const int grid_width = input_grid.size(2);
    const int grid_depth = input_grid.size(3);
    const int channel_size = input_grid.size(4);

    const int image_height = input_guide.size(1);
    const int image_width = input_guide.size(2);

    torch::Device device = input_grid.device();

    torch::Tensor grid_grad_tensor = torch::empty({batch_size, grid_height, grid_width, grid_depth, channel_size}, torch::kFloat);
    grid_grad_tensor = grid_grad_tensor.to(device);
    auto grid_grad = grid_grad_tensor.flatten();

    torch::Tensor guide_grad_tensor = torch::empty({batch_size, image_height, image_width}, torch::kFloat);
    guide_grad_tensor = guide_grad_tensor.to(device);
    auto guide_grad = guide_grad_tensor.flatten();

    auto backprop = grad.flatten();

    const auto& scalar_type = input_flat_grid.scalar_type();

    slice_grid_grad_kernel_launcher
    (
	      input_flat_grid.const_data_ptr<float>(),
        input_flat_guide.const_data_ptr<float>(),
	      backprop.const_data_ptr<float>(),
        batch_size,
        image_height,
        image_width,
        channel_size,
        grid_height,
        grid_width,
        grid_depth,
        sr,
        grid_grad.data_ptr<float>(),
	      guide_grad.data_ptr<float>(),
        device,
        scalar_type
    );

    return std::make_tuple(grid_grad_tensor, guide_grad_tensor);
}

TORCH_LIBRARY (my_ops, m){
    m.def("create_grid_wrapper", &create_grid_wrapper);
    m.def("create_grid_grad_wrapper", &create_grid_grad_wrapper);
    m.def("slice_grid_wrapper", &slice_grid_wrapper);
    m.def("slice_grid_grad_wrapper", &slice_grid_grad_wrapper);
}