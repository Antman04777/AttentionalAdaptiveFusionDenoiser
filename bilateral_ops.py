import torch

torch.ops.load_library("build/libbilateral_grid_kernels.so")
print("libbilateral_grid_kernels.so loaded")

#def handle_creation(input_image, input_guide, attrs):
    #flatten inputs, check dimensions
    #call  creation wrapper op which callls launchers wrapper




#def handle_creation_grad(input_image, input_guide, attrs):
#        #flatten inputs, check dimensions
#    #call  creation grad wrapper op which calls the launchers wrapper


#def handle_slicing(input_grid, input_guide, attrs):
    #flatten inputs, check dimensions
    #call  creation wrapper op which callls launchers wrapper



#def handle_slicing_grad(input_grid, input_guide_attrs):
#    #flatten inputs, check dimensions
#    #call  creation grad wrapper op which calls the launchers wrapper



@torch.library.custom_op("wrapper_ops::creation_wrapper_op", mutates_args=())
def creation_wrapper(input_image: torch.Tensor, input_guide: torch.Tensor, input_attrs: torch.Tensor) -> torch.Tensor:

    return torch.ops.my_ops.create_grid_wrapper(input_image, input_guide, input_attrs)

@creation_wrapper.register_fake
def _(x, y, z):
    ss = z.shape[0]
    sr = z.shape[1]
    grid_height = int(x.shape[1] / ss)
    grid_width = int(x.shape[2] / ss)
    grid_depth = int(256 / sr)
    batch_size = x.shape[0]
    channels = 3
    a = torch.empty((batch_size, grid_height, grid_width, grid_depth, channels), dtype=x.dtype, device=x.device)
    return torch.empty_like(a)

def creation_back(ctx, grad):
    # get inputs into cration_wrapper from ctx
    #create grad tensors

    input_image,input_guide, input_attrs = ctx.saved_tensors

    #calls wrapper for grad launcher
    image_grad,guide_grad=torch.ops.my_ops.create_grid_grad_wrapper(input_image, input_guide, input_attrs, grad)     #grad acts as backprop

    return image_grad, guide_grad, None

def setup_create(ctx, inputs, output):
    input_image, input_guide, input_attrs = inputs
    ctx.save_for_backward(input_image,input_guide, input_attrs)

creation_wrapper.register_autograd(creation_back, setup_context=setup_create)


@torch.library.custom_op("wrapper_ops::slicing_wrapper_op", mutates_args=())
def slicing_wrapper(input_grid: torch.Tensor, input_guide: torch.Tensor, input_attrs: torch.Tensor) -> torch.Tensor:

    return torch.ops.my_ops.slice_grid_wrapper(input_grid, input_guide, input_attrs)

@slicing_wrapper.register_fake
def _(x, y, z):
    image_height = y.shape[1]
    image_width = y.shape[2]
    batch_size = y.shape[0]
    channels = 3
    a = torch.empty((batch_size, image_height, image_width, channels), dtype=x.dtype, device=x.device)
    return torch.empty_like(a)

def slice_back(ctx, grad):
    # get inputs into cration_wrapper from ctx
    #create grad tensors

    input_grid, input_guide, input_attrs = ctx.saved_tensors

    #calls wrapper for grad launcher
    grid_grad, guide_grad = torch.ops.my_ops.slice_grid_grad_wrapper(input_grid, input_guide, input_attrs, grad)     #grad acts as backprop

    return grid_grad, guide_grad, None

def setup_slice(ctx, inputs, output):
    input_grid, input_guide, input_attrs = inputs
    ctx.save_for_backward(input_grid, input_guide, input_attrs)

slicing_wrapper.register_autograd(slice_back, setup_context=setup_slice)