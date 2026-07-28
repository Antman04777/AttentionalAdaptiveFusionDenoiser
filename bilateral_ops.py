import torch
torch.ops.load_library("build/libkernals_op.so")


def handle_creation(input_image, input_guide, attrs):
    #flatten inputs, check dimensions
    #call  creation wrapper op which callls launchers wrapper
    if (input_image.ndims() != 4):
      raise ValueError("Error: Input image dimension should be 4: {batch, height, width, channel}.")
    input_image = input_image.flat<float>()



#def handle_creation_grad(input_image, input_guide, attrs):
#        #flatten inputs, check dimensions
#    #call  creation grad wrapper op which calls the launchers wrapper


def handle_slicing(input_grid, input_guide, attrs):
    #flatten inputs, check dimensions
    #call  creation wrapper op which callls launchers wrapper



#def handle_slicing_grad(input_grid, input_guide_attrs):
#    #flatten inputs, check dimensions
#    #call  creation grad wrapper op which calls the launchers wrapper



@torch.library.custom_op("wrapper_ops::creation_wrapper_op", mutates_args=())
def creation_wrapper(input_image: torch.Tensor, input_guide: torch.Tensor, input_attrs: torch.Tensor) -> torch.Tensor:

    
    return torch.ops.my_ops.create_grid(input_image, input_guide,input_attrs)


def creation_back(ctx, grad):
    # get inputs into cration_wrapper from ctx
    #create grad tensors

    input_image,input_guide, input_attrs = ctx.saved_tensors

    #calls wrapper for grad launcher
    grad_tensors=torch.ops.my_ops.creation_grad(input_image, input_guide, input_attrs,grad)     #grad acts as backprop

    return grad_tensors[0], grad_tensors[1]

def setUp(ctx, inputs, output):
    input_image,input_guide, input_attrs = inputs
    ctx.save_for_backward(input_image,input_guide, input_attrs)