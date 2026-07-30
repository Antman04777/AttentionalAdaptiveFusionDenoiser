import torch
torch.ops.load_library("build/libbilateral_grid_kernels.so")


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

    
    return torch.ops.my_ops.create_grid_wrapper(input_image, input_guide,input_attrs)


def creation_back(ctx, grad):
    # get inputs into cration_wrapper from ctx
    #create grad tensors

    input_image,input_guide, input_attrs = ctx.saved_tensors

    #calls wrapper for grad launcher
    image_grad,guide_grad=torch.ops.my_ops.create_grid_grad_wrapper(input_image, input_guide, input_attrs, grad)     #grad acts as backprop

    return image_grad, guide_grad, None

def setUp(ctx, inputs, output):
    input_image, input_guide, input_attrs = inputs
    ctx.save_for_backward(input_image,input_guide, input_attrs)

creation_wrapper.register_autograd(creation_back, setup_context=setUp)