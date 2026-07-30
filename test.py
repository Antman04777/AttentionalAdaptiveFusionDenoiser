import bilateral_ops
import torch

torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
print('GPU: ', torch.cuda.get_device_name(0))

input_tensor = torch.rand((20,40,10,3), dtype=torch.float32, requires_grad=True, device=device)
guide_tensor = torch.rand((20,40,10), dtype=torch.float32, requires_grad=True, device=device)
attrs = torch.zeros((4,4))

# input_grad_tensor = torch.zeros_like(input_tensor)
# guide_grad_tensor = torch.zeros_like(guide_tensor)

outputs = bilateral_ops.creation_wrapper(input_tensor,guide_tensor,attrs)

initial_grad = torch.ones(outputs.shape)
initial_grad = initial_grad.to(device)

outputs.backward(initial_grad)

print("Input: ", input_tensor.grad)
print("Guide: ", guide_tensor.grad)
print("Attrs: ", attrs.grad)