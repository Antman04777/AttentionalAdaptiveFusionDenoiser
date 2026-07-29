import bilateral_ops
import torch

input_tensor = torch.rand((20,40,10,3), dtype=torch.float32, requires_grad=True, device='cuda')
guide_tensor = torch.rand((20,40,10), dtype=torch.float32, requires_grad=True, device='cuda')
attrs = torch.zeros((4,4))

outputs = bilateral_ops.creation_wrapper(input_tensor,guide_tensor,attrs)
outputs.backward(torch.ones(outputs.shape))

print("Input: ", input_tensor.grad)
print("Guide: ", guide_tensor.grad)
print("Attrs: ", attrs.grad)