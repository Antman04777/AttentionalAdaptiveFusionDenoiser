import bilateral_ops
import torch


input_tensor = torch.rand((20,40,10,3), dtype=float, requires_grad=True, device='cuda')
guide_tensor = torch.rand((20,40,10,1), dtype = float, requires_grad=True, device='cuda')
attrs = torch.zeros((4,4))

bilateral_ops.creation_wrapper(input_tensor,guide_tensor,attrs)

print("Input: ", input_tensor.grad)
print("Guide: ", guide_tensor.grad)
print("Attrs: ", attrs.grad)