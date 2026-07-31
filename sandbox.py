from network import RepVGG

import torch

class SMAPELoss(torch.nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps=eps

    def forward(self, outputs, references):
        denominator = outputs
        loss = torch.mean(torch.abs(outputs - references) / (denominator.abs() + references.abs() + self.eps))
        return loss

if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    print('GPU: ', torch.cuda.get_device_name(0))

    input_tensor = torch.rand((5, 16, 10, 10), dtype=torch.float32, requires_grad=True, device=device)
    ref_tensor = torch.rand((5, 10, 10, 3), dtype=torch.float32, requires_grad=False, device=device)

    model = RepVGG(device).to(device)
    criterion = SMAPELoss().to(device)

    outputs = model(input_tensor)
    loss = criterion(outputs, ref_tensor)

    loss.backward()

    print("Outputs: ", outputs.shape)
    print("Loss: ", loss)
    print("Input grads: ", input_tensor.grad)