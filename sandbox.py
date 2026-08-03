from network import RepVGG
from dataloader import RawDataReader, BMFRFinalDataset
import numpy as np
import sys

import torch
from torch.profiler import profile, ProfilerActivity, record_function

import gc

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

    learning_rate = 1e-3
    batch_size = 64
    # batch_size = 10
    epochs = 10

    data = RawDataReader()
    train_dataset = BMFRFinalDataset(data, use_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    #input_tensor = torch.rand((5, 16, 10, 10), dtype=torch.float32, requires_grad=True, device=device)
    #ref_tensor = torch.rand((5, 10, 10, 3), dtype=torch.float32, requires_grad=False, device=device)

    model = RepVGG(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch in range(epochs):
        print(f'Epoch {epoch}')

        model.train()
        criterion = SMAPELoss().to(device)

        losses = []
        count = 1
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            for (inputs, targets) in train_dataloader:
                print("sub epoch: ", count)
                count += 1
                # torch.cuda.memory._record_memory_history()
                optimizer.zero_grad(set_to_none=True)
                # print(f"inputs before sending device: {inputs.device}")
                inputs = inputs.to(device, non_blocking=True)
                # print(f"inputs after sending device: {inputs.device}")
                targets = targets.to(device, non_blocking=True)
                targets = targets.transpose(1, 3)
                targets = targets.transpose(1, 2)
                #print("input shape: ", inputs.shape)
                #print("targets shape: ", targets.shape)

                outputs = model(inputs)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
                # sys.exit(0)
                # del outputs, loss
                
                # prof.step()

                # del outputs, loss

                # gc.collect()
                # torch.cuda.empty_cache()
        
        # print(prof.key_averages().table(row_limit=10))
        # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=40))
        print(f'Loss at epoch {epoch}: {np.mean(losses)}')

    #print("Outputs: ", outputs.shape)
    #print("Loss: ", loss)
    #print("Input grads: ", input_tensor.grad)