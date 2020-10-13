import torch
import torch.utils.data as data
torch.manual_seed(1)

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dateset = Data.TensorDataset(data)