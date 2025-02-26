import torch

a = torch.Tensor([[1],[1]])
b = torch.Tensor([[1],[1]])
c = torch.Tensor([[1],[1]])
d = torch.Tensor([[1],[1]])


T = torch.Tensor([[a,b], [c,d]])

print(T.shape)