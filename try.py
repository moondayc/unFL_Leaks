import torch

a = torch.Tensor([1, 2, 3])
b = torch.Tensor([0, 0, 0])
c = torch.cat([a, b])
print(c)
print(c.data)
print(c.detach)

