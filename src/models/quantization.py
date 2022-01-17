import torch


t = torch.rand((1, 2))
qt = torch.quantize_per_tensor(t, 0.1, 0, torch.quint8)
dqt = torch.dequantize(qt)
print(t)
print(qt)
print(dqt)
