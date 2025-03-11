import torch

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)