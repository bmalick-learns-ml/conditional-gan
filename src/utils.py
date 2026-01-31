import torch


def one_hot(x, length=10):
    out = torch.zeros(length)
    out[x] = 1.
    return out