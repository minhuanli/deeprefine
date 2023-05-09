import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_split(size, ratio=0.8):
    choice = np.random.choice(size, int(ratio * size), replace=False)
    train_ind = np.zeros(size, dtype=bool)
    train_ind[choice] = True
    test_ind = ~train_ind
    return train_ind, test_ind


def linlogcut(x, a=0, b=1000):
    """Function which is linear until a, logarithmic until b and then constant.

    y = x                  x <= a
    y = a + log(x-a+1)   a < x < b
    y = a + log(b-a+1)   b < x

    """
    # cutoff x after b - this should also cutoff infinities
    x = torch.where(x < b, x, b * torch.ones_like(x))
    # log after a
    y = a + torch.where(x < a, x - a, torch.log(x - a + 1.0))
    # make sure everything is finite
    y = torch.where(torch.isfinite(y), y, b * torch.ones_like(y))
    return y
