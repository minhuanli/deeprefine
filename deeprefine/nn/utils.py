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


def sigma_tilt(x, A=5, B=1e6, C=3e5, D=5, alpha=1.0):
    """
    Transform function for rugged loss function, adapted from 
    OpenMM loss for AF2 training, 10.1016/j.bpj.2023.12.011
    """
    return alpha * (1/(1/A + torch.exp(-(D*x + B)/C)) - A + 1)


def sigma_star_generator(alpha=1.0, uplimit=1e12, A=5, B=1e6, C=3e5, D=5):
    """
    Generate a transform function
    """
    def sigma_star(x):
       return alpha * (x / uplimit + sigma_tilt(x, A, B, C, D))
    return sigma_star
