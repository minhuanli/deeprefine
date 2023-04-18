import numpy as np
import torch


def is_list_or_tuple(x):
    return isinstance(x, list) or isinstance(x, tuple)


def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x


def assert_tensor(x, arr_type=None):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=try_gpu())
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.tensor(x, device=try_gpu())
    assert isinstance(x, torch.tensor)
    if arr_type is not None:
        x = x.to(arr_type)
    return x
    

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


