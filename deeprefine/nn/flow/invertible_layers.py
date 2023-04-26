import numpy as np
import torch

import torch.nn as nn
from deeprefine.nn.flow.basics import Flow, InverseFlow


def split_merge_indices(ndim, nchannels=2, channels=None):
    if channels is None:
        channels = np.tile(np.arange(nchannels), int(ndim / nchannels) + 1)[:ndim]
    else:
        channels = np.array(channels)
        nchannels = np.max(channels) + 1
    indices_split = []
    for c in range(nchannels):
        isplit = np.where(channels == c)[0]
        indices_split.append(isplit)
    indices_merge = np.concatenate(indices_split).argsort()
    return channels, indices_split, indices_merge


class SplitChannels(Flow):
    """Split channels forward and merge them backward

    Parameters
    ----------
    n_dim : int
        Number of features in the target axis
    nchannels : int, default 2
        Number of channels to split into
    split_dim : int, default -1
        Along which axis to perform the split and merge
    channels : None or list or array, default None
        Index of channel for each feature
    """

    def __init__(self, ndim, nchannels=2, split_dim=-1, channels=None):
        """Splits channels forward and merges them backward"""
        super().__init__()
        self.channels, self.indices_split, self.indices_merge = split_merge_indices(
            ndim, nchannels=nchannels, channels=channels
        )
        self._split_dim = split_dim

    def _forward(self, x):
        # split X into different coordinate channels
        self.output_z = [
            x[self._range(isplit, len(x.shape))] for isplit in self.indices_split
        ]
        return (*self.output_z, self._log_det_Jxz(x))

    def _inverse(self, z):
        x_scrambled = torch.concatenate(z, dim=self._split_dim)
        self.output_x = x_scrambled[
            self._range(self.indices_merge, len(x_scrambled.shape))
        ]
        return (self.output_x, self._log_det_Jzx(z))

    def _log_det_Jxz(self, x):
        index = [slice(None)] * len(x.shape)
        index[self._split_dim] = slice(1)
        return torch.zeros_like(x[index])

    def _log_det_Jzx(self, z):
        return self._log_det_Jxz(z[0])

    def _range(self, indices, n_dimensions):
        dims = [slice(None) for _ in range(n_dimensions)]
        dims[self._split_dim] = list(indices)
        return dims


class MergeChannels(InverseFlow):
    """Merge channels forward and split them backward
    defined as an inverse flow of SplitChannels

    Parameters
    ----------
    n_dim : int
        Number of features in the target axis
    nchannels : int, default 2
        Number of channels to split into
    split_dim : int, default -1
        Along which axis to perform the split and merge
    channels : None or list or array, default None
        Index of channel for each feature
    """
    def __init__(self, ndim, nchannels=2, split_dim=-1, channels=None):
        super().__init__(SplitChannels(ndim, nchannels, split_dim, channels))
