import numpy as np
import torch

import torch.nn as nn
from deeprefine.nn.flow.basics import Flow, InverseFlow
from deeprefine.nn.basics import DenseNet


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
        return self.output_z, self._log_det_Jxz(x)

    def _inverse(self, z):
        x_scrambled = torch.concatenate(z, dim=self._split_dim)
        self.output_x = x_scrambled[
            self._range(self.indices_merge, len(x_scrambled.shape))
        ]
        return self.output_x, self._log_det_Jzx(z)

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


class RealNVP(Flow):
    def __init__(
        self,
        dim_L,
        dim_R,
        n_layers=2,
        n_hidden=100,
        activation="relu",
        n_layers_scale=None,
        n_hidden_scale=None,
        activation_scale="tanh",
        init_output_scale=0,
        **layer_args
    ):
        """Two sequential NVP transformations and their inverse transformatinos.

        Parameters
        ----------
        dim_L : int
            Number of features on the left channel
        dim_R : int
            Number of features on the right channel
        n_layers : int
            Number of hidden layers in the nonlinear transformations
        n_hidden : int
            Number of hidden units in each nonlinear layer
        activation : str
            Hidden-neuron activation functions used in the nonlinear layers
        n_layers_scale : int
            Number of hidden layers in the nonlinear transformations for the scaling network
            if None, will use n_layers
        n_hidden_scale : int
            Number of hidden units in each nonlinear layer for the scaling network.
            if None, will use n_hidden
        activation_scale : str
            Hidden-neuron activation functions used in scaling networks. If None, nl_activation will be used.
            if None, will use activation
        init_output_scale : float
            Initialize weights and bias of scaling networks to set the initial output value
        """
        super().__init__()
        if activation_scale is None:
            activation_scale = activation
        if n_layers_scale is None:
            n_layers_scale = n_layers
        if n_hidden_scale is None:
            n_hidden_scale = n_hidden

        self.S1 = DenseNet(
            dim_L,
            dim_R,
            nlayers=n_layers_scale,
            nhidden=n_hidden_scale,
            activation=activation_scale,
            init_outputs=init_output_scale,
            **layer_args
        )
        self.T1 = DenseNet(
            dim_L,
            dim_R,
            nlayers=n_layers,
            nhidden=n_hidden,
            activation=activation,
            **layer_args
        )
        self.S2 = DenseNet(
            dim_R,
            dim_L,
            nlayers=n_layers_scale,
            nhidden=n_hidden_scale,
            activation=activation_scale,
            init_outputs=init_output_scale,
            **layer_args
        )
        self.T2 = DenseNet(
            dim_R,
            dim_L,
            nlayers=n_layers,
            nhidden=n_hidden,
            activation=activation,
            **layer_args
        )

    def _forward(self, x):
        def lambda_sum(x):
            return torch.sum(x[0], dim=-1, keepdim=True) + torch.sum(
                x[1], dim=-1, keepdim=True
            )

        x1 = x[0]  # dim_L
        x2 = x[1]  # dim_R

        y1 = x1
        s1x1 = self.S1(x1)
        t1x1 = self.T1(x1)
        y2 = x2 * torch.exp(s1x1) + t1x1  # dim_R

        self.output_z2 = y2
        s2y2 = self.S2(y2)
        t2y2 = self.T2(y2)
        self.output_z1 = y1 * torch.exp(s2y2) + t2y2  # dim_L

        # log det(dz/dx)
        self.log_det_Jxz = lambda_sum([s1x1, s2y2])

        return [self.output_z1, self.output_z2], self.log_det_Jxz

    def _inverse(self, z):
        def lambda_negsum(x):
            return torch.sum(-x[0], dim=-1, keepdim=True) + torch.sum(
                -x[1], dim=-1, keepdim=True
            )

        z1 = z[0]  # dim_L
        z2 = z[1]  # dim_R

        y2 = z2
        s2z2 = self.S2(z2)
        t2z2 = self.T2(z2)
        y1 = (z1 - t2z2) * torch.exp(-s2z2)  # dim_L

        self.output_x1 = y1
        s1y1 = self.S1(y1)
        t1y1 = self.T1(y1)
        self.output_x2 = (y2 - t1y1) * torch.exp(-s1y1)  # dim_R

        # log det(dx/dz)
        self.log_det_Jzx = lambda_negsum([s2z2, s1y1])

        return [self.output_x1, self.output_x2], self.log_det_Jzx
