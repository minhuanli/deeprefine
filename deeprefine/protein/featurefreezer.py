import torch
import numpy as np

from deeprefine.utils.types import try_gpu, assert_numpy, assert_tensor


class FeatureFreezer(object):
    """
    Freeze the features you don't want to send to the generative models by:
    1. kicking them out in the forward process
    2. filling them back with mean values in the backward process

    Support two modes to determine frozen features:
    1. Mannually input the feature indices, like you know which are bond length features
    2. Automatically freeze features with std/mean < 0.05, never kicking the cartesian signals

    Parameters:
    -----------
    X0 : Tensor, [n_batch, n_features]
        batch data to do statistics for mean

    frozen_indices : array, [n_frozen_features,] or None
        indices of features you want to freeze. If None, use automatic cutoff

    cutoff : float
        cutoff of std/mean to determine frozen features automatically
    """

    def __init__(self, X0, frozen_indices=None, cutoff=0.05, dim_cart_signal=None, from_dict=False):
        if from_dict:
            self.dim_in = None
            self.dim_out = None
            self.keep_idx = None
            self.frozen_idx = None
            self.freeze_mean = None
        else:
            self.dim_in = X0.shape[1]
            if frozen_indices is not None:
                self.frozen_idx = frozen_indices
                self.cutoff = None
            else:
                self.cutoff = cutoff
                if dim_cart_signal is not None:
                    abs_mean = torch.abs(torch.mean(X0[:, dim_cart_signal:], axis=0))
                    std = torch.std(X0[: dim_cart_signal:], axis=0)
                    self.frozen_idx = dim_cart_signal + assert_numpy(
                        torch.argwhere(std / abs_mean < cutoff).reshape(-1)
                    )
                else:
                    abs_mean = torch.abs(torch.mean(X0, axis=0))
                    std = torch.std(X0, axis=0)
                    self.frozen_idx = assert_numpy(
                        torch.argwhere(std / abs_mean < cutoff).reshape(-1)
                    )
            self.dim_out = self.dim_in - len(self.frozen_idx)
            self.keep_idx = np.setdiff1d(np.arange(self.dim_in), self.frozen_idx)
            self.freeze_mean = torch.mean(X0[:, self.frozen_idx], axis=0).to(try_gpu())

    def forward(self, X):
        """drop frozen features"""
        return X[:, self.keep_idx]
    
    def backward(self, Z):
        """fill in the frozen features with saved means"""
        X = torch.zeros([Z.shape[0], self.dim_in]).to(Z)
        X[:, self.keep_idx] = Z
        X[:, self.frozen_idx] = self.freeze_mean
        return X
    
    @classmethod
    def from_dict(cls, d):
        ff = cls(None, from_dict=True)
        ff.dim_in = d["dim_in"]
        ff.dim_out = d["dim_out"]
        ff.keep_idx = d["keep_idx"]
        ff.frozen_idx = d["frozen_idx"]
        ff.freeze_mean = assert_tensor(d["freeze_mean"], arr_type=torch.float32)
        return ff

    def to_dict(self):
        d = {}
        d["dim_in"] = self.dim_in
        d["dim_out"] = self.dim_out
        d["keep_idx"] = self.keep_idx
        d["frozen_idx"] = self.frozen_idx
        d["freeze_mean"] = assert_numpy(self.freeze_mean)
        return d
    

