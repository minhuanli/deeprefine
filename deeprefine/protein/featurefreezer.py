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
    2. Automatically freeze features following:
        a. never freeze the cartesian signals
        b. for distance signals, freeze those with std/mean < 0.05
        c. for angles with vi = (sin, cos) features, freeze those with c = |sum_vi|/sum|vi| > 0.996, equal to std of 5 degree

    Parameters:
    -----------
    X0 : Tensor, [n_batch, n_features]
        batch data to do statistics for mean

    frozen_indices : array, [n_frozen_features,] or None
        indices of features you want to freeze. If None, use automatic cutoff

    bond_idx: array
        indices according to bond length features

    cosangle_idx: array
        indices according to cos angle (including torsions) features

    singangle_idx: array
        indices according to sin angle (including torsions) features

    """
    def __init__(
        self,
        X0,
        frozen_indices=None,
        bond_idx=None,
        cosangle_idx=None,
        sinangle_idx=None,
        dist_cutoff=0.05,
        angle_cuoff=0.996,
        from_dict=False,
    ):
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
                self.cutoff = [dist_cutoff, angle_cuoff]
                # get bond frozen features
                bond_features = X0[:, bond_idx]
                bond_abs_mean = torch.abs(torch.mean(bond_features, axis=0))
                bond_std = torch.std(bond_features, axis=0)
                bond_frozen_idx = bond_idx[
                    assert_numpy(
                        torch.argwhere(bond_std / bond_abs_mean < dist_cutoff).reshape(
                            -1
                        )
                    )
                ]
                # get angle frozen features
                cosangle_features = X0[:, cosangle_idx]
                sinangle_features = X0[:, sinangle_idx]
                c_features = (
                    torch.sqrt(
                        torch.sum(cosangle_features, dim=0) ** 2
                        + torch.sum(sinangle_features, dim=0) ** 2
                    )
                    / X0.shape[0]
                )
                _angle_frozen_idx = assert_numpy(
                    torch.argwhere(c_features > angle_cuoff).reshape(-1)
                )
                cos_frozen_idx = cosangle_idx[_angle_frozen_idx]
                sin_frozen_idx = sinangle_idx[_angle_frozen_idx]
                self.frozen_idx = np.concatenate(
                    [bond_frozen_idx, cos_frozen_idx, sin_frozen_idx]
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
