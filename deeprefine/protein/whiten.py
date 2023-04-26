import numpy as np
import torch
from deeprefine.utils import assert_numpy, assert_tensor


def _pca(X0, keepdims=None, save_memory=True):
    """Implements PCA in Numpy"""
    if keepdims is None:
        keepdims = X0.shape[1]

    if save_memory:
        Nbatch = min(20000, X0.shape[0])
        batch_indices = np.random.choice(X0.shape[0], Nbatch)
        X0 = X0[batch_indices]

    # pca
    X0mean = X0.mean(axis=0)
    X0meanfree = X0 - X0mean
    C = np.matmul(X0meanfree.T, X0meanfree) / (X0meanfree.shape[0] - 1.0)
    eigval, eigvec = np.linalg.eigh(C)

    # sort in descending order and keep only the wanted eigenpairs
    I = np.argsort(eigval)[::-1]
    I = I[:keepdims]
    eigval = eigval[I]
    std = np.sqrt(eigval)
    eigvec = eigvec[:, I]

    # whiten and unwhiten matrices
    Twhiten = np.matmul(eigvec, np.diag(1.0 / std))
    Tblacken = np.matmul(np.diag(std), eigvec.T)
    return X0mean, Twhiten, Tblacken, std


def directsum_np(A, B):
    """Do direct sum of two matrices
    return a block matrix = [[A,0],[0,B]]
    """
    dsum = np.zeros(np.add(A.shape, B.shape))
    dsum[: A.shape[0], : A.shape[1]] = A
    dsum[A.shape[0] :, A.shape[1] :] = B
    return dsum


class Whitener(object):
    def __init__(self, X0, dim_cart_signal=None, keepdims=None):
        """Performs static whitening of the data given PCA of X0

        Parameters:
        -----------
        X0 : array, [n_batch, n_feature]
            Initial Data on which PCA will be computed.
        dim_cart_signal : int or None
            Number of cartesian indices in the features. By default treat all features in the same way
        keepdims : int or None
            Number of dimensions to keep. By default, all dimensions will be kept
        whiten_inverse : bool
            Whitens when calling inverse (default). Otherwise when calling forward

        """
        super().__init__()
        if keepdims is None:
            keepdims = X0.shape[1]
        self.dim = X0.shape[1]
        self.keepdims = keepdims
        self.dim_cart_signal = dim_cart_signal
        X0_np = assert_numpy(X0)
        if dim_cart_signal is not None:
            X0_cart = X0_np[:, :dim_cart_signal]
            X0_ic = X0_np[:, dim_cart_signal:]
            X0mean_cart, Twhiten_cart, Tblacken_cart, std_cart = _pca(
                X0_cart, keepdims=keepdims
            )
            X0mean_ic, Twhiten_ic, Tblacken_ic, std_ic = _pca(X0_ic, keepdims=None)
            # Do direct sum
            X0mean = np.concatenate([X0mean_cart, X0mean_ic])
            Twhiten = directsum_np(Twhiten_cart, Twhiten_ic)
            Tblacken = directsum_np(Tblacken_cart, Tblacken_ic)
            std = np.concatenate([std_cart, std_ic])
        else:
            X0mean, Twhiten, Tblacken, std = _pca(X0_np, keepdims=keepdims)
        self.X0mean = assert_tensor(X0mean, arr_type=torch.float32)
        self.Twhiten = assert_tensor(Twhiten, arr_type=torch.float32)
        self.Tblacken = assert_tensor(Tblacken, arr_type=torch.float32)
        self.std = assert_tensor(std, arr_type=torch.float32)
        if torch.any(self.std <= 0):
            raise ValueError(
                "Cannot construct whiten layer because trying to keep nonpositive eigenvalues."
            )

    def whiten(self, x):
        """
        x : Tensor, [n_batch, n_features]
        """
        # Whiten
        output_z = torch.matmul(x - self.X0mean, self.Twhiten)
        return output_z

    def blacken(self, z):
        """
        z : Tensor, [n_batch, n_features]
        """
        output_x = torch.matmul(z, self.Tblacken) + self.X0mean
        return output_x

    @classmethod
    def from_dict(cls, D):
        dim = D["dim"]
        keepdims = D["keepdims"]
        dim_cart_signal = D["dim_cart_signal"]
        X0mean = D["X0mean"]
        Twhiten = D["Twhiten"]
        Tblacken = D["Tblacken"]
        std = D["std"]
        c = cls(
            np.random.randn(2 * dim, dim),
            dim_cart_signal=dim_cart_signal,
            keepdims=keepdims,
        )

        c.X0mean = assert_tensor(X0mean, arr_type=torch.float32)
        c.Twhiten = assert_tensor(Twhiten, arr_type=torch.float32)
        c.Tblacken = assert_tensor(Tblacken, arr_type=torch.float32)
        c.std = assert_tensor(std, arr_type=torch.float32)
        return c

    def to_dict(self):
        D = {}
        D["dim"] = self.dim
        D["keepdims"] = self.keepdims
        D["dim_cart_signal"] = self.dim_cart_signal
        D["X0mean"] = assert_numpy(self.X0mean)
        D["Twhiten"] = assert_numpy(self.Twhiten)
        D["Tblacken"] = assert_numpy(self.Tblacken)
        D["std"] = assert_numpy(self.std)
        return D
