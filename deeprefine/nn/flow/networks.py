import torch
import numpy as np
import torch.nn as nn

from deeprefine.nn.flow.basics import Flow
from deeprefine.nn.flow.invertible_blocks import SplitChannels, MergeChannels, RealNVP
from deeprefine.protein.icconverter import ICConverter
from deeprefine.protein.whiten import Whitener

class SequentialFlow(Flow):
    def __init__(self, blocks):
        """
        Represents a diffeomorphism that can be computed
        as a discrete finite stack of layers.
        
        Returns the transformed variable and the log determinant
        of the Jacobian matrix.
            
        Parameters
        ----------
        blocks : Tuple / List of flow blocks
        """
        super().__init__()
        self._blocks = nn.ModuleList(blocks)

    def forward(self, xs, inverse=False, **kwargs):
        """
        Transforms the input along the diffeomorphism and returns
        the transformed variable together with the volume change.
            
        Parameters
        ----------
        x : PyTorch Floating Tensor.
            Input variable to be transformed. 
            Tensor of shape `[..., n_dimensions]`.
        inverse: boolean.
            Indicates whether forward or inverse transformation shall be performed.
            If `True` computes the inverse transformation.
        
        Returns
        -------
        z: PyTorch Floating Tensor.
            Transformed variable. 
            Tensor of shape `[..., n_dimensions]`.
        log_det_J : PyTorch Floating Tensor.
            Total volume change as a result of the transformation.
            Corresponds to the log determinant of the Jacobian matrix.
        """
        log_det_J = 0.0
        blocks = self._blocks
        if inverse:
            blocks = reversed(blocks)
        for i, block in enumerate(blocks):
            xs, ddlogp = block(xs, inverse=inverse, **kwargs)
            log_det_J += ddlogp
        return xs, log_det_J

    def _forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs, inverse=False)

    def _inverse(self, *args, **kwargs):
        return self.forward(*args, **kwargs, inverse=True)

    def __iter__(self):
        return iter(self._blocks)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._blocks[index]
        else:
            indices = np.arange(len(self))[index]
            return SequentialFlow([self._blocks[i] for i in indices])

    def __len__(self):
        return len(self._blocks)
    

