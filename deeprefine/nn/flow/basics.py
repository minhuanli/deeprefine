import numpy as np

import torch.nn as nn


class Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, *xs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *xs, **kwargs):
        raise NotImplementedError()

    def forward(self, *xs, inverse=False, **kwargs):
        """Forward method of the flow

        Parameters
        ----------
        inverse : bool, optional
            Compute in forward or inverse mode
        """
        if inverse:
            return self._inverse(*xs, **kwargs)
        else:
            return self._forward(*xs, **kwargs)


class InverseFlow(Flow):
    """The inverse of a given transform.

    Parameters
    ----------
    delegate : Flow
        The flow to invert.
    """

    def __init__(self, delegate):
        super().__init__()
        self._delegate = delegate

    def _forward(self, *xs, **kwargs):
        return self._delegate._inverse(*xs, **kwargs)

    def _inverse(self, *xs, **kwargs):
        return self._delegate._forward(*xs, **kwargs)
