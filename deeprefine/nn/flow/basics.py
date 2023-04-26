import numbers
import copy

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """Just your regular densely-connected NN layer

    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    activation : str, default None
        Nonlinear activation type
    dropout: float, 0 - 1, default None
        Probability of an element to be zeroed
    """

    def __init__(self, in_features, out_features, activation=None, dropout=None):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = self._get_activation(activation)
        self.dropout = self._get_dropout(dropout)

    def _get_activation(self, activation):
        if activation is None:
            return nn.Identity()
        elif isinstance(activation, str):
            try:
                return getattr(F, activation)
            except:
                raise ValueError(
                    "Unsupported activation function: {}".format(activation)
                )
        else:
            raise ValueError("Need a valid activation type in str")

    def _get_dropout(self, dropout):
        if dropout is None:
            return nn.Identity()
        elif isinstance(dropout, float):
            return nn.Dropout(dropout)
        else:
            raise ValueError("Need a valid dropout rate in float, between 0 and 1.")

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


def nonlinear_transform(
    input_size,
    output_size,
    nlayers=3,
    nhidden=100,
    activation="relu",
    dropout=None,
    init_outputs=None,
    **args
):
    """Generic dense trainable nonlinear transform

    Returns the layers of a dense feedforward network with nlayers-1 hidden layers with nhidden neurons
    and the specified activation functions. The last layer is linear in order to access the full real
    number range and has output_size output neurons.

    Parameters
    ----------
    input_size :  int
        number of input neurons
    output_size : int
        number of output neurons
    nlayers : int
        number of layers, including the linear output layer. nlayers=3 means two hidden layers with
        nonlinear activation and one linear output layer.
    nhidden : int
        number of neurons in each hidden layer, either a number or an array of length nlayers-1
        to specify the width of each hidden layer
    activation : str
        nonlinear activation function in hidden layers
    init_outputs : None or float or array
        None means default initialization for the output layer, otherwise it is currently initialized with 0
    **args : kwargs
        Additional keyword arguments passed to the layer

    Returns
    -------
    A list of nn.Module layers [layer1, layer2, ..., final_linear_layer]

    """
    if isinstance(nhidden, numbers.Integral):
        nhidden = nhidden * np.ones(nlayers - 1, dtype=int)
    else:
        nhidden = np.array(nhidden)
        if nhidden.size != nlayers - 1:
            raise ValueError(
                "Illegal size of nhidden. Expecting 1d array with nlayers-1 elements"
            )
    M = []
    M.append(
        DenseLayer(input_size, nhidden, activation=activation, dropout=dropout, **args)
    )
    for _ in range(nlayers - 2):
        M.append(
            DenseLayer(nhidden, nhidden, activation=activation, dropout=dropout, **args)
        )
    if init_outputs is None:
        final_layer = nn.Linear(nhidden, output_size, **args)
    else:
        final_layer = nn.Linear(nhidden, output_size, **args)
        nn.init.zeros_(final_layer.weight.data)
        nn.init.constant_(final_layer.bias.data, init_outputs)
    M.append(final_layer)

    return M


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
