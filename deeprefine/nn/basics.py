import numbers
import numpy as np

import torch
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
    activation : None or torch function
        Nonlinear activation type
    dropout: float, 0 - 1, default None
        Probability of an element to be zeroed
    """

    def __init__(self, in_features, out_features, activation=None, dropout=None):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation
        self.dropout = self._get_dropout(dropout)

    # def _get_activation(self, activation):
    #     if activation is None:
    #         return nn.Identity()
    #     elif isinstance(activation, str):
    #         try:
    #             return getattr(F, activation)
    #         except:
    #             raise ValueError(
    #                 "Unsupported activation function: {}".format(activation)
    #             )
    #     else:
    #         raise ValueError("Need a valid activation type in str")

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


class DenseNet(nn.Module):
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
    activation : torch element-wise function or None
        nonlinear activation function in hidden layers
    init_outputs : None or float or array
        None means default initialization for the output layer, otherwise it is currently initialized with 0
    **args : kwargs
        Additional keyword arguments passed to the layer
    """

    def __init__(
        self,
        input_size,
        output_size,
        nlayers=3,
        nhidden=100,
        activation=torch.relu,
        dropout=None,
        init_outputs=None,
        **args
    ):
        super().__init__()
        if isinstance(nhidden, numbers.Integral):
            nhidden = nhidden * np.ones(nlayers - 1, dtype=int)
        else:
            nhidden = np.array(nhidden)
            if nhidden.size != nlayers - 1:
                raise ValueError(
                    "Illegal size of nhidden. Expecting 1d array with nlayers-1 elements"
                )
        assert nlayers > 1, "nlayers should at least be 2!"

        layers = []
        layers.append(
            DenseLayer(
                input_size, nhidden[0], activation=activation, dropout=dropout, **args
            )
        )
        for i in range(nlayers - 2):
            layers.append(
                DenseLayer(
                    nhidden[i], nhidden[i+1], activation=activation, dropout=dropout, **args
                )
            )
        if init_outputs is None:
            final_layer = nn.Linear(nhidden[-1], output_size, **args)
        else:
            final_layer = nn.Linear(nhidden[-1], output_size, **args)
            nn.init.zeros_(final_layer.weight.data)
            nn.init.constant_(final_layer.bias.data, init_outputs)
        layers.append(final_layer)
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)
