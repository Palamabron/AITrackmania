"""Primitive building blocks shared by IMPALA-style CNN models.

Contains factory functions for GRU/LSTM recurrent layers, a small MLP builder,
and Kaiming weight initialisation applied to individual layers.
"""

import torch
from torch import nn


def gru(input_size: int, rnn_size: int, rnn_len: int, dropout: float = 0.1) -> nn.GRU:
    """Build a multi-layer batch-first GRU.

    Args:
        input_size: Number of features in each input step.
        rnn_size: Number of features in the hidden state.
        rnn_len: Number of stacked GRU layers. Must be >= 1.
        dropout: Dropout probability applied between layers (ignored if rnn_len == 1).

    Returns:
        Configured nn.GRU module with batch_first=True.
    """
    assert rnn_len >= 1
    return nn.GRU(
        input_size=input_size,
        hidden_size=rnn_size,
        num_layers=rnn_len,
        bias=True,
        batch_first=True,
        dropout=dropout,
        bidirectional=False,
    )


def lstm(input_size: int, rnn_size: int, rnn_len: int, dropout: float = 0.0) -> nn.LSTM:
    """Build a multi-layer batch-first LSTM.

    Args:
        input_size: Number of features in each input step.
        rnn_size: Number of features in the hidden state.
        rnn_len: Number of stacked LSTM layers. Must be >= 1.
        dropout: Dropout probability applied between layers (ignored if rnn_len == 1).

    Returns:
        Configured nn.LSTM module with batch_first=True.
    """
    assert rnn_len >= 1
    return nn.LSTM(
        input_size=input_size,
        hidden_size=rnn_size,
        num_layers=rnn_len,
        bias=True,
        batch_first=True,
        dropout=dropout,
        bidirectional=False,
    )


def mlp(sizes: list[int], dim_obs: int, activation=nn.ReLU) -> nn.Sequential:
    """Build a fully-connected MLP with the given layer widths.

    Args:
        sizes: Output width of each linear layer (e.g. [256, 256]).
        dim_obs: Input dimension fed into the first linear layer.
        activation: Activation class instantiated between every pair of layers.

    Returns:
        Sequential model mapping (batch, dim_obs) -> (batch, sizes[-1]).
    """
    layers = [nn.Linear(dim_obs, sizes[0]), activation()]
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        layers.append(activation())
    return nn.Sequential(*layers)


def init_kaiming(layer) -> None:
    """Initialise a Conv2d or Linear layer with Kaiming-normal weights and zero biases.

    Uses fan_in mode so the variance is preserved through the forward pass.

    Args:
        layer: Module with ``.weight`` and ``.bias`` attributes (nn.Linear or nn.Conv2d).
    """
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in")
    torch.nn.init.zeros_(layer.bias)
