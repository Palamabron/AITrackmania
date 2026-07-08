"""Neural network utility functions and PopArt normalization."""

from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any, cast

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter


def detach(x):
    """Detach tensor from computation graph, or recursively detach elements.

    Args:
        x: A torch.Tensor or iterable of tensors/objects.

    Returns:
        Detached tensor if input is a tensor, otherwise a list of detached elements.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    else:
        return [detach(elem) for elem in x]


def no_grad(model):
    """Set requires_grad=False for all parameters in the model.

    Args:
        model: A torch.nn.Module.

    Returns:
        The same model instance with all parameters frozen (requires_grad=False).
    """
    for p in model.parameters():
        p.requires_grad = False
    return model


def exponential_moving_average(averages, values, factor):
    """Update averages in-place using exponential moving average.

    Args:
        averages: Iterable of tensors to update (modified in-place).
        values: Iterable of current values.
        factor: EMA blending factor (0-1). Higher values weight new values more.
    """
    with torch.no_grad():
        for a, v in zip(averages, values, strict=True):
            a += factor * (v - a)


def copy_shared(model_a):
    """Create a deepcopy of a model with shared parameter storage.

    The copied model shares the underlying parameter tensors with the original,
    so updates to one affect the other. Useful for creating no-grad evaluation
    copies that stay in sync with a training model.

    Args:
        model_a: Model to copy.

    Returns:
        A deepcopy of the model with state_dict storage shared with the original.

    Note:
        torch.cuda.Stream cannot be pickled/deepcopied, so streams are replaced
        with None during copy and will be recreated on first use if needed.
    """
    import copy as copy_module

    stream_type = getattr(getattr(torch, "cuda", None), "Stream", None)
    dispatch = cast(
        MutableMapping[type, Any] | None, getattr(copy_module, "_deepcopy_dispatch", None)
    )
    old_dispatch = None
    if stream_type is not None and dispatch is not None:
        old_dispatch = dispatch.get(stream_type)
        dispatch[stream_type] = lambda obj, memo: None
    try:
        model_b = deepcopy(model_a)
    finally:
        if stream_type is not None and dispatch is not None:
            if old_dispatch is not None:
                dispatch[stream_type] = old_dispatch
            else:
                dispatch.pop(stream_type, None)
    sda = model_a.state_dict(keep_vars=True)
    sdb = model_b.state_dict(keep_vars=True)
    for key in sda:
        a, b = sda[key], sdb[key]
        b.data = a.data  # a.data and b.data differ but underlying data_ptr is shared
        assert b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
    return model_b


class PopArt(Module):
    """PopArt normalization for value functions.

    Adaptively normalizes targets and rescales output layer weights to handle
    rewards spanning many orders of magnitude.

    Reference:
        http://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude

    Args:
        output_layer: Linear layer(s) to adapt. Can be a single layer or tuple/list.
        beta: EMA update rate for statistics.
        zero_debias: If True, uses bias-corrected EMA (recommended).
        start_pop: Delay weight rescaling for this many updates to accumulate stable stats.
    """

    def __init__(
        self, output_layer, beta: float = 0.0003, zero_debias: bool = True, start_pop: int = 8
    ):
        super().__init__()
        self.start_pop = start_pop
        self.beta = beta
        self.zero_debias = zero_debias
        self.output_layers = (
            output_layer
            if isinstance(output_layer, (tuple, list, torch.nn.ModuleList))
            else (output_layer,)
        )
        layer0 = self.output_layers[0]
        shape = tuple(layer0.bias.shape)  # type: ignore[arg-type,union-attr]
        device = layer0.bias.device  # type: ignore[union-attr]
        assert all(shape == tuple(x.bias.shape) for x in self.output_layers)  # type: ignore[arg-type]
        self.mean = Parameter(torch.zeros(shape, device=device), requires_grad=False)  # type: ignore[arg-type]
        self.mean_square = Parameter(torch.ones(shape, device=device), requires_grad=False)  # type: ignore[arg-type]
        self.std = Parameter(torch.ones(shape, device=device), requires_grad=False)  # type: ignore[arg-type]
        self.updates = 0

    @torch.no_grad()
    def update(self, targets):
        """Update statistics and rescale output layer, then return normalized targets.

        Args:
            targets: Target values to incorporate into statistics.

        Returns:
            Normalized targets.
        """
        beta = max(1 / (self.updates + 1), self.beta) if self.zero_debias else self.beta

        new_mean = (1 - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1 - beta) * self.mean_square + beta * (targets * targets).mean(0)
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)

        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= (self.std / new_std)[:, None]  # type: ignore[operator]
                layer.bias *= self.std  # type: ignore[operator]
                layer.bias += self.mean - new_mean
                layer.bias /= new_std

        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.normalize(targets)

    def normalize(self, x):
        """Normalize input using current statistics.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        """Inverse of normalize - convert from normalized to original scale.

        Args:
            x: Normalized tensor.

        Returns:
            Unnormalized tensor.
        """
        return x * self.std + self.mean

    def normalize_sum(self, s):
        """Normalize sum while preserving relative weightings.

        Args:
            s: Sum of values.

        Returns:
            Normalized sum.
        """
        return (s - self.mean.sum()) / self.std.norm()
