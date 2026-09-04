"""Stateless temporal identity core."""

from __future__ import annotations

import torch
from torch import nn

from trackmaniarl.core.pytree import PyTree


class IdentityTemporalCore(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = input_dim

    def unroll(self, features: torch.Tensor, burn_in: int) -> torch.Tensor:
        if burn_in != 0:
            raise ValueError("identity temporal core does not support burn-in")
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("temporal features must have shape (batch, time, input_dim)")
        return features

    def initial_state(self, batch_size: int, device: torch.device) -> PyTree:
        del batch_size, device
        return ()

    def step(self, feature: torch.Tensor, state: PyTree) -> tuple[torch.Tensor, PyTree]:
        if state != ():
            raise ValueError("identity temporal state must be empty")
        return feature, state
