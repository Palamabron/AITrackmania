"""GRU temporal core with detached recurrent burn-in."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from trackmaniarl.core.pytree import PyTree


class GruTemporalCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("GRU dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.normalization = nn.LayerNorm(hidden_dim)

    def unroll(self, features: torch.Tensor, burn_in: int) -> torch.Tensor:
        self._validate(features, burn_in)
        hidden: torch.Tensor | None = None
        if burn_in:
            with torch.no_grad():
                _, hidden = self.recurrent(features[:, :burn_in])
            hidden = hidden.detach()
        values, _ = self.recurrent(features[:, burn_in:], hidden)
        return cast(torch.Tensor, self.normalization(values))

    def initial_state(self, batch_size: int, device: torch.device) -> PyTree:
        return torch.zeros(1, batch_size, self.output_dim, device=device)

    def step(self, feature: torch.Tensor, state: PyTree) -> tuple[torch.Tensor, PyTree]:
        if not isinstance(state, torch.Tensor):
            raise TypeError("GRU state must be a tensor")
        if feature.ndim != 2 or feature.shape[-1] != self.input_dim:
            raise ValueError("GRU step feature must have shape (batch, input_dim)")
        values, hidden = self.recurrent(feature.unsqueeze(1), state)
        return cast(torch.Tensor, self.normalization(values[:, 0])), hidden

    def _validate(self, features: torch.Tensor, burn_in: int) -> None:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("GRU features must have shape (batch, time, input_dim)")
        if not 0 <= burn_in < features.shape[1]:
            raise ValueError("burn_in must be in [0, time)")
