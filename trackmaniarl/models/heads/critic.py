"""Encoder-independent continuous critic heads."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ContinuousCriticHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, output_count: int = 1) -> None:
        super().__init__()
        if min(feature_dim, action_dim, output_count) < 1:
            raise ValueError("critic dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, output_count),
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(torch.cat([features, actions], dim=-1)))
