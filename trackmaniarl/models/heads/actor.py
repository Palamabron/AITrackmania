"""Encoder-independent actor heads."""

from __future__ import annotations

import torch
from torch import nn


class ContinuousActorHead(nn.Module):
    """Deterministic feature-to-action head for composed actor bundles."""

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        if feature_dim < 1 or action_dim < 1:
            raise ValueError("actor dimensions must be positive")
        self.action_dim = action_dim
        self.projection = nn.Linear(feature_dim, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(features))
