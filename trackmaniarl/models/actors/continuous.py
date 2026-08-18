"""Numerically stable squashed Gaussian policies."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """Continuous actor whose deterministic path is the distribution mean."""

    def __init__(self, encoder: nn.Module, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.mean = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)

    def forward(
        self, observation: Any, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = _encode(self.encoder, observation)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        distribution: Any = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        correction = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
        log_probability = (distribution.log_prob(raw) - correction).sum(dim=-1)
        return action, log_probability


def _encode(encoder: nn.Module, observation: Any) -> torch.Tensor:
    """Call encoders with tensor, tuple, or mapping observations."""

    if isinstance(observation, tuple):
        return cast(torch.Tensor, encoder(*observation))
    if isinstance(observation, dict):
        return cast(torch.Tensor, encoder(**observation))
    return cast(torch.Tensor, encoder(observation))
