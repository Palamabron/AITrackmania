"""Compact critic modules with batch-safe output shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class QuantileCriticConfig:
    feature_dim: int
    action_dim: int
    quantile_count: int


class ContinuousQCritic(nn.Module):
    """Scalar Q critic with a replaceable observation encoder."""

    def __init__(self, encoder: nn.Module, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.value = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim), nn.SiLU(), nn.Linear(feature_dim, 1)
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.value(torch.cat([self.encoder(observation), action], dim=-1)).squeeze(-1),
        )


class ContinuousValueCritic(nn.Module):
    """Scalar state-value critic with a replaceable observation encoder."""

    def __init__(self, encoder: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.value = nn.Linear(feature_dim, 1)

    def forward(self, observation: Any) -> torch.Tensor:
        features = self.encoder(observation)
        return cast(torch.Tensor, self.value(features).squeeze(-1))


class QuantileCritic(nn.Module):
    """Continuous-action critic producing fixed quantile locations."""

    def __init__(self, encoder: nn.Module, config: QuantileCriticConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantile_count = config.quantile_count
        self.value = nn.Sequential(
            nn.Linear(config.feature_dim + config.action_dim, config.feature_dim),
            nn.SiLU(),
            nn.Linear(config.feature_dim, config.quantile_count),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, self.value(torch.cat([self.encoder(observation), action], dim=-1))
        )
