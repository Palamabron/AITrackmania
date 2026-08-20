"""Fixed-location quantile head used by QR-DQN."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from trackmaniarl.models.contracts import ValueRepresentation, ValueSupport


class FixedQuantileHead(nn.Module):
    representation = ValueRepresentation.FIXED_QUANTILE

    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        quantile_count: int = 32,
        *,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        if min(feature_dim, action_count, quantile_count) < 1:
            raise ValueError("head dimensions must be positive")
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.quantile_count = quantile_count
        self.advantage = nn.Linear(feature_dim, action_count * quantile_count)
        self.value = nn.Linear(feature_dim, quantile_count) if dueling else None

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        self._validate_support(support)
        leading = features.shape[:-1]
        advantages = self.advantage(features).reshape(
            *leading, self.quantile_count, self.action_count
        )
        if self.value is None:
            return cast(torch.Tensor, advantages)
        value = self.value(features).unsqueeze(-1)
        return cast(torch.Tensor, value + advantages - advantages.mean(dim=-1, keepdim=True))

    def evaluate_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_support(support)
        quantiles = torch.arange(self.quantile_count, device=actions.device)
        rows = actions.unsqueeze(-1) + self.action_count * quantiles
        selected_weight = self.advantage.weight[rows]
        selected_bias = self.advantage.bias[rows]
        selected = (features.unsqueeze(-2) * selected_weight).sum(dim=-1) + selected_bias
        if self.value is None:
            return selected
        weights = self.advantage.weight.reshape(
            self.quantile_count, self.action_count, self.feature_dim
        )
        biases = self.advantage.bias.reshape(self.quantile_count, self.action_count)
        mean_advantage = torch.einsum("...d,nd->...n", features, weights.mean(dim=1))
        mean_advantage = mean_advantage + biases.mean(dim=1)
        return cast(torch.Tensor, self.value(features) + selected - mean_advantage)

    def _validate_support(self, support: ValueSupport) -> None:
        if support.points.shape[-1] != self.quantile_count:
            raise ValueError("fixed head and strategy quantile counts must match")
