"""Scalar discrete Q head."""

from __future__ import annotations

from enum import StrEnum
from typing import cast

import torch
from torch import nn

from trackmaniarl.models.contracts import ValueRepresentation, ValueSupport


class ScalarQMode(StrEnum):
    STANDARD = "standard"
    DUELING = "dueling"


class ScalarQHead(nn.Module):
    representation = ValueRepresentation.SCALAR

    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        mode: ScalarQMode = ScalarQMode.STANDARD,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or action_count < 1:
            raise ValueError("head dimensions must be positive")
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.advantage = nn.Linear(feature_dim, action_count)
        self.value = nn.Linear(feature_dim, 1) if mode is ScalarQMode.DUELING else None

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        del support
        advantages = self.advantage(features)
        values = self._dueling(advantages, features)
        return values.unsqueeze(-2)

    def evaluate_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        del support
        values = self._selected(features, actions)
        return values.unsqueeze(-1)

    def _dueling(self, advantages: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.value is None:
            return advantages
        return cast(
            torch.Tensor,
            self.value(features) + advantages - advantages.mean(dim=-1, keepdim=True),
        )

    def _selected(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        selected_weight = self.advantage.weight[actions]
        selected_bias = self.advantage.bias[actions]
        selected = (features * selected_weight).sum(dim=-1) + selected_bias
        if self.value is None:
            return selected
        mean_weight = self.advantage.weight.mean(dim=0)
        mean_bias = self.advantage.bias.mean()
        mean_advantage = (features * mean_weight).sum(dim=-1) + mean_bias
        return cast(torch.Tensor, self.value(features).squeeze(-1) + selected - mean_advantage)
