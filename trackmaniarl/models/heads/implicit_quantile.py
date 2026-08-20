"""Cosine-embedded quantile head shared by IQN and FQF."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from trackmaniarl.models.contracts import ValueRepresentation, ValueSupport


class ImplicitQuantileHead(nn.Module):
    representation = ValueRepresentation.IMPLICIT_QUANTILE

    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        cosine_count: int = 64,
        *,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        if min(feature_dim, action_count, cosine_count) < 1:
            raise ValueError("head dimensions must be positive")
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.register_buffer("frequencies", torch.arange(1, cosine_count + 1).float())
        self.quantile_embedding = nn.Sequential(nn.Linear(cosine_count, feature_dim), nn.SiLU())
        self.advantage = nn.Linear(feature_dim, action_count)
        self.value = nn.Linear(feature_dim, 1) if dueling else None

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        combined = self._combined(features, support.points)
        advantages = self.advantage(combined)
        if self.value is None:
            return cast(torch.Tensor, advantages)
        return cast(
            torch.Tensor,
            self.value(combined) + advantages - advantages.mean(dim=-1, keepdim=True),
        )

    def evaluate_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        combined = self._combined(features, support.points)
        selected_weight = self.advantage.weight[actions].unsqueeze(-2)
        selected_bias = self.advantage.bias[actions].unsqueeze(-1)
        selected = (combined * selected_weight).sum(dim=-1) + selected_bias
        if self.value is None:
            return selected
        mean_weight = self.advantage.weight.mean(dim=0)
        mean_bias = self.advantage.bias.mean()
        mean_advantage = F.linear(combined, mean_weight.unsqueeze(0), mean_bias.reshape(1))
        return cast(
            torch.Tensor,
            self.value(combined).squeeze(-1) + selected - mean_advantage.squeeze(-1),
        )

    def _combined(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.shape[:-1] != features.shape[:-1]:
            raise ValueError("quantile support must share feature leading dimensions")
        frequencies = cast(torch.Tensor, self.frequencies)
        cosine = torch.cos(torch.pi * points.unsqueeze(-1) * frequencies)
        embedding = cast(torch.Tensor, self.quantile_embedding(cosine))
        return features.unsqueeze(-2) * embedding
