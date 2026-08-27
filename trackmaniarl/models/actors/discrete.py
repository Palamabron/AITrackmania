"""Categorical policy heads for discrete TrackMania actions."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from trackmaniarl.core.contracts import PolicyMode


class CategoricalActor(nn.Module):
    """Categorical policy exposing logits, sampled actions and deterministic argmax actions."""

    def __init__(self, encoder: nn.Module, feature_dim: int, action_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.logits = nn.Linear(feature_dim, action_count)

    def forward(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(_encode(self.encoder, observation))
        distribution: Any = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if mode is PolicyMode.EVALUATION else distribution.sample()
        return actions, distribution.log_prob(actions)

    def probabilities(self, observation: Any) -> torch.Tensor:
        return cast(torch.Tensor, self.logits(_encode(self.encoder, observation)).softmax(dim=-1))


def _encode(encoder: nn.Module, observation: Any) -> torch.Tensor:
    """Call encoders with tensor, tuple, or mapping observations."""

    if isinstance(observation, tuple):
        return cast(torch.Tensor, encoder(*observation))
    if isinstance(observation, dict):
        return cast(torch.Tensor, encoder(**observation))
    return cast(torch.Tensor, encoder(observation))
