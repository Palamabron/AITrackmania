"""Numerically stable squashed Gaussian policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """Continuous actor whose deterministic path is the distribution mean."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        action_dim: int,
        *,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(
            [-1.0] * action_dim if action_low is None else action_low, dtype=torch.float32
        )
        high = torch.as_tensor(
            [1.0] * action_dim if action_high is None else action_high, dtype=torch.float32
        )
        if low.shape != (action_dim,) or high.shape != (action_dim,) or torch.any(high <= low):
            raise ValueError("action bounds must match action_dim and satisfy high > low")
        self.encoder = encoder
        self.mean = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(
        self, observation: Any, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, log_probability, _ = self.sample_with_latent(
            observation, deterministic=deterministic
        )
        return action, log_probability

    def sample_with_latent(
        self, observation: Any, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)
        mean = distribution.mean
        raw = mean if deterministic else distribution.rsample()
        scale = cast(torch.Tensor, self.action_scale)
        bias = cast(torch.Tensor, self.action_bias)
        action = raw.tanh() * scale + bias
        log_probability = self._log_probability(distribution, raw)
        return action, log_probability, raw

    def evaluate_latent_actions(
        self, observation: Any, latent_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score exact pre-squash actions and estimate bounded-policy entropy."""

        distribution = self._distribution(observation)
        log_probability = self._log_probability(distribution, latent_actions)
        entropy_sample = -self._log_probability(distribution, distribution.rsample())
        return log_probability, entropy_sample

    def _distribution(self, observation: Any) -> Any:
        features = _encode(self.encoder, observation)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-5, 2)
        return Normal(mean, log_std.exp())

    def _log_probability(self, distribution: Any, raw: torch.Tensor) -> torch.Tensor:
        correction = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
        return cast(
            torch.Tensor,
            (
                distribution.log_prob(raw)
                - correction
                - cast(torch.Tensor, self.action_scale).log()
            ).sum(dim=-1),
        )


def _encode(encoder: nn.Module, observation: Any) -> torch.Tensor:
    """Call encoders with tensor, tuple, or mapping observations."""

    if isinstance(observation, tuple):
        return cast(torch.Tensor, encoder(*observation))
    if isinstance(observation, dict):
        return cast(torch.Tensor, encoder(**observation))
    return cast(torch.Tensor, encoder(observation))
