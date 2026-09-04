"""Numerically stable squashed Gaussian policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.distributions import Normal

from trackmaniarl.core.contracts import PolicyMode


@dataclass(frozen=True, slots=True)
class GaussianActorConfig:
    feature_dim: int
    action_dim: int
    action_low: Sequence[float] | None = None
    action_high: Sequence[float] | None = None


class GaussianActor(nn.Module):
    """Continuous actor whose deterministic path is the distribution mean."""

    def __init__(self, encoder: nn.Module, config: GaussianActorConfig) -> None:
        super().__init__()
        low, high = _action_bounds(config.action_dim, config.action_low, config.action_high)
        self.encoder = encoder
        self.mean = nn.Linear(config.feature_dim, config.action_dim)
        self.log_std: nn.Linear | nn.Parameter = nn.Linear(config.feature_dim, config.action_dim)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, log_probability, _ = self.sample_with_latent(observation, mode)
        return action, log_probability

    def sample_with_latent(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)
        mean = distribution.mean
        raw = mean if mode is PolicyMode.EVALUATION else distribution.rsample()
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
        if not isinstance(self.log_std, nn.Linear):
            raise TypeError("GaussianActor requires an observation-dependent standard deviation")
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


class PpoGaussianActor(GaussianActor):
    """Squashed Gaussian PPO policy with state-independent exploration."""

    def __init__(self, encoder: nn.Module, config: GaussianActorConfig) -> None:
        super().__init__(encoder, config)
        del self.log_std
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        self._initialize_weights()

    def _distribution(self, observation: Any) -> Any:
        features = _encode(self.encoder, observation)
        mean = self.mean(features)
        if not isinstance(self.log_std, nn.Parameter):
            raise TypeError("PpoGaussianActor requires a state-independent standard deviation")
        log_std = self.log_std.expand_as(mean).clamp(-5, 2)
        return Normal(mean, log_std.exp())

    def _initialize_weights(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.zeros_(self.mean.bias)


def _encode(encoder: nn.Module, observation: Any) -> torch.Tensor:
    """Call encoders with tensor, tuple, or mapping observations."""

    if isinstance(observation, tuple):
        return cast(torch.Tensor, encoder(*observation))
    if isinstance(observation, dict):
        return cast(torch.Tensor, encoder(**observation))
    return cast(torch.Tensor, encoder(observation))


def _action_bounds(
    action_dim: int,
    action_low: Sequence[float] | None,
    action_high: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.as_tensor(
        [-1.0] * action_dim if action_low is None else action_low, dtype=torch.float32
    )
    high = torch.as_tensor(
        [1.0] * action_dim if action_high is None else action_high, dtype=torch.float32
    )
    if low.shape != (action_dim,) or high.shape != (action_dim,) or torch.any(high <= low):
        raise ValueError("action bounds must match action_dim and satisfy high > low")
    return low, high
