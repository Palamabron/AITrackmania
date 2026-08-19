"""Compact critic modules with batch-safe output shapes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn


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

    def __init__(
        self, encoder: nn.Module, feature_dim: int, action_dim: int, quantile_count: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantile_count = quantile_count
        self.value = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, quantile_count),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, self.value(torch.cat([self.encoder(observation), action], dim=-1))
        )


class DiscreteQuantileNetwork(nn.Module):
    """IQN-compatible quantile network for a discrete action space."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        action_count: int,
        cosine_count: int = 64,
        *,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.register_buffer("frequencies", torch.arange(1, cosine_count + 1).float())
        self.quantile_embedding = nn.Sequential(nn.Linear(cosine_count, feature_dim), nn.SiLU())
        self.dueling = dueling
        self.head = nn.Linear(feature_dim, action_count)
        self.value = nn.Linear(feature_dim, 1) if dueling else None

    def forward(self, observation: Any, quantiles: torch.Tensor) -> torch.Tensor:
        features = cast(torch.Tensor, self.encoder(observation))
        return self.quantiles_from_features(features, quantiles)

    def quantiles_from_features(
        self, features: torch.Tensor, quantiles: torch.Tensor
    ) -> torch.Tensor:
        """Apply the IQN head to ``(batch, dim)`` or per-step ``(batch, steps, dim)`` features."""

        if quantiles.ndim != 2:
            raise ValueError("quantiles must have shape (batch, quantile_count)")
        frequencies = cast(torch.Tensor, self.frequencies)
        cosine = torch.cos(torch.pi * quantiles.unsqueeze(-1) * frequencies)
        embedding = cast(torch.Tensor, self.quantile_embedding(cosine))
        if features.ndim == 2:
            combined = features.unsqueeze(1) * embedding
        elif features.ndim == 3:
            combined = features.unsqueeze(2) * embedding.unsqueeze(1)
        else:
            raise ValueError("features must have shape (batch, dim) or (batch, steps, dim)")
        advantages = self.head(combined)
        if self.value is None:
            return cast(torch.Tensor, advantages)
        return cast(
            torch.Tensor,
            self.value(combined) + advantages - advantages.mean(dim=-1, keepdim=True),
        )

    def supports_sequence_training(self) -> bool:
        return callable(getattr(self.encoder, "encode_steps", None))

    def encode_sequence(self, observation: Any) -> torch.Tensor:
        """Return per-step features when the encoder exposes ``encode_steps``."""

        encode_steps = getattr(self.encoder, "encode_steps", None)
        if not callable(encode_steps):
            raise TypeError("encoder does not expose per-step sequence features")
        return cast(torch.Tensor, encode_steps(observation))

    def evaluation_quantiles(self, quantile_count: int, batch_size: int) -> torch.Tensor:
        device = cast(torch.Tensor, self.frequencies).device
        quantiles = torch.linspace(
            0.5 / quantile_count,
            1 - 0.5 / quantile_count,
            quantile_count,
            device=device,
        )
        return quantiles.expand(batch_size, -1)

    def q_values(self, observation: Any, quantile_count: int = 32) -> torch.Tensor:
        _, batch_size = _observation_device_and_batch(observation)
        quantiles = self.evaluation_quantiles(quantile_count, batch_size)
        return cast(torch.Tensor, self(observation, quantiles).mean(dim=1))


def _observation_device_and_batch(observation: Any) -> tuple[torch.device, int]:
    if isinstance(observation, torch.Tensor):
        if observation.ndim < 1:
            raise ValueError("observation tensor requires a batch axis")
        return observation.device, int(observation.shape[0])
    if isinstance(observation, Mapping):
        for value in observation.values():
            if isinstance(value, torch.Tensor):
                if value.ndim < 1:
                    raise ValueError("observation tensor requires a batch axis")
                return value.device, int(value.shape[0])
    raise TypeError("IQN observation must contain at least one batched tensor")
