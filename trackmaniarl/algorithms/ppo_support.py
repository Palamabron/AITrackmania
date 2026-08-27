"""Normalization and tensor helpers shared by PPO updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from trackmaniarl.core.pytree import tree_map


@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    enabled: bool
    clip: float


@dataclass(frozen=True, slots=True)
class _RewardStep:
    rewards: torch.Tensor
    ends: torch.Tensor
    normalized: torch.Tensor
    index: int
    gamma: float


@dataclass(frozen=True, slots=True)
class AdvantageInputs:
    rewards: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    bootstrap_discounts: torch.Tensor
    episode_ends: torch.Tensor
    gae_lambda: float


@dataclass(frozen=True, slots=True)
class _AdvantageRecurrence:
    discounts: torch.Tensor
    episode_ends: torch.Tensor
    gae_lambda: float


class _ObservationNormalizer:
    def __init__(self, config: NormalizerConfig) -> None:
        self.enabled = config.enabled
        self.clip = config.clip
        self._moments: dict[str, _Moments] = {}

    def normalize(self, value: Any, sample_dimensions: int) -> Any:
        if not self.enabled:
            return value
        return _map_tensor_tree(
            value,
            lambda path, leaf: self._normalize_leaf(path, leaf),
        )

    def update(self, value: Any, sample_dimensions: int) -> None:
        if not self.enabled:
            return
        _map_tensor_tree(
            value,
            lambda path, leaf: self._update_leaf(path, leaf, sample_dimensions),
        )

    def _normalize_leaf(self, path: str, value: torch.Tensor) -> torch.Tensor:
        moments = self._moments.get(path)
        if moments is None:
            return value
        mean = moments.mean.to(value.device)
        variance = moments.variance.to(value.device)
        return ((value - mean) / (variance + 1e-8).sqrt()).clamp(-self.clip, self.clip)

    def _update_leaf(self, path: str, value: torch.Tensor, sample_dimensions: int) -> torch.Tensor:
        dimensions = tuple(range(sample_dimensions))
        mean = value.detach().mean(dim=dimensions).cpu()
        variance = value.detach().var(dim=dimensions, unbiased=False).cpu()
        count = int(np.prod(value.shape[:sample_dimensions]))
        moments = self._moments.setdefault(path, _Moments.zeros_like(mean))
        moments.update(mean, variance, count)
        return value

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "clip": self.clip,
            "moments": {key: value.state_dict() for key, value in self._moments.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state["enabled"]) != self.enabled or float(state["clip"]) != self.clip:
            raise ValueError("PPO observation normalizer configuration does not match")
        self._moments = {
            str(key): _Moments.from_state(cast(Mapping[str, Any], value))
            for key, value in cast(Mapping[str, Any], state["moments"]).items()
        }


class _RewardNormalizer:
    def __init__(self, config: NormalizerConfig) -> None:
        self.enabled = config.enabled
        self.clip = config.clip
        self.moments = _Moments.zeros_like(torch.zeros(()))
        self.discounted_returns: torch.Tensor | None = None

    def normalize(
        self,
        rewards: torch.Tensor,
        episode_ends: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return rewards
        sequence = rewards if rewards.ndim > 1 else rewards.unsqueeze(0)
        ends = episode_ends if rewards.ndim > 1 else episode_ends.unsqueeze(0)
        gamma = float(discounts.max().item()) if discounts.numel() else 0.0
        self._ensure_returns(sequence.shape[0], rewards.device, rewards.dtype)
        normalized = torch.empty_like(sequence)
        for step in range(sequence.shape[1]):
            self._normalize_step(_RewardStep(sequence, ends, normalized, step, gamma))
        return normalized if rewards.ndim > 1 else normalized[0]

    def _normalize_step(self, step: _RewardStep) -> None:
        assert self.discounted_returns is not None
        self.discounted_returns.mul_(step.gamma).add_(step.rewards[:, step.index])
        values = self.discounted_returns.detach().cpu()
        self.moments.update(values.mean(), values.var(unbiased=False), values.numel())
        scale = float((self.moments.variance + 1e-8).sqrt().item())
        step.normalized[:, step.index] = (step.rewards[:, step.index] / scale).clamp(
            -self.clip, self.clip
        )
        self.discounted_returns.masked_fill_(step.ends[:, step.index], 0.0)

    def _ensure_returns(self, count: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.discounted_returns is None or self.discounted_returns.shape != (count,):
            self.discounted_returns = torch.zeros(count, device=device, dtype=dtype)
        else:
            self.discounted_returns = self.discounted_returns.to(device=device, dtype=dtype)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "clip": self.clip,
            "moments": self.moments.state_dict(),
            "discounted_returns": self.discounted_returns,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state["enabled"]) != self.enabled or float(state["clip"]) != self.clip:
            raise ValueError("PPO reward normalizer configuration does not match")
        self.moments = _Moments.from_state(cast(Mapping[str, Any], state["moments"]))
        value = state["discounted_returns"]
        self.discounted_returns = cast(torch.Tensor | None, value)


class _Moments:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor, count: float) -> None:
        self.mean = mean
        self.variance = variance
        self.count = count

    @classmethod
    def zeros_like(cls, value: torch.Tensor) -> _Moments:
        return cls(torch.zeros_like(value), torch.ones_like(value), 1e-4)

    def update(self, mean: torch.Tensor, variance: torch.Tensor, count: int) -> None:
        if count < 1:
            return
        delta = mean - self.mean
        total = self.count + count
        combined = self.variance * self.count + variance * count
        combined += delta.square() * self.count * count / total
        self.mean = self.mean + delta * count / total
        self.variance = combined / total
        self.count = total

    def state_dict(self) -> Mapping[str, Any]:
        return {"mean": self.mean, "variance": self.variance, "count": self.count}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> _Moments:
        return cls(
            cast(torch.Tensor, state["mean"]),
            cast(torch.Tensor, state["variance"]),
            float(state["count"]),
        )


def _map_tensor_tree(value: Any, function: Any, path: str = "root") -> Any:
    if isinstance(value, torch.Tensor):
        return function(path, value)
    if isinstance(value, Mapping):
        return {
            key: _map_tensor_tree(item, function, f"{path}.{key}") for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _map_tensor_tree(item, function, f"{path}.{index}") for index, item in enumerate(value)
        )
    raise TypeError("PPO observation PyTrees must contain tensors, mappings, or tuples")


def generalized_advantage_estimate(inputs: AdvantageInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute time-limit-aware GAE over flat transitions or ordered sequences."""

    return _generalized_advantage(inputs)


def _generalized_advantage(inputs: AdvantageInputs) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = inputs.rewards.ndim > 1
    rewards_2d = _as_sequence(inputs.rewards)
    values_2d = _as_sequence(inputs.values)
    next_values_2d = _as_sequence(inputs.next_values)
    discounts_2d = _as_sequence(inputs.bootstrap_discounts)
    ends_2d = _as_sequence(inputs.episode_ends)
    deltas = rewards_2d + discounts_2d * next_values_2d - values_2d
    recurrence = _AdvantageRecurrence(discounts_2d, ends_2d, inputs.gae_lambda)
    advantages = _backward_advantages(deltas, recurrence)
    returns = advantages + values_2d
    return (advantages, returns) if sequence else (advantages[:, 0], returns[:, 0])


def _backward_advantages(deltas: torch.Tensor, recurrence: _AdvantageRecurrence) -> torch.Tensor:
    advantages = torch.zeros_like(deltas)
    running = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    for step in range(deltas.shape[1] - 1, -1, -1):
        continuation = (~recurrence.episode_ends[:, step]).to(deltas.dtype)
        running = deltas[:, step] + (
            recurrence.discounts[:, step] * recurrence.gae_lambda * continuation * running
        )
        advantages[:, step] = running
    return advantages


def _as_sequence(value: torch.Tensor) -> torch.Tensor:
    return value if value.ndim > 1 else value[:, None]


def _flatten_samples(value: Any, reference: torch.Tensor) -> Any:
    sample_dimensions = reference.ndim
    if sample_dimensions == 1:
        return value

    def flatten(leaf: Any) -> torch.Tensor:
        if not isinstance(leaf, torch.Tensor):
            raise TypeError("PPO batch PyTrees must contain only tensor leaves")
        return leaf.reshape(-1, *leaf.shape[sample_dimensions:])

    return tree_map(flatten, value)


def _index_samples(value: Any, indices: torch.Tensor) -> Any:
    return tree_map(lambda leaf: leaf[indices], value)


def _float_tensor_tree(value: Any, name: str) -> Any:
    def convert(leaf: Any) -> torch.Tensor:
        if not isinstance(leaf, torch.Tensor):
            raise TypeError(f"{name} PyTrees must contain only tensor leaves")
        return leaf.float()

    return tree_map(convert, value)
