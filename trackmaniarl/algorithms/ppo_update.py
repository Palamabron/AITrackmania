from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from trackmaniarl.algorithms.ppo_support import (
    AdvantageInputs,
    _flatten_samples,
    _float_tensor_tree,
    _index_samples,
    _ObservationNormalizer,
    _RewardNormalizer,
    generalized_advantage_estimate,
)
from trackmaniarl.core.data import TrainingBatch


class PPOUpdateOwner(Protocol):
    model: Any
    optimizer: torch.optim.Optimizer
    scaler: Any
    device: torch.device
    observation_normalizer: _ObservationNormalizer
    reward_normalizer: _RewardNormalizer
    gae_lambda: float
    clip_epsilon: float
    value_clip_epsilon: float
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    update_epochs: int
    minibatch_size: int
    target_kl: float | None

    def autocast(self) -> AbstractContextManager[Any]: ...


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    raw_observations: Any
    observations: Any
    latent_actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    episode_ends: torch.Tensor
    next_observations: Any
    sample_dimensions: int


@dataclass(frozen=True, slots=True)
class _PreparedCore:
    observations: Any
    normalized: Any
    normalized_next: Any
    rewards: torch.Tensor
    discounts: torch.Tensor
    episode_ends: torch.Tensor


@dataclass(frozen=True, slots=True)
class _FlatBatch:
    observations: Any
    latent_actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Minibatch:
    observations: Any
    latent_actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Losses:
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    ratio: torch.Tensor
    log_ratio: torch.Tensor


@dataclass(frozen=True, slots=True)
class _EpochResult:
    totals: torch.Tensor
    updates: int
    mean_kl: float


@dataclass(frozen=True, slots=True)
class _TrainingOutcome:
    totals: torch.Tensor
    updates: int
    stopped_early: bool


class PPOUpdater:
    def __init__(self, owner: PPOUpdateOwner) -> None:
        self.owner = owner

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], int]:
        prepared = self._prepare_batch(batch)
        flat = self._flat_batch(prepared)
        outcome = self._optimize_epochs(flat)
        self.owner.observation_normalizer.update(
            prepared.raw_observations, prepared.sample_dimensions
        )
        return self._metrics(outcome), prepared.rewards.numel()

    def _prepare_batch(self, batch: TrainingBatch) -> _PreparedBatch:
        observations = _float_tensor_tree(batch.observations, "observations")
        next_observations = _float_tensor_tree(batch.next_observations, "next_observations")
        rewards = _tensor(batch.rewards, "rewards").float()
        discounts = _tensor(batch.bootstrap_discounts, "bootstrap_discounts").float()
        ends = _tensor(batch.terminated, "terminated").bool()
        ends |= _tensor(batch.truncated, "truncated").bool()
        sample_dimensions = rewards.ndim
        normalized = self.owner.observation_normalizer.normalize(observations, sample_dimensions)
        normalized_next = self.owner.observation_normalizer.normalize(
            next_observations, sample_dimensions
        )
        rewards = self.owner.reward_normalizer.normalize(rewards, ends, discounts)
        core = _PreparedCore(observations, normalized, normalized_next, rewards, discounts, ends)
        return self._prepared_values(batch, core)

    def _prepared_values(self, batch: TrainingBatch, core: _PreparedCore) -> _PreparedBatch:
        return _PreparedBatch(
            core.observations,
            core.normalized,
            _behavior_tensor(batch, "behavior_latent_actions", self.owner.device),
            _behavior_tensor(batch, "behavior_log_probabilities", self.owner.device),
            _behavior_tensor(batch, "behavior_values", self.owner.device),
            core.rewards,
            core.discounts,
            core.episode_ends,
            core.normalized_next,
            core.rewards.ndim,
        )

    def _flat_batch(self, prepared: _PreparedBatch) -> _FlatBatch:
        advantages, returns = self._advantages(prepared)
        return _FlatBatch(
            _flatten_samples(prepared.observations, returns),
            _flatten_samples(prepared.latent_actions, returns),
            prepared.old_log_probabilities.reshape(-1),
            prepared.old_values.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
        )

    def _advantages(self, prepared: _PreparedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_values = self.owner.model.value(prepared.next_observations)
            advantages, returns = generalized_advantage_estimate(
                AdvantageInputs(
                    prepared.rewards,
                    prepared.old_values,
                    next_values,
                    prepared.discounts,
                    prepared.episode_ends,
                    self.owner.gae_lambda,
                )
            )
            scale = advantages.std(unbiased=False).clamp_min(1e-8)
            return (advantages - advantages.mean()) / scale, returns

    def _optimize_epochs(self, batch: _FlatBatch) -> _TrainingOutcome:
        totals = torch.zeros(5, device=self.owner.device)
        updates = 0
        stopped_early = False
        for _ in range(self.owner.update_epochs):
            epoch = self._optimize_epoch(batch)
            totals += epoch.totals
            updates += epoch.updates
            if self.owner.target_kl is not None and epoch.mean_kl > self.owner.target_kl:
                stopped_early = True
                break
        return _TrainingOutcome(totals, updates, stopped_early)

    def _optimize_epoch(self, batch: _FlatBatch) -> _EpochResult:
        permutation = torch.randperm(batch.returns.numel(), device=self.owner.device)
        totals = torch.zeros(5, device=self.owner.device)
        updates = 0
        for start in range(0, batch.returns.numel(), self.owner.minibatch_size):
            indices = permutation[start : start + self.owner.minibatch_size]
            totals += self._minibatch_step(self._minibatch(batch, indices))
            updates += 1
        return _EpochResult(totals, updates, float(totals[3].item()) / updates)

    @staticmethod
    def _minibatch(batch: _FlatBatch, indices: torch.Tensor) -> _Minibatch:
        return _Minibatch(
            _index_samples(batch.observations, indices),
            batch.latent_actions[indices],
            batch.old_log_probabilities[indices],
            batch.old_values[indices],
            batch.advantages[indices],
            batch.returns[indices],
        )

    def _minibatch_step(self, batch: _Minibatch) -> torch.Tensor:
        losses = self._losses(batch)
        self._optimize(losses.total)
        return self._loss_metrics(losses)

    def _losses(self, batch: _Minibatch) -> _Losses:
        with self.owner.autocast():
            log_probabilities, entropy = self.owner.model.actor.evaluate_latent_actions(
                batch.observations, batch.latent_actions
            )
            values = self.owner.model.value(batch.observations)
            log_ratio = log_probabilities - batch.old_log_probabilities
            ratio = log_ratio.exp()
            policy = self._policy_loss(ratio, batch.advantages)
            value = self._value_loss(values, batch)
            total = policy + self.owner.value_coefficient * value
            total -= self.owner.entropy_coefficient * entropy.mean()
        return _Losses(policy, value, entropy, total, ratio, log_ratio)

    def _policy_loss(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        clipped = ratio.clamp(1.0 - self.owner.clip_epsilon, 1.0 + self.owner.clip_epsilon)
        return -torch.minimum(ratio * advantages, clipped * advantages).mean()

    def _value_loss(self, values: torch.Tensor, batch: _Minibatch) -> torch.Tensor:
        clipped = batch.old_values + (values - batch.old_values).clamp(
            -self.owner.value_clip_epsilon, self.owner.value_clip_epsilon
        )
        raw_error = (values - batch.returns).square()
        clipped_error = (clipped - batch.returns).square()
        return 0.5 * torch.maximum(raw_error, clipped_error).mean()

    def _optimize(self, loss: torch.Tensor) -> None:
        owner = self.owner
        owner.optimizer.zero_grad(set_to_none=True)
        owner.scaler.scale(loss).backward()
        owner.scaler.unscale_(owner.optimizer)
        torch.nn.utils.clip_grad_norm_(owner.model.parameters(), owner.max_gradient_norm)
        owner.scaler.step(owner.optimizer)
        owner.scaler.update()

    def _loss_metrics(self, losses: _Losses) -> torch.Tensor:
        with torch.no_grad():
            approximate_kl = ((losses.ratio - 1.0) - losses.log_ratio).mean()
            clip_fraction = ((losses.ratio - 1.0).abs() > self.owner.clip_epsilon).float().mean()
        return torch.stack(
            (losses.policy, losses.value, losses.entropy.mean(), approximate_kl, clip_fraction)
        ).detach()

    @staticmethod
    def _metrics(outcome: _TrainingOutcome) -> Mapping[str, float]:
        means = outcome.totals / outcome.updates
        return {
            "loss/policy": float(means[0].item()),
            "loss/value": float(means[1].item()),
            "state/entropy": float(means[2].item()),
            "state/approx_kl": float(means[3].item()),
            "state/clip_fraction": float(means[4].item()),
            "state/early_stop": float(outcome.stopped_early),
        }


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor after feature collation")
    return value


def _behavior_tensor(batch: TrainingBatch, key: str, device: torch.device) -> torch.Tensor:
    value = batch.metadata.get(key)
    if value is None:
        raise ValueError(f"PPO batch is missing {key} captured by EpisodeCollector")
    return torch.as_tensor(value, dtype=torch.float32, device=device)
