"""Continuous-control Proximal Policy Optimization for bounded racing actions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from trackmaniarl.algorithms._torch import TorchLearnerBase
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import sanitize_finite, tree_map, tree_to_device


class _PpoPolicy:
    def __init__(
        self,
        actor: nn.Module,
        value: nn.Module,
        device: torch.device,
        observation_normalizer: _ObservationNormalizer,
    ) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.value = deepcopy(value).to(device).eval()
        self.device = device
        self.observation_normalizer = deepcopy(observation_normalizer)

    def act(self, observation: Any, *, deterministic: bool = False) -> np.ndarray[Any, Any]:
        action, _ = self._sample(observation, deterministic=deterministic)
        return action

    def act_with_info(
        self, observation: Any, *, deterministic: bool = False
    ) -> tuple[np.ndarray[Any, Any], Mapping[str, Any]]:
        action, info = self._sample(observation, deterministic=deterministic)
        return action, info

    def _sample(
        self, observation: Any, *, deterministic: bool
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        prepared = tree_to_device(sanitize_finite(observation), self.device)
        prepared = self.observation_normalizer.normalize(prepared, sample_dimensions=0)
        prepared = tree_map(
            lambda leaf: leaf.unsqueeze(0) if isinstance(leaf, torch.Tensor) else leaf,
            prepared,
        )
        with torch.no_grad():
            sample = cast(Any, self.actor).sample_with_latent
            action, log_probability, latent_action = sample(prepared, deterministic=deterministic)
            value = self.value(prepared)
        if log_probability.numel() != 1 or value.numel() != 1:
            raise ValueError("PPO rollout policy expects one unbatched observation")
        return action[0].detach().cpu().numpy(), {
            "_trackmaniarl_behavior_log_probability": float(log_probability.item()),
            "_trackmaniarl_behavior_value": float(value.item()),
            "_trackmaniarl_behavior_latent_action": latent_action[0].detach().cpu().numpy(),
        }

    def export_state(self) -> Mapping[str, Any]:
        return {
            **{f"actor.{key}": value for key, value in self.actor.state_dict().items()},
            **{f"value.{key}": value for key, value in self.value.state_dict().items()},
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        actor = {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        value = {
            key.removeprefix("value."): item
            for key, item in state.items()
            if key.startswith("value.")
        }
        self.actor.load_state_dict(actor)
        self.value.load_state_dict(value)


class ProximalPolicyOptimization(TorchLearnerBase):
    """PPO with GAE, value clipping, KL stopping and bounded Gaussian actions."""

    accepted_model_contracts = frozenset({ModelContract.CONTINUOUS_ACTOR_VALUE})
    on_policy = True

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_factory: Any | None = None,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        max_gradient_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 256,
        target_kl: float | None = 0.02,
        normalize_observations: bool = True,
        observation_clip: float = 10.0,
        normalize_rewards: bool = True,
        reward_clip: float = 10.0,
        device: str | None = None,
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            model, model_factory=model_factory, device=device, execution=execution, seed=seed
        )
        if learning_rate <= 0.0 or max_gradient_norm <= 0.0:
            raise ValueError("learning_rate and max_gradient_norm must be positive")
        if not 0.0 < clip_epsilon < 1.0 or not 0.0 < value_clip_epsilon < 1.0:
            raise ValueError("PPO clipping epsilons must be between zero and one")
        if not 0.0 <= gae_lambda <= 1.0 or entropy_coefficient < 0.0 or value_coefficient < 0.0:
            raise ValueError("Invalid PPO GAE or loss coefficient")
        if update_epochs < 1 or minibatch_size < 1 or (target_kl is not None and target_kl <= 0.0):
            raise ValueError("Invalid PPO update schedule")
        if observation_clip <= 0.0 or reward_clip <= 0.0:
            raise ValueError("PPO normalization clips must be positive")
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.max_gradient_norm = max_gradient_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.observation_normalizer = _ObservationNormalizer(
            enabled=normalize_observations, clip=observation_clip
        )
        self.reward_normalizer = _RewardNormalizer(enabled=normalize_rewards, clip=reward_clip)
        self.total_transitions: int | None = None
        self.processed_transitions = 0

    def setup(self, context: Mapping[str, Any]) -> None:
        total_transitions = context.get("total_transitions")
        self.total_transitions = int(total_transitions) if total_transitions is not None else None
        super().setup(context)

    def _setup_model(self) -> None:
        assert self.model is not None
        if not all(hasattr(self.model, name) for name in ("actor", "value")):
            raise TypeError("PPO model must expose actor and value modules")
        required = ("sample_with_latent", "evaluate_latent_actions")
        if not all(callable(getattr(self.model.actor, name, None)) for name in required):
            raise TypeError(
                "PPO actor must expose sample_with_latent() and evaluate_latent_actions()"
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

    def update(self, batch: TrainingBatch) -> Mapping[str, float]:
        batch = self._batch(batch)
        observations = _float_tensor_tree(batch.observations, "observations")
        next_observations = _float_tensor_tree(batch.next_observations, "next_observations")
        rewards = self._tensor(batch.rewards, "rewards").float()
        discounts = self._tensor(batch.bootstrap_discounts, "bootstrap_discounts").float()
        terminated = self._tensor(batch.terminated, "terminated").bool()
        truncated = self._tensor(batch.truncated, "truncated").bool()
        old_log_probabilities = self._behavior_tensor(batch, "behavior_log_probabilities")
        old_values = self._behavior_tensor(batch, "behavior_values")
        latent_actions = self._behavior_tensor(batch, "behavior_latent_actions")
        sample_dimensions = rewards.ndim
        normalized_observations = self.observation_normalizer.normalize(
            observations, sample_dimensions
        )
        normalized_next_observations = self.observation_normalizer.normalize(
            next_observations, sample_dimensions
        )
        rewards = self.reward_normalizer.normalize(rewards, terminated | truncated, discounts)
        self._anneal_learning_rate()
        with torch.no_grad():
            next_values = self.model.value(normalized_next_observations)
            advantages, returns = generalized_advantage_estimate(
                rewards,
                old_values,
                next_values,
                discounts,
                terminated | truncated,
                self.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp_min(1e-8)
        metrics = self._optimize_epochs(
            normalized_observations,
            latent_actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        )
        self.observation_normalizer.update(observations, sample_dimensions)
        self.processed_transitions += rewards.numel()
        return {**metrics, "state/learning_rate": self._current_learning_rate()}

    def _optimize_epochs(
        self,
        observations: Any,
        latent_actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Mapping[str, float]:
        observations = _flatten_samples(observations, returns)
        latent_actions = _flatten_samples(latent_actions, returns)
        old_log_probabilities = old_log_probabilities.reshape(-1)
        old_values = old_values.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        totals = torch.zeros(5, device=self.device)
        updates = 0
        stopped_early = False
        for _ in range(self.update_epochs):
            permutation = torch.randperm(returns.numel(), device=self.device)
            epoch_kl = 0.0
            epoch_updates = 0
            for start in range(0, returns.numel(), self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                metrics = self._minibatch_step(
                    _index_samples(observations, indices),
                    latent_actions[indices],
                    old_log_probabilities[indices],
                    old_values[indices],
                    advantages[indices],
                    returns[indices],
                )
                totals += metrics
                updates += 1
                epoch_kl += float(metrics[3].item())
                epoch_updates += 1
            if self.target_kl is not None and epoch_kl / epoch_updates > self.target_kl:
                stopped_early = True
                break
        means = totals / updates
        return {
            "loss/policy": float(means[0].item()),
            "loss/value": float(means[1].item()),
            "state/entropy": float(means[2].item()),
            "state/approx_kl": float(means[3].item()),
            "state/clip_fraction": float(means[4].item()),
            "state/early_stop": float(stopped_early),
        }

    def _minibatch_step(
        self,
        observations: Any,
        latent_actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        if self.scaler is None:
            raise RuntimeError("Learner setup() must be called before update()")
        with self.autocast():
            log_probabilities, entropy = self.model.actor.evaluate_latent_actions(
                observations, latent_actions
            )
            values = self.model.value(observations)
            log_ratio = log_probabilities - old_log_probabilities
            ratio = log_ratio.exp()
            surrogate = torch.minimum(
                ratio * advantages,
                ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages,
            )
            policy_loss = -surrogate.mean()
            clipped_values = old_values + (values - old_values).clamp(
                -self.value_clip_epsilon, self.value_clip_epsilon
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (values - returns).square(), (clipped_values - returns).square()
                ).mean()
            )
            loss = (
                policy_loss
                + self.value_coefficient * value_loss
                - self.entropy_coefficient * entropy.mean()
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
        return torch.stack(
            (policy_loss, value_loss, entropy.mean(), approximate_kl, clip_fraction)
        ).detach()

    def _behavior_tensor(self, batch: TrainingBatch, key: str) -> torch.Tensor:
        value = batch.metadata.get(key)
        if value is None:
            raise ValueError(
                f"PPO requires {key} captured by EpisodeCollector; do not train it on legacy replay"
            )
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def policy(self) -> _PpoPolicy:
        assert self.model is not None
        return _PpoPolicy(
            self.model.actor,
            self.model.value,
            self.device,
            self.observation_normalizer,
        )

    def reset_environment_state(self) -> None:
        self.reward_normalizer.discounted_returns = None

    def state_dict(self) -> Mapping[str, Any]:
        assert self.model is not None
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "observation_normalizer": self.observation_normalizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "processed_transitions": self.processed_transitions,
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.observation_normalizer.load_state_dict(
            cast(Mapping[str, Any], state.get("observation_normalizer", {}))
        )
        self.reward_normalizer.load_state_dict(
            cast(Mapping[str, Any], state.get("reward_normalizer", {}))
        )
        self.processed_transitions = int(state.get("processed_transitions", 0))
        self._restore_rng(cast(Mapping[str, Any], state.get("rng", {})))

    def _anneal_learning_rate(self) -> None:
        if self.total_transitions is None:
            return
        fraction = 1.0 - min(1.0, self.processed_transitions / self.total_transitions)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * fraction

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


class _ObservationNormalizer:
    def __init__(self, *, enabled: bool, clip: float) -> None:
        self.enabled = enabled
        self.clip = clip
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
        if not state:
            return
        self._moments = {
            str(key): _Moments.from_state(cast(Mapping[str, Any], value))
            for key, value in cast(Mapping[str, Any], state["moments"]).items()
        }


class _RewardNormalizer:
    def __init__(self, *, enabled: bool, clip: float) -> None:
        self.enabled = enabled
        self.clip = clip
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
            self._normalize_step(sequence, ends, normalized, step, gamma)
        return normalized if rewards.ndim > 1 else normalized[0]

    def _normalize_step(
        self,
        rewards: torch.Tensor,
        ends: torch.Tensor,
        normalized: torch.Tensor,
        step: int,
        gamma: float,
    ) -> None:
        assert self.discounted_returns is not None
        self.discounted_returns.mul_(gamma).add_(rewards[:, step])
        values = self.discounted_returns.detach().cpu()
        self.moments.update(values.mean(), values.var(unbiased=False), values.numel())
        scale = float((self.moments.variance + 1e-8).sqrt().item())
        normalized[:, step] = (rewards[:, step] / scale).clamp(-self.clip, self.clip)
        self.discounted_returns.masked_fill_(ends[:, step], 0.0)

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
        if not state:
            return
        self.moments = _Moments.from_state(cast(Mapping[str, Any], state["moments"]))
        value = state.get("discounted_returns")
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


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_discounts: torch.Tensor,
    episode_ends: torch.Tensor,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute time-limit-aware GAE over flat transitions or ordered sequences."""

    sequence = rewards.ndim > 1
    rewards_2d = rewards if sequence else rewards[:, None]
    values_2d = values if sequence else values[:, None]
    next_values_2d = next_values if sequence else next_values[:, None]
    discounts_2d = bootstrap_discounts if sequence else bootstrap_discounts[:, None]
    ends_2d = episode_ends if sequence else episode_ends[:, None]
    deltas = rewards_2d + discounts_2d * next_values_2d - values_2d
    advantages = torch.zeros_like(deltas)
    running = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    for step in range(deltas.shape[1] - 1, -1, -1):
        continuation = (~ends_2d[:, step]).to(deltas.dtype)
        running = deltas[:, step] + discounts_2d[:, step] * gae_lambda * continuation * running
        advantages[:, step] = running
    returns = advantages + values_2d
    if sequence:
        return advantages, returns
    return advantages[:, 0], returns[:, 0]


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
