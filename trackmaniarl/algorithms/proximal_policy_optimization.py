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
    def __init__(self, actor: nn.Module, value: nn.Module, device: torch.device) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.value = deepcopy(value).to(device).eval()
        self.device = device

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
        with torch.no_grad():
            sample = cast(Any, self.actor).sample_with_latent
            action, log_probability, latent_action = sample(prepared, deterministic=deterministic)
            value = self.value(prepared)
        if log_probability.numel() != 1 or value.numel() != 1:
            raise ValueError("PPO rollout policy expects one unbatched observation")
        return action.detach().cpu().numpy(), {
            "_trackmaniarl_behavior_log_probability": float(log_probability.item()),
            "_trackmaniarl_behavior_value": float(value.item()),
            "_trackmaniarl_behavior_latent_action": latent_action.detach().cpu().numpy(),
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
        with torch.no_grad():
            next_values = self.model.value(next_observations)
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
        return self._optimize_epochs(
            observations,
            latent_actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        )

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
        return _PpoPolicy(self.model.actor, self.model.value, self.device)

    def state_dict(self) -> Mapping[str, Any]:
        assert self.model is not None
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._restore_rng(cast(Mapping[str, Any], state.get("rng", {})))


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
