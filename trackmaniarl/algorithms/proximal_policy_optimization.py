"""Continuous-control Proximal Policy Optimization for bounded racing actions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Unpack, cast

import numpy as np
import torch
from torch import nn

from trackmaniarl.algorithms._torch import TorchLearnerBase
from trackmaniarl.algorithms.ppo_config import PPOConfig, PPOOptions
from trackmaniarl.algorithms.ppo_support import (
    NormalizerConfig,
    _ObservationNormalizer,
    _RewardNormalizer,
)
from trackmaniarl.algorithms.ppo_update import PPOUpdateOwner, PPOUpdater
from trackmaniarl.core.contracts import ModelContract, PolicyMode
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import sanitize_finite, tree_map, tree_to_device


class _PpoPolicy:
    def __init__(
        self, model: nn.Module, device: torch.device, normalizer: _ObservationNormalizer
    ) -> None:
        self.actor = deepcopy(cast(Any, model).actor).to(device).eval()
        self.value = deepcopy(cast(Any, model).value).to(device).eval()
        self.device = device
        self.observation_normalizer = deepcopy(normalizer)

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> np.ndarray[Any, Any]:
        action, _ = self._sample(observation, mode)
        return action

    def act_with_info(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
    ) -> tuple[np.ndarray[Any, Any], Mapping[str, Any]]:
        action, info = self._sample(observation, mode)
        return action, info

    def _sample(
        self, observation: Any, mode: PolicyMode
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        prepared = self._prepare_observation(observation)
        with torch.no_grad():
            sample = cast(Any, self.actor).sample_with_latent
            action, log_probability, latent_action = sample(prepared, mode=mode)
            value = self.value(prepared)
        if log_probability.numel() != 1 or value.numel() != 1:
            raise ValueError("PPO rollout policy expects one unbatched observation")
        return action[0].detach().cpu().numpy(), self._rollout_info(
            log_probability, value, latent_action
        )

    def _prepare_observation(self, observation: Any) -> Any:
        prepared = tree_to_device(sanitize_finite(observation), self.device)
        prepared = self.observation_normalizer.normalize(prepared, sample_dimensions=0)
        return tree_map(
            lambda leaf: leaf.unsqueeze(0) if isinstance(leaf, torch.Tensor) else leaf,
            prepared,
        )

    @staticmethod
    def _rollout_info(
        log_probability: torch.Tensor, value: torch.Tensor, latent_action: torch.Tensor
    ) -> dict[str, Any]:
        return {
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
        expected = {
            *(f"actor.{key}" for key in self.actor.state_dict()),
            *(f"value.{key}" for key in self.value.state_dict()),
        }
        if set(state) != expected:
            raise ValueError("PPO policy state does not match the actor-value model")
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
        **options: Unpack[PPOOptions],
    ) -> None:
        config = PPOConfig(**options)
        super().__init__(
            model,
            model_factory=config.model_factory,
            execution=config.execution,
            seed=config.seed,
        )
        config.validate()
        self._configure(config)
        self._configure_normalizers(config)
        self.total_transitions: int | None = None
        self.processed_transitions = 0

    def _configure_normalizers(self, config: PPOConfig) -> None:
        self.observation_normalizer = _ObservationNormalizer(
            NormalizerConfig(config.normalize_observations, config.observation_clip)
        )
        self.reward_normalizer = _RewardNormalizer(
            NormalizerConfig(config.normalize_rewards, config.reward_clip)
        )

    def _configure(self, config: PPOConfig) -> None:
        self.learning_rate = config.learning_rate
        self.clip_epsilon = config.clip_epsilon
        self.value_clip_epsilon = config.value_clip_epsilon
        self.gae_lambda = config.gae_lambda
        self.entropy_coefficient = config.entropy_coefficient
        self.value_coefficient = config.value_coefficient
        self.max_gradient_norm = config.max_gradient_norm
        self.update_epochs = config.update_epochs
        self.minibatch_size = config.minibatch_size
        self.target_kl = config.target_kl

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
        self._anneal_learning_rate()
        metrics, transition_count = PPOUpdater(cast(PPOUpdateOwner, self)).update(batch)
        self.processed_transitions += transition_count
        return {**metrics, "state/learning_rate": self._current_learning_rate()}

    def policy(self) -> _PpoPolicy:
        assert self.model is not None
        return _PpoPolicy(
            self.model,
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
            "scaler": self._scaler_state(),
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.observation_normalizer.load_state_dict(
            cast(Mapping[str, Any], state["observation_normalizer"])
        )
        self.reward_normalizer.load_state_dict(cast(Mapping[str, Any], state["reward_normalizer"]))
        self.processed_transitions = int(state["processed_transitions"])
        self._restore_scaler(state["scaler"])
        self._restore_rng(state["rng"])

    def _anneal_learning_rate(self) -> None:
        if self.total_transitions is None:
            return
        fraction = 1.0 - min(1.0, self.processed_transitions / self.total_transitions)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * fraction

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
