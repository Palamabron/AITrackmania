"""Standard twin-critic Soft Actor-Critic learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Unpack

import torch
from torch import nn

from trackmaniarl.algorithms._torch import (
    TorchLearnerBase,
    TorchPolicy,
    evaluated_actor_state,
    polyak_update,
    weighted_mean,
)
from trackmaniarl.algorithms.sac_config import SACConfig, SACOptions
from trackmaniarl.algorithms.sac_support import (
    EntropyConfig,
    EntropyRestoreTarget,
    SACBatch,
    alpha_value,
    continuous_batch,
    entropy_state,
    freeze_modules,
    restore_entropy_state,
    scalar_batch_output,
    unfreeze_modules,
)
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch


@dataclass(frozen=True, slots=True)
class _CriticStep:
    loss: torch.Tensor
    td_error: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SACUpdate:
    critic: _CriticStep
    actor_loss: torch.Tensor
    alpha_loss: torch.Tensor
    alpha: torch.Tensor


class SoftActorCritic(TorchLearnerBase):
    """SAC v2 with explicit target semantics, optional temperature learning and PER feedback."""

    accepted_model_contracts = frozenset({ModelContract.CONTINUOUS_ACTOR_CRITIC})
    supports_sequence_training = False

    def __init__(
        self,
        model: nn.Module | None = None,
        **options: Unpack[SACOptions],
    ) -> None:
        config = SACConfig(**options)
        super().__init__(
            model,
            model_factory=config.model_factory,
            execution=config.execution,
            seed=config.seed,
        )
        config.validate()
        self.target_tau = config.target_tau
        self.learning_rate = config.learning_rate
        self.initial_entropy_coefficient = config.entropy_coefficient
        self.target_entropy = config.target_entropy
        self.entropy_mode = "learned" if config.learn_entropy_coefficient else "fixed"

    def _setup_model(self) -> None:
        assert self.model is not None
        self._validate_model()
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self._setup_optimizers()
        config = EntropyConfig(
            self.initial_entropy_coefficient, self.learning_rate, self.entropy_mode
        )
        self.log_alpha, self.alpha_optimizer = entropy_state(config, self.device)

    def _validate_model(self) -> None:
        assert self.model is not None
        for name in ("actor", "q1", "q2"):
            if not hasattr(self.model, name):
                raise TypeError(f"SAC model is missing required component {name!r}")

    def _setup_optimizers(self) -> None:
        assert self.model is not None
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            lr=self.learning_rate,
        )

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        with self.autocast():
            return self._update(batch)

    def _update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert self.model is not None
        prepared = continuous_batch(self._batch(batch))
        alpha = alpha_value(self.log_alpha, self.initial_entropy_coefficient, self.device)
        critic = self._critic_step(prepared, alpha)
        self._optimize(critic.loss, self.critic_optimizer)
        actor_loss, log_probabilities = self._actor_step(prepared, alpha)
        alpha_loss = self._entropy_loss(prepared.actions, log_probabilities)
        polyak_update(self.model, self.target_model, self.target_tau)
        return self._update_result(prepared, _SACUpdate(critic, actor_loss, alpha_loss, alpha))

    def _critic_step(self, batch: SACBatch, alpha: torch.Tensor) -> _CriticStep:
        targets = self._critic_targets(batch, alpha)
        batch_size = batch.rewards.shape[0]
        q1 = scalar_batch_output(
            self.model.q1(batch.observations, batch.actions), "q1 output", batch_size
        )
        q2 = scalar_batch_output(
            self.model.q2(batch.observations, batch.actions), "q2 output", batch_size
        )
        td_error = (0.5 * (q1 + q2) - targets).detach().abs()
        losses = (q1 - targets).square() + (q2 - targets).square()
        return _CriticStep(weighted_mean(losses, batch.weights), td_error)

    def _critic_targets(self, batch: SACBatch, alpha: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_probabilities = self.model.actor(batch.next_observations)
            batch_size = batch.rewards.shape[0]
            next_log_probabilities = scalar_batch_output(
                next_log_probabilities, "actor log probabilities", batch_size
            )
            next_values = self._target_values(batch, next_actions)
            result = batch.rewards + batch.discounts * (
                next_values - alpha * next_log_probabilities
            )
            return result

    def _target_values(self, batch: SACBatch, actions: torch.Tensor) -> torch.Tensor:
        batch_size = batch.rewards.shape[0]
        q1 = scalar_batch_output(
            self.target_model.q1(batch.next_observations, actions),
            "target q1 output",
            batch_size,
        )
        q2 = scalar_batch_output(
            self.target_model.q2(batch.next_observations, actions),
            "target q2 output",
            batch_size,
        )
        return torch.minimum(q1, q2)

    def _actor_step(
        self, batch: SACBatch, alpha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critics = (self.model.q1, self.model.q2)
        freeze_modules(critics)
        policy_actions, log_probabilities = self.model.actor(batch.observations)
        batch_size = batch.rewards.shape[0]
        log_probabilities = scalar_batch_output(
            log_probabilities, "actor log probabilities", batch_size
        )
        policy_values = self._policy_values(batch, policy_actions)
        actor_loss = (alpha * log_probabilities - policy_values).mean()
        self._optimize(actor_loss, self.actor_optimizer)
        unfreeze_modules(critics)
        return actor_loss, log_probabilities

    def _policy_values(self, batch: SACBatch, actions: torch.Tensor) -> torch.Tensor:
        batch_size = batch.rewards.shape[0]
        q1 = scalar_batch_output(
            self.model.q1(batch.observations, actions), "q1 output", batch_size
        )
        q2 = scalar_batch_output(
            self.model.q2(batch.observations, actions), "q2 output", batch_size
        )
        return torch.minimum(q1, q2)

    def _entropy_loss(self, actions: torch.Tensor, log_probabilities: torch.Tensor) -> torch.Tensor:
        alpha_loss = torch.zeros((), device=self.device)
        if self.log_alpha is None:
            return alpha_loss
        target = (
            self.target_entropy if self.target_entropy is not None else -float(actions.shape[-1])
        )
        alpha_loss = -(self.log_alpha * (log_probabilities.detach() + target)).mean()
        assert self.alpha_optimizer is not None
        self._optimize(alpha_loss, self.alpha_optimizer)
        return alpha_loss

    @staticmethod
    def _update_result(
        batch: SACBatch, update: _SACUpdate
    ) -> tuple[Mapping[str, float], PriorityUpdate]:
        metrics = {
            "loss/actor": float(update.actor_loss.item()),
            "loss/critic": float(update.critic.loss.item()),
            "loss/entropy": float(update.alpha_loss.item()),
            "state/alpha": float(update.alpha.item()),
        }
        priorities = update.critic.td_error.detach().cpu().tolist()
        return metrics, PriorityUpdate(batch.source.transition_ids, priorities)

    def policy(self) -> TorchPolicy:
        assert self.model is not None
        return TorchPolicy(self.model.actor, self.device)

    def state_dict(self) -> Mapping[str, Any]:
        assert self.model is not None
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "scaler": self._scaler_state(),
            "rng": self._rng_state(),
        }

    def state_dict_for_policy(self, policy_state: Mapping[str, Any]) -> Mapping[str, Any]:
        assert self.model is not None
        state = evaluated_actor_state(self.state_dict(), self.model, policy_state)
        state["actor_optimizer"] = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.learning_rate
        ).state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        entropy = EntropyRestoreTarget(self.log_alpha, self.alpha_optimizer, self.device)
        restore_entropy_state(entropy, state)
        self._restore_scaler(state["scaler"])
        self._restore_rng(state["rng"])
