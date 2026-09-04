"""Randomized Ensemble Double-Q SAC learner."""

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
from trackmaniarl.algorithms.sac_config import REDQConfig, REDQOptions
from trackmaniarl.algorithms.sac_support import (
    SACBatch,
    continuous_batch,
    freeze_modules,
    scalar_batch_output,
    unfreeze_modules,
)
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch


@dataclass(frozen=True, slots=True)
class _REDQCriticStep:
    loss: torch.Tensor
    td_errors: torch.Tensor


class RandomizedEnsembleSAC(TorchLearnerBase):
    """REDQ-SAC with a random target subset and an explicit policy-update interval."""

    accepted_model_contracts = frozenset({ModelContract.ENSEMBLE_ACTOR_CRITIC})
    supports_sequence_training = False

    def __init__(
        self,
        model: nn.Module | None = None,
        **options: Unpack[REDQOptions],
    ) -> None:
        config = REDQConfig(**options)
        super().__init__(
            model,
            model_factory=config.model_factory,
            execution=config.execution,
            seed=config.seed,
        )
        config.validate()
        self._configure(config)

    def _configure(self, config: REDQConfig) -> None:
        self.target_tau = config.target_tau
        self.learning_rate = config.learning_rate
        self.entropy_coefficient = config.entropy_coefficient
        self.target_subset_size = config.target_subset_size
        self.policy_update_interval = config.policy_update_interval
        self.update_count = 0

    def _setup_model(self) -> None:
        assert self.model is not None
        if not hasattr(self.model, "actor") or not hasattr(self.model, "critics"):
            raise TypeError("REDQ model must expose actor and critics")
        if len(self.model.critics) < self.target_subset_size:
            raise ValueError("target_subset_size exceeds ensemble size")
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critics.parameters(), lr=self.learning_rate
        )
        self._target_rng = torch.Generator().manual_seed(self.seed)

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        with self.autocast():
            return self._update(batch)

    def _update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert self.model is not None
        prepared = continuous_batch(self._batch(batch))
        critic = self._critic_step(prepared)
        self._optimize(critic.loss, self.critic_optimizer)
        self.update_count += 1
        actor_loss = self._actor_step(prepared)
        polyak_update(self.model, self.target_model, self.target_tau)
        metrics = {"loss/actor": float(actor_loss.item()), "loss/critic": float(critic.loss.item())}
        priorities = critic.td_errors.cpu().tolist()
        return metrics, PriorityUpdate(prepared.source.transition_ids, priorities)

    def _critic_targets(self, batch: SACBatch) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_probabilities = self.model.actor(batch.next_observations)
            batch_size = batch.rewards.shape[0]
            next_log_probabilities = scalar_batch_output(
                next_log_probabilities, "actor log probabilities", batch_size
            )
            target_values = self._target_values(batch, next_actions)
            result = batch.rewards + batch.discounts * (
                target_values - self.entropy_coefficient * next_log_probabilities
            )
            return result

    def _target_values(self, batch: SACBatch, actions: torch.Tensor) -> torch.Tensor:
        subset = torch.randperm(len(self.model.critics), generator=self._target_rng)[
            : self.target_subset_size
        ].tolist()
        values = [
            scalar_batch_output(
                self.target_model.critics[index](batch.next_observations, actions),
                f"target critic {index} output",
                batch.rewards.shape[0],
            )
            for index in subset
        ]
        return torch.stack(values).amin(dim=0)

    def _critic_step(self, batch: SACBatch) -> _REDQCriticStep:
        targets = self._critic_targets(batch)
        batch_size = batch.rewards.shape[0]
        predictions = torch.stack(
            [
                scalar_batch_output(
                    critic(batch.observations, batch.actions),
                    f"critic {index} output",
                    batch_size,
                )
                for index, critic in enumerate(self.model.critics)
            ]
        )
        td_errors = (predictions.mean(dim=0) - targets).detach().abs()
        loss = weighted_mean(
            (predictions - targets.unsqueeze(0)).square().mean(dim=0), batch.weights
        )
        return _REDQCriticStep(loss, td_errors)

    def _actor_step(self, batch: SACBatch) -> torch.Tensor:
        actor_loss = torch.zeros((), device=self.device)
        if self.update_count % self.policy_update_interval != 0:
            return actor_loss
        freeze_modules(self.model.critics)
        policy_actions, log_probabilities = self.model.actor(batch.observations)
        batch_size = batch.rewards.shape[0]
        log_probabilities = scalar_batch_output(
            log_probabilities, "actor log probabilities", batch_size
        )
        values = self._policy_values(batch, policy_actions)
        actor_loss = (self.entropy_coefficient * log_probabilities - values).mean()
        self._optimize(actor_loss, self.actor_optimizer)
        unfreeze_modules(self.model.critics)
        return actor_loss

    def _policy_values(self, batch: SACBatch, actions: torch.Tensor) -> torch.Tensor:
        values = [
            scalar_batch_output(
                critic(batch.observations, actions),
                f"critic {index} output",
                batch.rewards.shape[0],
            )
            for index, critic in enumerate(self.model.critics)
        ]
        return torch.stack(values).mean(dim=0)

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
            "update_count": self.update_count,
            "target_rng": self._target_rng.get_state(),
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
        self.update_count = int(state["update_count"])
        self._target_rng.set_state(state["target_rng"])
        self._restore_scaler(state["scaler"])
        self._restore_rng(state["rng"])
