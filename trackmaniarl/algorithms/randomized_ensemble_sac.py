"""Randomized Ensemble Double-Q SAC learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from trackmaniarl.algorithms._torch import (
    TorchLearnerBase,
    TorchPolicy,
    polyak_update,
    weighted_mean,
)
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch


class RandomizedEnsembleSAC(TorchLearnerBase):
    """REDQ-SAC with a random target subset and an explicit policy-update interval."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_factory: Any | None = None,
        target_tau: float = 0.005,
        learning_rate: float = 3e-4,
        entropy_coefficient: float = 0.2,
        target_subset_size: int = 2,
        policy_update_interval: int = 20,
        device: str | None = None,
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            model,
            model_factory=model_factory,
            device=device,
            execution=execution,
            seed=seed,
        )
        if not 0.0 < target_tau <= 1.0 or learning_rate <= 0.0 or entropy_coefficient <= 0.0:
            raise ValueError("target_tau, learning_rate, and entropy_coefficient must be positive")
        if target_subset_size < 1 or policy_update_interval < 1:
            raise ValueError("target_subset_size and policy_update_interval must be positive")
        self.target_tau = target_tau
        self.learning_rate = learning_rate
        self.entropy_coefficient = entropy_coefficient
        self.target_subset_size = target_subset_size
        self.policy_update_interval = policy_update_interval
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
        self._target_rng = torch.Generator(device=self.device).manual_seed(self.seed)

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        with self.autocast():
            return self._update(batch)

    def _update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert self.model is not None
        batch = self._batch(batch)
        observations = self._tensor(batch.observations, "observations")
        actions = self._tensor(batch.actions, "actions").float()
        rewards = self._tensor(batch.rewards, "rewards").float().reshape(-1)
        next_observations = self._tensor(batch.next_observations, "next_observations")
        discounts = (
            self._tensor(batch.bootstrap_discounts, "bootstrap_discounts").float().reshape(-1)
        )
        weights = (
            batch.importance_weights if isinstance(batch.importance_weights, torch.Tensor) else None
        )
        with torch.no_grad():
            next_actions, next_log_probabilities = self.model.actor(next_observations)
            subset = torch.randperm(
                len(self.model.critics), generator=self._target_rng, device=self.device
            )[: self.target_subset_size]
            target_values = torch.stack(
                [
                    self.target_model.critics[index](next_observations, next_actions)
                    for index in subset
                ]
            ).amin(dim=0)
            targets = rewards + discounts * (
                target_values - self.entropy_coefficient * next_log_probabilities
            )
        predictions = torch.stack([critic(observations, actions) for critic in self.model.critics])
        td_errors = (predictions.mean(dim=0) - targets).detach().abs()
        critic_loss = weighted_mean(
            (predictions - targets.unsqueeze(0)).square().mean(dim=0), weights
        )
        self._optimize(critic_loss, self.critic_optimizer)
        self.update_count += 1
        actor_loss = torch.zeros((), device=self.device)
        if self.update_count % self.policy_update_interval == 0:
            for critic in self.model.critics:
                critic.requires_grad_(False)
            policy_actions, log_probabilities = self.model.actor(observations)
            values = torch.stack(
                [critic(observations, policy_actions) for critic in self.model.critics]
            ).mean(dim=0)
            actor_loss = (self.entropy_coefficient * log_probabilities - values).mean()
            self._optimize(actor_loss, self.actor_optimizer)
            for critic in self.model.critics:
                critic.requires_grad_(True)
        polyak_update(self.model, self.target_model, self.target_tau)
        return (
            {"loss/actor": float(actor_loss.item()), "loss/critic": float(critic_loss.item())},
            PriorityUpdate(batch.transition_ids, td_errors.cpu().tolist()),
        )

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
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_count = int(state["update_count"])
        if state.get("target_rng") is not None:
            self._target_rng.set_state(state["target_rng"])
        self._restore_rng(state.get("rng", {}))
