"""Standard twin-critic Soft Actor-Critic learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from trackmaniarl.algorithms._torch import (
    TorchLearnerBase,
    TorchPolicy,
    evaluated_actor_state,
    polyak_update,
    weighted_mean,
)
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch


class SoftActorCritic(TorchLearnerBase):
    """SAC v2 with explicit target semantics, optional temperature learning and PER feedback."""

    accepted_model_contracts = frozenset({ModelContract.CONTINUOUS_ACTOR_CRITIC})
    supports_sequence_training = False

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_factory: Any | None = None,
        target_tau: float = 0.005,
        learning_rate: float = 3e-4,
        entropy_coefficient: float = 0.2,
        target_entropy: float | None = None,
        learn_entropy_coefficient: bool = True,
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
        self.target_tau = target_tau
        self.learning_rate = learning_rate
        self.initial_entropy_coefficient = entropy_coefficient
        self.target_entropy = target_entropy
        self.learn_entropy_coefficient = learn_entropy_coefficient

    def _setup_model(self) -> None:
        assert self.model is not None
        for name in ("actor", "q1", "q2"):
            if not hasattr(self.model, name):
                raise TypeError(f"SAC model is missing required component {name!r}")
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            lr=self.learning_rate,
        )
        self.log_alpha: torch.Tensor | None
        self.alpha_optimizer: torch.optim.Optimizer | None
        if self.learn_entropy_coefficient:
            self.log_alpha = (
                torch.tensor(self.initial_entropy_coefficient, device=self.device)
                .log()
                .requires_grad_()
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

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
        alpha = (
            self.log_alpha.detach().exp()
            if self.log_alpha is not None
            else torch.tensor(self.initial_entropy_coefficient, device=self.device)
        )
        with torch.no_grad():
            next_actions, next_log_probabilities = self.model.actor(next_observations)
            next_values = torch.minimum(
                self.target_model.q1(next_observations, next_actions),
                self.target_model.q2(next_observations, next_actions),
            )
            targets = rewards + discounts * (next_values - alpha * next_log_probabilities)
        q1 = self.model.q1(observations, actions)
        q2 = self.model.q2(observations, actions)
        td_error = (0.5 * (q1 + q2) - targets).detach().abs()
        critic_loss = weighted_mean((q1 - targets).square() + (q2 - targets).square(), weights)
        self._optimize(critic_loss, self.critic_optimizer)
        for critic in (self.model.q1, self.model.q2):
            critic.requires_grad_(False)
        policy_actions, log_probabilities = self.model.actor(observations)
        policy_values = torch.minimum(
            self.model.q1(observations, policy_actions), self.model.q2(observations, policy_actions)
        )
        actor_loss = (alpha * log_probabilities - policy_values).mean()
        self._optimize(actor_loss, self.actor_optimizer)
        for critic in (self.model.q1, self.model.q2):
            critic.requires_grad_(True)
        alpha_loss = torch.zeros((), device=self.device)
        if self.log_alpha is not None:
            target_entropy = (
                self.target_entropy
                if self.target_entropy is not None
                else -float(actions.shape[-1])
            )
            alpha_loss = -(self.log_alpha * (log_probabilities.detach() + target_entropy)).mean()
            assert self.alpha_optimizer is not None
            self._optimize(alpha_loss, self.alpha_optimizer)
        polyak_update(self.model, self.target_model, self.target_tau)
        return (
            {
                "loss/actor": float(actor_loss.item()),
                "loss/critic": float(critic_loss.item()),
                "loss/entropy": float(alpha_loss.item()),
                "state/alpha": float(alpha.item()),
            },
            PriorityUpdate(batch.transition_ids, td_error.detach().cpu().tolist()),
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
        if self.log_alpha is not None and state.get("log_alpha") is not None:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        if self.alpha_optimizer is not None and state.get("alpha_optimizer") is not None:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self._restore_scaler(state.get("scaler"))
        self._restore_rng(state.get("rng", {}))
