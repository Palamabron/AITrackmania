"""SD-SAC-inspired discrete SAC with conservative double-Q targets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Unpack, cast

import torch
from torch import nn

from trackmaniarl.algorithms._torch import (
    TorchLearnerBase,
    evaluated_actor_state,
    polyak_update,
    weighted_mean,
)
from trackmaniarl.algorithms.sac_config import DiscreteSACConfig, DiscreteSACOptions
from trackmaniarl.algorithms.sac_support import (
    EntropyConfig,
    EntropyRestoreTarget,
    SACBatch,
    alpha_value,
    discrete_batch,
    entropy_state,
    restore_entropy_state,
)
from trackmaniarl.core.contracts import ModelContract, PolicyMode
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.core.pytree import sanitize_finite, tree_collate, tree_to_device


class _DiscretePolicy:
    def __init__(self, actor: nn.Module, device: torch.device) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.device = device

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> Any:
        observation = tree_to_device(tree_collate([sanitize_finite(observation)]), self.device)
        with torch.no_grad():
            action, _ = self.actor(observation, mode=mode)
        values = action.detach().cpu().reshape(-1)
        if values.numel() != 1:
            raise ValueError("Discrete policy must produce exactly one action per observation")
        return int(values.item())

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.actor.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state)


@dataclass(frozen=True, slots=True)
class _DiscreteCriticStep:
    loss: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DiscreteActorStep:
    loss: torch.Tensor
    entropy: torch.Tensor
    action_count: int


@dataclass(frozen=True, slots=True)
class _DiscreteUpdate:
    critic: _DiscreteCriticStep
    actor: _DiscreteActorStep
    alpha_loss: torch.Tensor
    alpha: torch.Tensor


class StableDiscreteSoftActorCritic(TorchLearnerBase):
    """Experimental SD-SAC-inspired learner using target-policy entropy anchoring."""

    accepted_model_contracts = frozenset({ModelContract.DISCRETE_ACTOR_CRITIC})
    supports_sequence_training = False

    def __init__(
        self,
        model: nn.Module | None = None,
        **options: Unpack[DiscreteSACOptions],
    ) -> None:
        config = DiscreteSACConfig(**options)
        super().__init__(
            model,
            model_factory=config.model_factory,
            execution=config.execution,
            seed=config.seed,
        )
        config.validate()
        self._configure(config)

    def _configure(self, config: DiscreteSACConfig) -> None:
        self.learning_rate = config.learning_rate
        self.target_tau = config.target_tau
        self.initial_entropy_coefficient = config.entropy_coefficient
        self.entropy_mode = "learned" if config.learn_entropy_coefficient else "fixed"
        self.target_entropy = config.target_entropy
        self.q_clip_epsilon = config.q_clip_epsilon
        self.entropy_penalty_coefficient = config.entropy_penalty_coefficient

    def _setup_model(self) -> None:
        assert self.model is not None
        if not all(hasattr(self.model, name) for name in ("actor", "q1", "q2")):
            raise TypeError("SD-SAC model must expose actor, q1 and q2")
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self._setup_optimizers()
        config = EntropyConfig(
            self.initial_entropy_coefficient, self.learning_rate, self.entropy_mode
        )
        self.log_alpha, self.alpha_optimizer = entropy_state(config, self.device)

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
        prepared = discrete_batch(self._batch(batch))
        alpha = alpha_value(self.log_alpha, self.initial_entropy_coefficient, self.device)
        critic = self._critic_step(prepared, alpha)
        self._optimize(critic.loss, self.critic_optimizer)
        actor = self._actor_step(prepared, alpha)
        self._optimize(actor.loss, self.actor_optimizer)
        alpha_loss = self._entropy_loss(actor)
        polyak_update(self.model, self.target_model, self.target_tau)
        return self._update_result(prepared, _DiscreteUpdate(critic, actor, alpha_loss, alpha))

    def _critic_targets(self, batch: SACBatch, alpha: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_probabilities = self.model.actor.probabilities(batch.next_observations)
            next_log_probabilities = next_probabilities.clamp_min(1e-8).log()
            next_q = 0.5 * (
                self.target_model.q1(batch.next_observations)
                + self.target_model.q2(batch.next_observations)
            )
            next_value = (next_probabilities * (next_q - alpha * next_log_probabilities)).sum(1)
            return cast(torch.Tensor, batch.rewards + batch.discounts * next_value)

    def _critic_step(self, batch: SACBatch, alpha: torch.Tensor) -> _DiscreteCriticStep:
        targets = self._critic_targets(batch, alpha)
        indices = batch.actions[:, None]
        q1 = self.model.q1(batch.observations).gather(1, indices).squeeze(1)
        q2 = self.model.q2(batch.observations).gather(1, indices).squeeze(1)
        losses = self._critic_losses(batch, (q1, q2, targets))
        return _DiscreteCriticStep(weighted_mean(losses, batch.weights), q1, q2, targets)

    def _critic_losses(
        self, batch: SACBatch, values: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        q1, q2, targets = values
        indices = batch.actions[:, None]
        with torch.no_grad():
            target_q1 = self.target_model.q1(batch.observations).gather(1, indices).squeeze(1)
            target_q2 = self.target_model.q2(batch.observations).gather(1, indices).squeeze(1)
        clipped_q1 = target_q1 + (q1 - target_q1).clamp(-self.q_clip_epsilon, self.q_clip_epsilon)
        clipped_q2 = target_q2 + (q2 - target_q2).clamp(-self.q_clip_epsilon, self.q_clip_epsilon)
        losses = torch.maximum((q1 - targets).square(), (clipped_q1 - targets).square())
        return losses + torch.maximum((q2 - targets).square(), (clipped_q2 - targets).square())

    def _actor_step(self, batch: SACBatch, alpha: torch.Tensor) -> _DiscreteActorStep:
        probabilities = self.model.actor.probabilities(batch.observations)
        log_probabilities = probabilities.clamp_min(1e-8).log()
        q_values = (
            0.5 * (self.model.q1(batch.observations) + self.model.q2(batch.observations))
        ).detach()
        actor_loss = (probabilities * (alpha * log_probabilities - q_values)).sum(1).mean()
        entropy = -(probabilities * log_probabilities).sum(1)
        penalty = self._entropy_penalty(batch, entropy)
        loss = actor_loss + self.entropy_penalty_coefficient * penalty
        return _DiscreteActorStep(loss, entropy, int(probabilities.shape[-1]))

    def _entropy_penalty(self, batch: SACBatch, entropy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_probabilities = self.target_model.actor.probabilities(batch.observations)
            target_entropy = -(
                target_probabilities * target_probabilities.clamp_min(1e-8).log()
            ).sum(1)
        return cast(torch.Tensor, (entropy - target_entropy).square().mean())

    def _entropy_loss(self, actor: _DiscreteActorStep) -> torch.Tensor:
        alpha_loss = torch.zeros((), device=self.device)
        if self.log_alpha is None:
            return alpha_loss
        desired = self.target_entropy
        desired = desired if desired is not None else 0.98 * math.log(actor.action_count)
        alpha_loss = (self.log_alpha * (actor.entropy.detach() - desired)).mean()
        assert self.alpha_optimizer is not None
        self._optimize(alpha_loss, self.alpha_optimizer)
        return alpha_loss

    @staticmethod
    def _update_result(
        batch: SACBatch, update: _DiscreteUpdate
    ) -> tuple[Mapping[str, float], PriorityUpdate]:
        metrics = {
            "loss/actor": float(update.actor.loss.item()),
            "loss/critic": float(update.critic.loss.item()),
            "loss/entropy": float(update.alpha_loss.item()),
            "state/alpha": float(update.alpha.item()),
        }
        td_errors = (0.5 * (update.critic.q1 + update.critic.q2) - update.critic.targets).abs()
        return metrics, PriorityUpdate(
            batch.source.transition_ids, td_errors.detach().cpu().tolist()
        )

    def policy(self) -> _DiscretePolicy:
        assert self.model is not None
        return _DiscretePolicy(self.model.actor, self.device)

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
