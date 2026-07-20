"""Discrete SAC learner with conservative double-Q targets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from tmrl.algorithms._torch import TorchLearnerBase, weighted_mean
from tmrl.core.data import PriorityUpdate, TrainingBatch
from tmrl.core.pytree import sanitize_finite, tree_to_device


class _DiscretePolicy:
    def __init__(self, actor: nn.Module, device: torch.device) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.device = device

    def act(self, observation: Any, *, deterministic: bool = False) -> Any:
        observation = tree_to_device(sanitize_finite(observation), self.device)
        if not isinstance(observation, torch.Tensor):
            raise TypeError(
                "Discrete policy requires tensor observations from the feature pipeline"
            )
        with torch.no_grad():
            action, _ = self.actor(observation, deterministic=deterministic)
        return action.cpu().numpy()

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.actor.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state)


class StableDiscreteSoftActorCritic(TorchLearnerBase):
    """SD-SAC with double-average Q learning, Q-clip and entropy penalty."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_factory: Any | None = None,
        learning_rate: float = 3e-4,
        target_tau: float = 0.005,
        entropy_coefficient: float = 0.2,
        learn_entropy_coefficient: bool = True,
        target_entropy: float | None = None,
        q_clip_epsilon: float = 0.5,
        entropy_penalty_coefficient: float = 0.5,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(model, model_factory=model_factory, device=device, seed=seed)
        self.learning_rate = learning_rate
        self.target_tau = target_tau
        if entropy_coefficient <= 0:
            raise ValueError("entropy_coefficient must be positive")
        self.initial_entropy_coefficient = entropy_coefficient
        self.learn_entropy_coefficient = learn_entropy_coefficient
        self.target_entropy = target_entropy
        if q_clip_epsilon < 0 or entropy_penalty_coefficient < 0:
            raise ValueError(
                "SD-SAC clipping and entropy penalty coefficients must be non-negative"
            )
        self.q_clip_epsilon = q_clip_epsilon
        self.entropy_penalty_coefficient = entropy_penalty_coefficient

    def _setup_model(self) -> None:
        assert self.model is not None
        if not all(hasattr(self.model, name) for name in ("actor", "q1", "q2")):
            raise TypeError("SD-SAC model must expose actor, q1 and q2")
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
        assert self.model is not None
        batch = self._batch(batch)
        observations = self._tensor(batch.observations, "observations")
        actions = self._tensor(batch.actions, "actions").long().reshape(-1)
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
            next_probabilities = self.model.actor.probabilities(next_observations)
            next_log_probabilities = next_probabilities.clamp_min(1e-8).log()
            next_q = 0.5 * (
                self.target_model.q1(next_observations) + self.target_model.q2(next_observations)
            )
            next_value = (next_probabilities * (next_q - alpha * next_log_probabilities)).sum(1)
            targets = rewards + discounts * next_value
        q1 = self.model.q1(observations).gather(1, actions[:, None]).squeeze(1)
        q2 = self.model.q2(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target_q1 = self.target_model.q1(observations).gather(1, actions[:, None]).squeeze(1)
            target_q2 = self.target_model.q2(observations).gather(1, actions[:, None]).squeeze(1)
        clipped_q1 = target_q1 + (q1 - target_q1).clamp(-self.q_clip_epsilon, self.q_clip_epsilon)
        clipped_q2 = target_q2 + (q2 - target_q2).clamp(-self.q_clip_epsilon, self.q_clip_epsilon)
        critic_losses = torch.maximum((q1 - targets).square(), (clipped_q1 - targets).square())
        critic_losses = critic_losses + torch.maximum(
            (q2 - targets).square(), (clipped_q2 - targets).square()
        )
        critic_loss = weighted_mean(critic_losses, weights)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        probabilities = self.model.actor.probabilities(observations)
        log_probabilities = probabilities.clamp_min(1e-8).log()
        q_values = (0.5 * (self.model.q1(observations) + self.model.q2(observations))).detach()
        actor_loss = (probabilities * (alpha * log_probabilities - q_values)).sum(1).mean()
        entropy = -(probabilities * log_probabilities).sum(1)
        with torch.no_grad():
            target_probabilities = self.target_model.actor.probabilities(observations)
            target_entropy = -(
                target_probabilities * target_probabilities.clamp_min(1e-8).log()
            ).sum(1)
        entropy_penalty = (entropy - target_entropy).square().mean()
        actor_loss = actor_loss + self.entropy_penalty_coefficient * entropy_penalty
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = torch.zeros((), device=self.device)
        if self.log_alpha is not None:
            desired_entropy = (
                self.target_entropy
                if self.target_entropy is not None
                else 0.98 * math.log(probabilities.shape[-1])
            )
            alpha_loss = (self.log_alpha * (entropy.detach() - desired_entropy)).mean()
            assert self.alpha_optimizer is not None
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            for target, source in zip(
                self.target_model.parameters(), self.model.parameters(), strict=True
            ):
                target.lerp_(source, self.target_tau)
        td_errors = (0.5 * (q1 + q2) - targets).detach().abs()
        return (
            {
                "loss/actor": float(actor_loss.item()),
                "loss/critic": float(critic_loss.item()),
                "loss/entropy": float(alpha_loss.item()),
                "state/alpha": float(alpha.item()),
            },
            PriorityUpdate(batch.transition_ids, td_errors.cpu().tolist()),
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
            "rng": self._rng_state(),
        }

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
        self._restore_rng(state.get("rng", {}))
