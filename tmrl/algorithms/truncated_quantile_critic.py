"""Truncated Quantile Critic learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from tmrl.algorithms._torch import TorchLearnerBase, TorchPolicy, polyak_update, weighted_mean
from tmrl.core.data import PriorityUpdate, TrainingBatch


def quantile_huber_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample quantile Huber loss for ``(batch, quantiles)`` predictions."""

    delta = targets[:, None, :] - predictions[:, :, None]
    huber = torch.where(delta.abs() <= 1, 0.5 * delta.square(), delta.abs() - 0.5)
    count = predictions.shape[1]
    tau = (torch.arange(count, device=predictions.device) + 0.5) / count
    return (torch.abs(tau[None, :, None] - (delta.detach() < 0).float()) * huber).mean(dim=(1, 2))


class TruncatedQuantileCritic(TorchLearnerBase):
    """TQC with global target truncation and direct quantile-regression loss."""

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
        top_quantiles_to_drop: int = 2,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(model, model_factory=model_factory, device=device, seed=seed)
        self.target_tau = target_tau
        self.learning_rate = learning_rate
        self.entropy_coefficient = entropy_coefficient
        self.target_entropy = target_entropy
        self.learn_entropy_coefficient = learn_entropy_coefficient
        self.top_quantiles_to_drop = top_quantiles_to_drop

    def _setup_model(self) -> None:
        assert self.model is not None
        if not hasattr(self.model, "actor"):
            raise TypeError("TQC model must expose actor and quantile critics")
        critics = getattr(self.model, "critics", None)
        if critics is None:
            if not all(hasattr(self.model, name) for name in ("q1", "q2")):
                raise TypeError("TQC model must expose a critics ModuleList or q1 and q2")
            critics = nn.ModuleList([self.model.q1, self.model.q2])
        if len(critics) < 2:
            raise ValueError("TQC requires at least two quantile critics")
        self.critics = critics
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.learning_rate)
        self.log_alpha: torch.Tensor | None
        self.alpha_optimizer: torch.optim.Optimizer | None
        if self.learn_entropy_coefficient:
            self.log_alpha = (
                torch.tensor(self.entropy_coefficient, device=self.device).log().requires_grad_()
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
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
            else torch.tensor(self.entropy_coefficient, device=self.device)
        )
        target_critics = getattr(self.target_model, "critics", None)
        if target_critics is None:
            target_critics = (self.target_model.q1, self.target_model.q2)
        with torch.no_grad():
            next_actions, next_log_probabilities = self.model.actor(next_observations)
            target_quantiles = (
                torch.cat(
                    [critic(next_observations, next_actions) for critic in target_critics], dim=1
                )
                .sort(dim=1)
                .values
            )
            if self.top_quantiles_to_drop:
                if target_quantiles.shape[1] <= self.top_quantiles_to_drop:
                    raise ValueError("top_quantiles_to_drop removes every target quantile")
                target_quantiles = target_quantiles[:, : -self.top_quantiles_to_drop]
            targets = rewards[:, None] + discounts[:, None] * (
                target_quantiles - alpha * next_log_probabilities[:, None]
            )
        predictions = [critic(observations, actions) for critic in self.critics]
        losses = torch.stack(
            [quantile_huber_loss(prediction, targets) for prediction in predictions]
        ).sum(dim=0)
        critic_loss = weighted_mean(losses, weights)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for critic in self.critics:
            critic.requires_grad_(False)
        policy_actions, log_probabilities = self.model.actor(observations)
        policy_values = torch.stack(
            [critic(observations, policy_actions).mean(dim=1) for critic in self.critics]
        ).amin(dim=0)
        actor_loss = (alpha * log_probabilities - policy_values).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in self.critics:
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
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        polyak_update(self.model, self.target_model, self.target_tau)
        td_errors = (
            (torch.stack([item.mean(1) for item in predictions]).mean(0) - targets.mean(1))
            .detach()
            .abs()
        )
        return (
            {
                "loss/actor": float(actor_loss.item()),
                "loss/critic": float(critic_loss.item()),
                "loss/alpha": float(alpha_loss.item()),
                "state/alpha": float(alpha.item()),
            },
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
