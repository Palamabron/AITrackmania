"""Truncated Quantile Critic learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Unpack, cast

import torch
from torch import nn

from trackmaniarl.algorithms._torch import (
    TorchLearnerBase,
    TorchPolicy,
    evaluated_actor_state,
    polyak_update,
    weighted_mean,
)
from trackmaniarl.algorithms.sac_config import TQCConfig, TQCOptions
from trackmaniarl.algorithms.sac_support import (
    EntropyConfig,
    EntropyRestoreTarget,
    SACBatch,
    alpha_value,
    continuous_batch,
    entropy_state,
    freeze_modules,
    restore_entropy_state,
    unfreeze_modules,
)
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch


def quantile_huber_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample quantile Huber loss for ``(batch, quantiles)`` predictions."""

    delta = targets[:, None, :] - predictions[:, :, None]
    huber = torch.where(delta.abs() <= 1, 0.5 * delta.square(), delta.abs() - 0.5)
    count = predictions.shape[1]
    tau = (torch.arange(count, device=predictions.device) + 0.5) / count
    return (torch.abs(tau[None, :, None] - (delta.detach() < 0).float()) * huber).mean(dim=(1, 2))


def _truncate_quantile_mixture(
    quantiles: torch.Tensor,
    *,
    critic_count: int,
    top_quantiles_to_drop_per_critic: int,
) -> torch.Tensor:
    drop_count = critic_count * top_quantiles_to_drop_per_critic
    if not drop_count:
        return quantiles
    if quantiles.shape[1] <= drop_count:
        raise ValueError("top_quantiles_to_drop_per_critic removes every target quantile")
    return quantiles[:, :-drop_count]


@dataclass(frozen=True, slots=True)
class _TQCCriticStep:
    loss: torch.Tensor
    predictions: list[torch.Tensor]
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class _TQCUpdate:
    critic: _TQCCriticStep
    actor_loss: torch.Tensor
    alpha_loss: torch.Tensor
    alpha: torch.Tensor


class TruncatedQuantileCritic(TorchLearnerBase):
    """TQC with global target truncation and direct quantile-regression loss."""

    accepted_model_contracts = frozenset({ModelContract.CONTINUOUS_QUANTILE_ACTOR_CRITIC})
    supports_sequence_training = False

    def __init__(
        self,
        model: nn.Module | None = None,
        **options: Unpack[TQCOptions],
    ) -> None:
        config = TQCConfig(**options)
        super().__init__(
            model,
            model_factory=config.model_factory,
            execution=config.execution,
            seed=config.seed,
        )
        config.validate()
        self._configure(config)

    def _configure(self, config: TQCConfig) -> None:
        self.target_tau = config.target_tau
        self.learning_rate = config.learning_rate
        self.entropy_coefficient = config.entropy_coefficient
        self.target_entropy = config.target_entropy
        self.entropy_mode = "learned" if config.learn_entropy_coefficient else "fixed"
        self.top_quantiles_to_drop_per_critic = config.top_quantiles_to_drop_per_critic

    def _setup_model(self) -> None:
        assert self.model is not None
        self.critics = self._model_critics()
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self._setup_optimizers()
        config = EntropyConfig(self.entropy_coefficient, self.learning_rate, self.entropy_mode)
        self.log_alpha, self.alpha_optimizer = entropy_state(config, self.device)

    def _model_critics(self) -> nn.ModuleList:
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
        return critics

    def _setup_optimizers(self) -> None:
        assert self.model is not None
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.learning_rate)

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        with self.autocast():
            return self._update(batch)

    def _update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert self.model is not None
        prepared = continuous_batch(self._batch(batch))
        alpha = alpha_value(self.log_alpha, self.entropy_coefficient, self.device)
        critic = self._critic_step(prepared, alpha)
        self._optimize(critic.loss, self.critic_optimizer)
        actor_loss, log_probabilities = self._actor_step(prepared, alpha)
        alpha_loss = self._entropy_loss(prepared.actions, log_probabilities)
        polyak_update(self.model, self.target_model, self.target_tau)
        update = _TQCUpdate(critic, actor_loss, alpha_loss, alpha)
        return self._update_result(prepared, update)

    def _target_critics(self) -> Any:
        target_critics = getattr(self.target_model, "critics", None)
        if target_critics is None:
            target_critics = (self.target_model.q1, self.target_model.q2)
        return target_critics

    def _target_quantiles(self, batch: SACBatch, alpha: torch.Tensor) -> torch.Tensor:
        target_critics = self._target_critics()
        with torch.no_grad():
            next_actions, next_log_probabilities = self.model.actor(batch.next_observations)
            target_quantiles = self._sorted_target_quantiles(
                batch.next_observations, next_actions, target_critics
            )
            target_quantiles = _truncate_quantile_mixture(
                target_quantiles,
                critic_count=len(target_critics),
                top_quantiles_to_drop_per_critic=self.top_quantiles_to_drop_per_critic,
            )
            result = batch.rewards[:, None] + batch.discounts[:, None] * (
                target_quantiles - alpha * next_log_probabilities[:, None]
            )
            return cast(torch.Tensor, result)

    @staticmethod
    def _sorted_target_quantiles(
        observations: torch.Tensor, actions: torch.Tensor, critics: Any
    ) -> torch.Tensor:
        values = torch.cat([critic(observations, actions) for critic in critics], dim=1)
        return values.sort(dim=1).values

    def _critic_step(self, batch: SACBatch, alpha: torch.Tensor) -> _TQCCriticStep:
        targets = self._target_quantiles(batch, alpha)
        predictions = [critic(batch.observations, batch.actions) for critic in self.critics]
        losses = torch.stack(
            [quantile_huber_loss(prediction, targets) for prediction in predictions]
        ).sum(dim=0)
        return _TQCCriticStep(weighted_mean(losses, batch.weights), predictions, targets)

    def _actor_step(
        self, batch: SACBatch, alpha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freeze_modules(self.critics)
        policy_actions, log_probabilities = self.model.actor(batch.observations)
        policy_values = torch.cat(
            [critic(batch.observations, policy_actions) for critic in self.critics], dim=1
        ).mean(dim=1)
        actor_loss = (alpha * log_probabilities - policy_values).mean()
        self._optimize(actor_loss, self.actor_optimizer)
        unfreeze_modules(self.critics)
        return actor_loss, log_probabilities

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
        batch: SACBatch, update: _TQCUpdate
    ) -> tuple[Mapping[str, float], PriorityUpdate]:
        metrics = {
            "loss/actor": float(update.actor_loss.item()),
            "loss/critic": float(update.critic.loss.item()),
            "loss/alpha": float(update.alpha_loss.item()),
            "state/alpha": float(update.alpha.item()),
        }
        predictions = torch.stack([item.mean(1) for item in update.critic.predictions]).mean(0)
        td_errors = (predictions - update.critic.targets.mean(1)).detach().abs()
        return metrics, PriorityUpdate(batch.source.transition_ids, td_errors.cpu().tolist())

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
