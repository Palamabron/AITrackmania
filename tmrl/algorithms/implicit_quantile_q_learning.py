"""Double-DQN Implicit Quantile Q-learning learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any

import torch
from torch import nn

from tmrl.algorithms._torch import TorchLearnerBase
from tmrl.algorithms.execution import TorchExecutionConfig
from tmrl.core.data import PriorityUpdate, TrainingBatch
from tmrl.core.pytree import sanitize_finite, tree_map, tree_to_device


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("IQN observations must contain a tensor leaf")


def _unsqueeze_observation(value: Any) -> Any:
    return tree_map(
        lambda leaf: leaf.unsqueeze(0) if isinstance(leaf, torch.Tensor) else leaf, value
    )


def implicit_quantile_huber_loss(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Per-sample IQN quantile-Huber loss in float32 value space."""

    delta = targets[:, None, :] - predictions[:, :, None]
    huber = torch.where(delta.abs() <= 1, 0.5 * delta.square(), delta.abs() - 0.5)
    weights = torch.abs(quantiles[:, :, None] - (delta.detach() < 0).float())
    return (weights * huber).mean(dim=(1, 2))


class _IQNPolicy:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        quantile_count: int,
        exploration_epsilon: float,
    ) -> None:
        self.model: Any = deepcopy(model).to(device).eval()
        self.device = device
        self.quantile_count = quantile_count
        self.exploration_epsilon = exploration_epsilon

    def act(self, observation: Any, *, deterministic: bool = False) -> Any:
        observation = tree_to_device(sanitize_finite(observation), self.device)
        is_single_observation = _first_tensor(observation).ndim in {1, 2}
        if is_single_observation:
            observation = _unsqueeze_observation(observation)
        with torch.no_grad():
            q_values = self.model.q_values(observation, self.quantile_count)
            action = q_values.argmax(dim=-1)
            if not deterministic and self.exploration_epsilon:
                exploratory = (
                    torch.rand(action.shape, device=self.device) < self.exploration_epsilon
                )
                random_actions = torch.randint(
                    q_values.shape[-1], action.shape, device=self.device, dtype=action.dtype
                )
                action = torch.where(exploratory, random_actions, action)
        if is_single_observation:
            return int(action.item())
        return action.cpu().numpy()

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.model.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state)

    def set_exploration_epsilon(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("exploration epsilon must be between 0 and 1")
        self.exploration_epsilon = epsilon


class ImplicitQuantileQLearning(TorchLearnerBase):
    """Distributional Double-DQN with IQN fractions and hard/soft target updates."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_factory: Any | None = None,
        learning_rate: float = 1e-4,
        train_quantile_count: int = 64,
        target_quantile_count: int = 64,
        evaluation_quantile_count: int = 32,
        target_update_interval: int = 1_000,
        target_tau: float = 0.0,
        gradient_clip_norm: float = 10.0,
        exploration_epsilon: float = 0.1,
        exploration_epsilon_final: float | None = None,
        exploration_epsilon_decay_updates: int = 0,
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(model, model_factory=model_factory, execution=execution, seed=seed)
        self.learning_rate = learning_rate
        self.train_quantile_count = train_quantile_count
        self.target_quantile_count = target_quantile_count
        self.evaluation_quantile_count = evaluation_quantile_count
        self.target_update_interval = target_update_interval
        self.target_tau = target_tau
        self.gradient_clip_norm = gradient_clip_norm
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be between 0 and 1")
        self.exploration_epsilon = exploration_epsilon
        final = (
            exploration_epsilon if exploration_epsilon_final is None else exploration_epsilon_final
        )
        if not 0.0 <= final <= 1.0 or exploration_epsilon_decay_updates < 0:
            raise ValueError("IQN epsilon schedule parameters are invalid")
        self.exploration_epsilon_final = final
        self.exploration_epsilon_decay_updates = exploration_epsilon_decay_updates
        self.update_count = 0
        self._train_model: Any = None
        self._compile_pending = False

    def _setup_model(self) -> None:
        assert self.model is not None
        if not hasattr(self.model, "q_values"):
            raise TypeError("IQN model must expose q_values(observation, quantile_count)")
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._train_model = self.model
        assert self.resolved_execution is not None
        if self.resolved_execution.compile_requested:
            try:
                self._train_model = torch.compile(
                    self.model,
                    mode=self.resolved_execution.compile_mode,
                )
                self._compile_pending = True
            except (RuntimeError, TypeError) as exc:
                self.resolved_execution = self.resolved_execution.with_compile_result(
                    effective=False,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
                self._record_execution_result()

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        try:
            return self._update(batch)
        except (RuntimeError, TypeError) as exc:
            if not self._compile_pending:
                raise
            self._train_model = self.model
            self._compile_pending = False
            assert self.resolved_execution is not None
            self.resolved_execution = self.resolved_execution.with_compile_result(
                effective=False,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
            self._record_execution_result()
            self.optimizer.zero_grad(set_to_none=True)
            return self._update(batch)

    def _update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert self.model is not None
        started = perf_counter()
        batch = self._batch(batch)
        transfer_finished = perf_counter()
        observations = self._observation(batch.observations, "observations")
        actions = self._tensor(batch.actions, "actions").long().reshape(-1)
        rewards = self._tensor(batch.rewards, "rewards").float().reshape(-1)
        next_observations = self._observation(batch.next_observations, "next_observations")
        discounts = (
            self._tensor(batch.bootstrap_discounts, "bootstrap_discounts").float().reshape(-1)
        )
        batch_size = _first_tensor(observations).shape[0]
        quantiles = torch.rand(batch_size, self.train_quantile_count, device=self.device)
        with self.autocast():
            predictions = self._train_model(observations, quantiles)
            selected = predictions.gather(
                2, actions[:, None, None].expand(-1, self.train_quantile_count, 1)
            ).squeeze(-1)
            with torch.no_grad():
                next_actions = self._train_model.q_values(
                    next_observations, self.evaluation_quantile_count
                ).argmax(dim=-1)
                target_quantiles = torch.rand(
                    batch_size, self.target_quantile_count, device=self.device
                )
                target_values = (
                    self.target_model(next_observations, target_quantiles)
                    .gather(
                        2,
                        next_actions[:, None, None].expand(-1, self.target_quantile_count, 1),
                    )
                    .squeeze(-1)
                )
                targets = rewards[:, None] + discounts[:, None] * target_values
        forward_finished = perf_counter()
        losses = implicit_quantile_huber_loss(selected.float(), targets.float(), quantiles)
        weights = (
            batch.importance_weights if isinstance(batch.importance_weights, torch.Tensor) else None
        )
        loss = (
            (losses * weights).sum() / weights.sum().clamp_min(1e-8)
            if weights is not None
            else losses.mean()
        )
        self.optimizer.zero_grad(set_to_none=True)
        assert self.scaler is not None
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        backward_finished = perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        clipping_finished = perf_counter()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer_finished = perf_counter()
        self.update_count += 1
        if self._compile_pending:
            self._compile_pending = False
            assert self.resolved_execution is not None
            self.resolved_execution = self.resolved_execution.with_compile_result(effective=True)
            self._record_execution_result()
        if self.target_tau > 0:
            with torch.no_grad():
                for target, source in zip(
                    self.target_model.parameters(), self.model.parameters(), strict=True
                ):
                    target.lerp_(source, self.target_tau)
        elif self.update_count % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        td_errors = (selected.mean(1) - targets.mean(1)).detach().abs()
        return (
            {
                "loss/iqn": float(loss.item()),
                "debug/gradient_norm": float(gradient_norm),
                "debug/td_abs_mean": float(td_errors.mean().item()),
                "timing/host_to_device_s": transfer_finished - started,
                "timing/forward_s": forward_finished - transfer_finished,
                "timing/backward_s": backward_finished - forward_finished,
                "timing/gradient_clip_s": clipping_finished - backward_finished,
                "timing/optimizer_s": optimizer_finished - clipping_finished,
            },
            PriorityUpdate(batch.transition_ids, td_errors.cpu().tolist()),
        )

    def policy(self) -> _IQNPolicy:
        assert self.model is not None
        return _IQNPolicy(
            self.model,
            self.device,
            self.evaluation_quantile_count,
            self._current_epsilon(),
        )

    def _observation(self, value: Any, name: str) -> Any:
        value = tree_to_device(sanitize_finite(value), self.device)
        if _first_tensor(value).ndim < 1:
            raise ValueError(f"{name} tensors require a batch axis")
        return value

    def _current_epsilon(self) -> float:
        if self.exploration_epsilon_decay_updates == 0:
            return self.exploration_epsilon_final
        fraction = min(1.0, self.update_count / self.exploration_epsilon_decay_updates)
        return self.exploration_epsilon + fraction * (
            self.exploration_epsilon_final - self.exploration_epsilon
        )

    def state_dict(self) -> Mapping[str, Any]:
        assert self.model is not None
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state["update_count"])
        self._restore_rng(state.get("rng", {}))
