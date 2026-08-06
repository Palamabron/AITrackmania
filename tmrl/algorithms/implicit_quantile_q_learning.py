"""Double-DQN Implicit Quantile Q-learning learner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from tmrl.algorithms._torch import TorchLearnerBase, backward, polyak_update, weighted_mean
from tmrl.algorithms.execution import TorchExecutionConfig
from tmrl.core.builtins import TorchCheckpointCodec
from tmrl.core.data import PriorityUpdate, TrainingBatch
from tmrl.core.pytree import sanitize_finite, tree_map, tree_to_device

_SEQUENCE_PRIORITY_MAX_WEIGHT = 0.9
_VALUE_RESCALING_EPSILON = 1e-3
QuantileDistortion = Literal["neutral", "upper_cvar"]


def rescale_value(value: torch.Tensor) -> torch.Tensor:
    """R2D2 invertible value rescaling ``h(x) = sign(x)(sqrt(|x|+1)-1) + eps*x``."""

    return value.sign() * ((value.abs() + 1.0).sqrt() - 1.0) + _VALUE_RESCALING_EPSILON * value


def inverse_rescale_value(value: torch.Tensor) -> torch.Tensor:
    epsilon = _VALUE_RESCALING_EPSILON
    inner = (1.0 + 4.0 * epsilon * (value.abs() + 1.0 + epsilon)).sqrt() - 1.0
    return value.sign() * ((inner / (2.0 * epsilon)).square() - 1.0)


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


@dataclass(slots=True)
class _LossComputation:
    per_sample_losses: torch.Tensor
    priorities: torch.Tensor
    selected: torch.Tensor
    targets: torch.Tensor
    step_actions: torch.Tensor
    action_count: int
    rewards: torch.Tensor
    margin_loss: torch.Tensor | None
    policy_anchor_loss: torch.Tensor | None
    trained_positions: int
    expected_q: torch.Tensor
    q_valid: torch.Tensor


class _IQNPolicy:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        quantile_count: int,
        exploration_epsilon: float,
        policy_action_ids: tuple[int, ...] | None,
        online_quantile_distortion: QuantileDistortion,
        evaluation_quantile_distortion: QuantileDistortion,
        upper_cvar_alpha: float,
    ) -> None:
        self.model: Any = deepcopy(model).to(device).eval()
        self.device = device
        self.quantile_count = quantile_count
        self.exploration_epsilon = exploration_epsilon
        self.policy_action_ids = None if policy_action_ids is None else tuple(policy_action_ids)
        self.online_quantile_distortion = online_quantile_distortion
        self.evaluation_quantile_distortion = evaluation_quantile_distortion
        self.upper_cvar_alpha = upper_cvar_alpha
        self.last_q_margin: float | None = None
        self.last_q_max: float | None = None

    def act(self, observation: Any, *, deterministic: bool = False) -> Any:
        prepare = getattr(self.model, "prepare_policy_observation", None)
        if callable(prepare):
            observation = prepare(observation)
        observation = tree_to_device(sanitize_finite(observation), self.device)
        detector = getattr(self.model, "observation_is_single", None)
        is_single_observation = (
            bool(detector(observation))
            if callable(detector)
            else _first_tensor(observation).ndim in {1, 2}
        )
        if is_single_observation:
            observation = _unsqueeze_observation(observation)
        with torch.no_grad():
            q_values = self._action_q_values(observation, deterministic=deterministic)
            policy_q_values = self._policy_q_values(q_values)
            action = policy_q_values.argmax(dim=-1)
            self._record_action_gap(policy_q_values, single=is_single_observation)
            if not deterministic and self.exploration_epsilon:
                exploratory = (
                    torch.rand(action.shape, device=self.device) < self.exploration_epsilon
                )
                random_actions = self._exploration_actions(q_values, action)
                action = torch.where(exploratory, random_actions, action)
        if is_single_observation:
            return int(action.item())
        return action.cpu().numpy()

    def _action_q_values(self, observation: Any, *, deterministic: bool) -> torch.Tensor:
        distortion = (
            self.evaluation_quantile_distortion
            if deterministic
            else self.online_quantile_distortion
        )
        if distortion == "neutral":
            return cast(torch.Tensor, self.model.q_values(observation, self.quantile_count))
        batch_size = _first_tensor(observation).shape[0]
        offsets = (
            torch.arange(self.quantile_count, device=self.device, dtype=torch.float32) + 0.5
        ) / self.quantile_count
        quantiles = 1.0 - self.upper_cvar_alpha + self.upper_cvar_alpha * offsets
        values = self.model(observation, quantiles.expand(batch_size, -1))
        return cast(torch.Tensor, values).mean(dim=1)

    def _policy_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.policy_action_ids is None:
            return q_values
        mask = torch.zeros(q_values.shape[-1], dtype=torch.bool, device=self.device)
        mask[list(self.policy_action_ids)] = True
        return q_values.masked_fill(~mask, -torch.inf)

    def _record_action_gap(self, q_values: torch.Tensor, *, single: bool) -> None:
        if not single or q_values.shape[-1] < 2:
            self.last_q_margin = None
            self.last_q_max = None
            return
        best, runner_up = q_values[0].topk(2).values.tolist()
        self.last_q_max = float(best)
        self.last_q_margin = float(best - runner_up)

    def _exploration_actions(self, q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        weights = getattr(self.model, "exploration_action_weights", None)
        if isinstance(weights, torch.Tensor) and weights.shape == (q_values.shape[-1],):
            sampling_weights = weights.to(device=self.device, dtype=torch.float32)
            if self.policy_action_ids is not None:
                allowed = torch.zeros_like(sampling_weights, dtype=torch.bool)
                allowed[list(self.policy_action_ids)] = True
                sampling_weights = sampling_weights * allowed
            global_actions = torch.multinomial(
                sampling_weights,
                actions.numel(),
                replacement=True,
            ).reshape(actions.shape)
            if self.policy_action_ids is not None:
                return global_actions.to(actions.dtype)
            modes_per_steering = 6
            steering_bins = q_values.shape[-1] // modes_per_steering
            if steering_bins * modes_per_steering == q_values.shape[-1]:
                steering = actions // modes_per_steering
                mode = actions % modes_per_steering
                delta = torch.randint(
                    -1,
                    2,
                    actions.shape,
                    device=self.device,
                    dtype=actions.dtype,
                )
                neighboring = (steering + delta).clamp(
                    0, steering_bins - 1
                ) * modes_per_steering + mode
                change_mode = torch.rand(actions.shape, device=self.device) < 0.15
                return torch.where(change_mode, global_actions, neighboring).to(actions.dtype)
            return global_actions.to(actions.dtype)
        if self.policy_action_ids is None:
            return torch.randint(
                q_values.shape[-1], actions.shape, device=self.device, dtype=actions.dtype
            )
        choices = torch.as_tensor(self.policy_action_ids, device=self.device, dtype=actions.dtype)
        indices = torch.randint(len(choices), actions.shape, device=self.device)
        return choices[indices]

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.model.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state)

    def set_exploration_epsilon(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("exploration epsilon must be between 0 and 1")
        self.exploration_epsilon = epsilon

    def reset_episode(self) -> None:
        reset = getattr(self.model, "reset_policy_state", None)
        if callable(reset):
            reset()


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
        target_tau: float = 0.005,
        gradient_clip_norm: float = 10.0,
        exploration_epsilon: float = 0.1,
        exploration_epsilon_final: float | None = None,
        exploration_epsilon_decay_updates: int = 0,
        value_rescaling: bool = False,
        demonstration_margin: float = 0.8,
        demonstration_margin_weight: float = 0.0,
        policy_anchor_weight: float = 0.0,
        policy_action_ids: tuple[int, ...] | None = None,
        online_quantile_distortion: QuantileDistortion = "neutral",
        evaluation_quantile_distortion: QuantileDistortion = "neutral",
        upper_cvar_alpha: float = 0.25,
        policy_anchor_checkpoint: str | Path | None = None,
        model_initialization_checkpoint: str | Path | None = None,
        base_dir: str | Path = ".",
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(model, model_factory=model_factory, execution=execution, seed=seed)
        self.learning_rate = learning_rate
        self.train_quantile_count = train_quantile_count
        self.target_quantile_count = target_quantile_count
        self.evaluation_quantile_count = evaluation_quantile_count
        self.target_update_interval = target_update_interval
        if not 0.0 <= target_tau <= 1.0:
            raise ValueError("target_tau must be between zero and one")
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
        if (
            demonstration_margin < 0.0
            or demonstration_margin_weight < 0.0
            or policy_anchor_weight < 0.0
        ):
            raise ValueError("IQN auxiliary loss parameters must be non-negative")
        if policy_action_ids is not None and (
            not policy_action_ids
            or len(set(policy_action_ids)) != len(policy_action_ids)
            or any(action < 0 for action in policy_action_ids)
        ):
            raise ValueError("policy_action_ids must be non-empty, unique, non-negative IDs")
        if online_quantile_distortion not in {"neutral", "upper_cvar"}:
            raise ValueError("online_quantile_distortion must be 'neutral' or 'upper_cvar'")
        if evaluation_quantile_distortion not in {"neutral", "upper_cvar"}:
            raise ValueError("evaluation_quantile_distortion must be 'neutral' or 'upper_cvar'")
        if not 0.0 < upper_cvar_alpha <= 1.0:
            raise ValueError("upper_cvar_alpha must be in (0, 1]")
        if policy_anchor_checkpoint is not None and not policy_anchor_weight:
            raise ValueError("policy_anchor_checkpoint requires policy_anchor_weight")
        self.value_rescaling = value_rescaling
        self.demonstration_margin = demonstration_margin
        self.demonstration_margin_weight = demonstration_margin_weight
        self.policy_anchor_weight = policy_anchor_weight
        self.policy_action_ids = None if policy_action_ids is None else tuple(policy_action_ids)
        self.online_quantile_distortion = online_quantile_distortion
        self.evaluation_quantile_distortion = evaluation_quantile_distortion
        self.upper_cvar_alpha = upper_cvar_alpha
        anchor_path = (
            Path(policy_anchor_checkpoint) if policy_anchor_checkpoint is not None else None
        )
        self.policy_anchor_checkpoint = (
            None if anchor_path is None else (Path(base_dir) / anchor_path).resolve()
        )
        initialization_path = (
            Path(model_initialization_checkpoint)
            if model_initialization_checkpoint is not None
            else None
        )
        self.model_initialization_checkpoint = (
            None
            if initialization_path is None
            else (Path(base_dir) / initialization_path).resolve()
        )
        self.initialized_exact_tensors = 0
        self.initialized_expanded_tensors = 0
        self.update_count = 0
        self._train_model: Any = None
        self.policy_anchor_model: Any = None
        self._compile_pending = False

    def _setup_model(self) -> None:
        assert self.model is not None
        if not hasattr(self.model, "q_values"):
            raise TypeError("IQN model must expose q_values(observation, quantile_count)")
        action_count = getattr(self.model, "action_count", None)
        if self.policy_action_ids is not None and (
            not isinstance(action_count, int)
            or any(action >= action_count for action in self.policy_action_ids)
        ):
            raise ValueError("policy_action_ids must be valid model action indices")
        self._load_model_initialization()
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        if self.policy_anchor_weight:
            self.policy_anchor_model = deepcopy(self.model).to(self.device).eval()
            for parameter in self.policy_anchor_model.parameters():
                parameter.requires_grad_(False)
            self._load_configured_policy_anchor()
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
        host_to_device_s = float(
            batch.metadata.get("_tmrl_host_to_device_s", transfer_finished - started)
        )
        if self._is_sequence_batch(batch):
            computation = self._sequence_losses(batch)
        else:
            computation = self._single_step_losses(batch)
        forward_finished = perf_counter()
        weights = (
            batch.importance_weights.float().reshape(-1)
            if isinstance(batch.importance_weights, torch.Tensor)
            else None
        )
        iqn_loss = weighted_mean(computation.per_sample_losses, weights)
        loss = iqn_loss
        if computation.margin_loss is not None:
            loss = loss + self.demonstration_margin_weight * computation.margin_loss
        if computation.policy_anchor_loss is not None:
            loss = loss + self.policy_anchor_weight * computation.policy_anchor_loss
        self.optimizer.zero_grad(set_to_none=True)
        assert self.scaler is not None
        backward(self.scaler.scale(loss))
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
        target_synced = False
        if self.target_tau > 0:
            polyak_update(self.model, self.target_model, self.target_tau)
            target_synced = True
        elif self.update_count % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            target_synced = True
        metrics, priorities = self._transfer_metrics(
            batch, computation, iqn_loss, loss, gradient_norm, weights
        )
        metrics.update(
            {
                "debug/target_synced_fraction": float(target_synced),
                "debug/target_update_hard": float(self.target_tau == 0.0),
                "debug/target_update_interval": float(self.target_update_interval),
                "debug/trained_positions": float(computation.trained_positions),
                "timing/host_to_device_s": host_to_device_s,
                "timing/forward_s": forward_finished - transfer_finished,
                "timing/backward_s": backward_finished - forward_finished,
                "timing/gradient_clip_s": clipping_finished - backward_finished,
                "timing/optimizer_s": optimizer_finished - clipping_finished,
            }
        )
        metrics.update(
            {
                key: float(value)
                for key, value in batch.metadata.items()
                if key.startswith("replay/") and isinstance(value, (float, int))
            }
        )
        return metrics, PriorityUpdate(self._priority_transition_ids(batch), priorities)

    def _is_sequence_batch(self, batch: TrainingBatch) -> bool:
        rewards = self._tensor(batch.rewards, "rewards")
        if rewards.ndim != 2:
            return False
        supports = getattr(self.model, "supports_sequence_training", None)
        return (
            callable(supports)
            and bool(supports())
            and isinstance(batch.masks, torch.Tensor)
            and "gamma" in batch.metadata
            and "n_step" in batch.metadata
        )

    def _single_step_losses(self, batch: TrainingBatch) -> _LossComputation:
        observations = self._observation(batch.observations, "observations")
        actions = self._sequence_target(self._tensor(batch.actions, "actions")).long()
        rewards = self._sequence_target(self._tensor(batch.rewards, "rewards")).float()
        next_observations = self._observation(batch.next_observations, "next_observations")
        discounts = self._sequence_target(
            self._tensor(batch.bootstrap_discounts, "bootstrap_discounts")
        ).float()
        batch_size = _first_tensor(observations).shape[0]
        quantiles = torch.rand(batch_size, self.train_quantile_count, device=self.device)
        with self.autocast():
            predictions = self._train_model(observations, quantiles)
            selected = predictions.gather(
                2, actions[:, None, None].expand(-1, self.train_quantile_count, 1)
            ).squeeze(-1)
            with torch.no_grad():
                next_q_values = self._train_model.q_values(
                    next_observations, self.evaluation_quantile_count
                )
                next_actions = self._masked_argmax(next_q_values)
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
                targets = self._bootstrap_targets(
                    rewards[:, None], discounts[:, None], target_values
                )
        selected_fp32 = selected.float()
        targets_fp32 = targets.float()
        losses = implicit_quantile_huber_loss(selected_fp32, targets_fp32, quantiles)
        td_errors = (selected_fp32.mean(1) - targets_fp32.mean(1)).detach().abs()
        margin_loss = self._margin_loss(
            batch,
            predictions.float().mean(dim=1),
            actions,
            valid=None,
        )
        policy_anchor_loss = self._policy_anchor_loss(
            predictions.float().mean(dim=1),
            self._single_step_anchor_q_values(observations),
            valid=None,
        )
        return _LossComputation(
            per_sample_losses=losses,
            priorities=td_errors,
            selected=selected_fp32,
            targets=targets_fp32,
            step_actions=actions,
            action_count=int(predictions.shape[-1]),
            rewards=rewards,
            margin_loss=margin_loss,
            policy_anchor_loss=policy_anchor_loss,
            trained_positions=1,
            expected_q=predictions.float().mean(dim=1),
            q_valid=torch.ones(batch_size, dtype=torch.bool, device=self.device),
        )

    def _sequence_losses(self, batch: TrainingBatch) -> _LossComputation:
        observations = self._observation(batch.observations, "observations")
        next_observations = self._observation(batch.next_observations, "next_observations")
        actions = self._tensor(batch.actions, "actions").long()
        rewards = self._tensor(batch.rewards, "rewards").float()
        discounts = self._tensor(batch.bootstrap_discounts, "bootstrap_discounts").float()
        masks = self._tensor(batch.masks, "masks").bool()
        gamma = float(batch.metadata["gamma"])
        n_step = int(batch.metadata["n_step"])
        burn_in = int(getattr(self.model, "sequence_burn_in", 0))
        batch_size, sequence_length = actions.shape
        inner = list(range(burn_in, sequence_length - n_step))
        positions = [*inner, sequence_length - 1]
        feature_positions = torch.tensor(
            [position - burn_in for position in positions], device=self.device
        )
        step_actions = actions[:, positions]
        quantiles = torch.rand(batch_size, self.train_quantile_count, device=self.device)
        with self.autocast():
            features = self._train_model.encode_sequence(observations)
            predictions = self._train_model.quantiles_from_features(
                features[:, feature_positions], quantiles
            )
            selected = predictions.gather(
                3,
                step_actions[:, :, None, None].expand(-1, -1, self.train_quantile_count, 1),
            ).squeeze(-1)
            with torch.no_grad():
                target_quantiles = torch.rand(
                    batch_size, self.target_quantile_count, device=self.device
                )
                final_targets = self._final_step_targets(
                    next_observations, rewards, discounts, target_quantiles
                )
                if inner:
                    inner_targets = self._inner_step_targets(
                        observations,
                        features,
                        rewards,
                        target_quantiles,
                        inner=inner,
                        burn_in=burn_in,
                        n_step=n_step,
                        gamma=gamma,
                    )
                    targets = torch.cat([inner_targets, final_targets[:, None]], dim=1)
                else:
                    targets = final_targets[:, None]
        selected_fp32 = selected.float()
        targets_fp32 = targets.float()
        position_count = len(positions)
        losses = implicit_quantile_huber_loss(
            selected_fp32.flatten(0, 1),
            targets_fp32.flatten(0, 1),
            quantiles.repeat_interleave(position_count, dim=0),
        ).reshape(batch_size, position_count)
        valid = masks[:, positions]
        per_sample_losses = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        td_matrix = (selected_fp32.mean(2) - targets_fp32.mean(2)).detach().abs() * valid
        td_max = td_matrix.max(dim=1).values
        td_mean = td_matrix.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        priorities = (
            _SEQUENCE_PRIORITY_MAX_WEIGHT * td_max + (1.0 - _SEQUENCE_PRIORITY_MAX_WEIGHT) * td_mean
        )
        margin_loss = self._margin_loss(
            batch,
            predictions.float().mean(dim=2),
            step_actions,
            valid=valid,
        )
        policy_anchor_loss = self._policy_anchor_loss(
            predictions.float().mean(dim=2),
            self._sequence_anchor_q_values(observations, feature_positions),
            valid=valid,
        )
        return _LossComputation(
            per_sample_losses=per_sample_losses,
            priorities=priorities,
            selected=selected_fp32.flatten(0, 1),
            targets=targets_fp32.flatten(0, 1),
            step_actions=step_actions.reshape(-1),
            action_count=int(predictions.shape[-1]),
            rewards=rewards[:, -1],
            margin_loss=margin_loss,
            policy_anchor_loss=policy_anchor_loss,
            trained_positions=position_count,
            expected_q=predictions.float().mean(dim=2).flatten(0, 1),
            q_valid=valid.flatten(),
        )

    def _final_step_targets(
        self,
        next_observations: Any,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        target_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        next_q_values = self._train_model.q_values(
            next_observations, self.evaluation_quantile_count
        )
        next_actions = self._masked_argmax(next_q_values)
        target_values = (
            self.target_model(next_observations, target_quantiles)
            .gather(
                2,
                next_actions[:, None, None].expand(-1, self.target_quantile_count, 1),
            )
            .squeeze(-1)
        )
        return self._bootstrap_targets(rewards[:, -1, None], discounts[:, -1, None], target_values)

    def _inner_step_targets(
        self,
        observations: Any,
        online_features: torch.Tensor,
        rewards: torch.Tensor,
        target_quantiles: torch.Tensor,
        *,
        inner: list[int],
        burn_in: int,
        n_step: int,
        gamma: float,
    ) -> torch.Tensor:
        bootstrap_positions = torch.tensor(
            [position + n_step - burn_in for position in inner], device=self.device
        )
        evaluation_quantiles = self.model.evaluation_quantiles(
            self.evaluation_quantile_count, rewards.shape[0]
        )
        bootstrap_q_values = self._train_model.quantiles_from_features(
            online_features.detach()[:, bootstrap_positions], evaluation_quantiles
        ).mean(dim=2)
        bootstrap_actions = self._masked_argmax(bootstrap_q_values)
        target_features = self.target_model.encode_sequence(observations)
        target_values = (
            self.target_model.quantiles_from_features(
                target_features[:, bootstrap_positions], target_quantiles
            )
            .gather(
                3,
                bootstrap_actions[:, :, None, None].expand(-1, -1, self.target_quantile_count, 1),
            )
            .squeeze(-1)
        )
        kernel = gamma ** torch.arange(n_step, device=self.device, dtype=rewards.dtype)
        windows = rewards.unfold(1, n_step, 1)[:, burn_in : rewards.shape[1] - n_step]
        returns = (windows * kernel).sum(dim=-1)
        return self._bootstrap_targets(
            returns[:, :, None],
            torch.full_like(returns[:, :, None], gamma**n_step),
            target_values,
        )

    def _bootstrap_targets(
        self, rewards: torch.Tensor, discounts: torch.Tensor, target_values: torch.Tensor
    ) -> torch.Tensor:
        if not self.value_rescaling:
            return rewards + discounts * target_values
        return rescale_value(rewards + discounts * inverse_rescale_value(target_values))

    def _margin_loss(
        self,
        batch: TrainingBatch,
        expected_q: torch.Tensor,
        actions: torch.Tensor,
        *,
        valid: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.demonstration_margin_weight:
            return None
        flags = batch.metadata.get("expert_demo_flags", batch.metadata.get("demo_flags"))
        if flags is None:
            return None
        demo = torch.as_tensor(flags, dtype=torch.bool, device=self.device)
        if not bool(demo.any()):
            return None
        policy_q = self._masked_q_values(expected_q)
        if self.policy_action_ids is not None:
            allowed = torch.zeros(expected_q.shape[-1], dtype=torch.bool, device=self.device)
            allowed[list(self.policy_action_ids)] = True
            expert_actions = actions[demo]
            if not bool(allowed[expert_actions].all()):
                raise ValueError(
                    "expert demonstration contains an action outside policy_action_ids"
                )
        margins = torch.full(
            expected_q.shape, self.demonstration_margin, device=self.device
        ).scatter(-1, actions.unsqueeze(-1), 0.0)
        margin_losses = (policy_q + margins).max(dim=-1).values - expected_q.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        weight = demo.float() if valid is None else valid.float() * demo[:, None].float()
        return (margin_losses * weight).sum() / weight.sum().clamp_min(1.0)

    def _single_step_anchor_q_values(self, observations: Any) -> torch.Tensor | None:
        if self.policy_anchor_model is None:
            return None
        with torch.no_grad():
            values = self.policy_anchor_model.q_values(observations, self.evaluation_quantile_count)
        return cast(torch.Tensor, values).float()

    def _sequence_anchor_q_values(
        self, observations: Any, feature_positions: torch.Tensor
    ) -> torch.Tensor | None:
        if self.policy_anchor_model is None:
            return None
        batch_size = _first_tensor(observations).shape[0]
        with torch.no_grad():
            features = self.policy_anchor_model.encode_sequence(observations)
            quantiles = self.policy_anchor_model.evaluation_quantiles(
                self.evaluation_quantile_count, batch_size
            )
            values = self.policy_anchor_model.quantiles_from_features(
                features[:, feature_positions], quantiles
            ).mean(dim=2)
        return cast(torch.Tensor, values).float()

    def _policy_anchor_loss(
        self,
        expected_q: torch.Tensor,
        anchor_q: torch.Tensor | None,
        *,
        valid: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if anchor_q is None:
            return None
        advantages = expected_q - expected_q.mean(dim=-1, keepdim=True)
        anchor_advantages = anchor_q - anchor_q.mean(dim=-1, keepdim=True)
        losses = F.smooth_l1_loss(advantages, anchor_advantages, reduction="none").mean(dim=-1)
        if valid is None:
            return losses.mean()
        return (losses * valid).sum() / valid.sum().clamp_min(1)

    def _transfer_metrics(
        self,
        batch: TrainingBatch,
        computation: _LossComputation,
        iqn_loss: torch.Tensor,
        loss: torch.Tensor,
        gradient_norm: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[dict[str, float], list[float]]:
        selected = computation.selected
        targets = computation.targets
        rewards = computation.rewards
        action_counts = torch.bincount(
            computation.step_actions, minlength=computation.action_count
        ).float()
        action_probabilities = action_counts / action_counts.sum().clamp_min(1.0)
        positive = action_probabilities[action_probabilities > 0.0]
        action_entropy = -(positive * positive.log()).sum() / torch.log(
            torch.tensor(float(max(2, computation.action_count)), device=self.device)
        )
        importance = (
            weights
            if weights is not None
            else torch.ones(rewards.shape[0], device=self.device, dtype=torch.float32)
        )
        margin = (
            computation.margin_loss
            if computation.margin_loss is not None
            else torch.zeros((), device=self.device)
        )
        policy_anchor = (
            computation.policy_anchor_loss
            if computation.policy_anchor_loss is not None
            else torch.zeros((), device=self.device)
        )
        named = {
            "loss/iqn": iqn_loss.detach().float(),
            "loss/total": loss.detach().float(),
            "loss/demonstration_margin": margin.detach().float(),
            "loss/policy_anchor": policy_anchor.detach().float(),
            "loss/policy_anchor_weighted": (self.policy_anchor_weight * policy_anchor)
            .detach()
            .float(),
            "debug/gradient_norm": gradient_norm.detach().float(),
            "debug/td_abs_mean": computation.priorities.mean(),
            "debug/td_abs_max": computation.priorities.max(),
            "debug/reward_mean": rewards.mean(),
            "debug/reward_abs_max": rewards.abs().max(),
            "debug/q_selected_mean": selected.mean(),
            "debug/q_selected_max": selected.max(),
            "debug/q_selected_abs_max": selected.abs().max(),
            "debug/q_selected_std_mean": selected.std(dim=1, correction=0).mean(),
            "debug/target_mean": targets.mean(),
            "debug/target_abs_max": targets.abs().max(),
            "debug/target_std_mean": targets.std(dim=1, correction=0).mean(),
            "debug/action_entropy": action_entropy,
            "debug/action_unique_fraction": (action_counts > 0.0).float().mean(),
            "debug/importance_weight_mean": importance.mean(),
            "debug/importance_weight_min": importance.min(),
            "debug/initialized_exact_tensors": torch.tensor(
                float(self.initialized_exact_tensors), device=self.device
            ),
            "debug/initialized_expanded_tensors": torch.tensor(
                float(self.initialized_expanded_tensors), device=self.device
            ),
        }
        named.update(self._action_mask_metrics(computation))
        scalars = torch.stack(list(named.values()))
        transferred = torch.cat([scalars, computation.priorities]).cpu()
        metrics = dict(zip(named, transferred[: len(named)].tolist(), strict=True))
        priorities = transferred[len(named) :].tolist()
        gradient_norm_value = metrics["debug/gradient_norm"]
        metrics["debug/gradient_norm_max"] = gradient_norm_value
        metrics["debug/gradient_clipped_fraction"] = float(
            gradient_norm_value > self.gradient_clip_norm
        )
        metrics["debug/gradient_clip_coefficient"] = min(
            1.0, self.gradient_clip_norm / max(gradient_norm_value, 1e-12)
        )
        return metrics, priorities

    def _masked_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.policy_action_ids is None:
            return q_values
        action_count = q_values.shape[-1]
        if any(action >= action_count for action in self.policy_action_ids):
            raise ValueError(f"policy_action_ids must be inside [0, {action_count})")
        mask = torch.zeros(action_count, dtype=torch.bool, device=q_values.device)
        mask[list(self.policy_action_ids)] = True
        return q_values.masked_fill(~mask, -torch.inf)

    def _masked_argmax(self, q_values: torch.Tensor) -> torch.Tensor:
        return self._masked_q_values(q_values).argmax(dim=-1)

    def _action_mask_metrics(self, computation: _LossComputation) -> dict[str, torch.Tensor]:
        if self.policy_action_ids is None:
            return {}
        q_values = computation.expected_q[computation.q_valid]
        action_count = q_values.shape[-1]
        allowed = torch.zeros(action_count, dtype=torch.bool, device=self.device)
        allowed[list(self.policy_action_ids)] = True
        if bool(allowed.all()):
            return {}
        allowed_max = q_values[:, allowed].max(dim=-1).values
        excluded_max = q_values[:, ~allowed].max(dim=-1).values
        raw_greedy = q_values.argmax(dim=-1)
        return {
            "debug/q_allowed_max_mean": allowed_max.mean(),
            "debug/q_excluded_max_mean": excluded_max.mean(),
            "debug/q_excluded_advantage_mean": (excluded_max - allowed_max).mean(),
            "debug/greedy_masked_out_fraction": (~allowed[raw_greedy]).float().mean(),
        }

    def policy(self) -> _IQNPolicy:
        assert self.model is not None
        return _IQNPolicy(
            self.model,
            self.device,
            self.evaluation_quantile_count,
            self._current_epsilon(),
            self.policy_action_ids,
            self.online_quantile_distortion,
            self.evaluation_quantile_distortion,
            self.upper_cvar_alpha,
        )

    def _observation(self, value: Any, name: str) -> Any:
        value = tree_to_device(sanitize_finite(value), self.device)
        if _first_tensor(value).ndim < 1:
            raise ValueError(f"{name} tensors require a batch axis")
        return value

    @staticmethod
    def _sequence_target(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 1:
            return value[:, -1].reshape(-1)
        return value.reshape(-1)

    @staticmethod
    def _priority_transition_ids(batch: TrainingBatch) -> list[int]:
        configured = batch.metadata.get("priority_transition_ids")
        if configured is not None:
            return [int(value) for value in configured]
        return [int(value) for value in batch.transition_ids]

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
            "policy_action_ids": self.policy_action_ids,
            **self._policy_anchor_state(),
        }

    def state_dict_for_policy(self, policy_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build a resumable checkpoint whose train and target models match a policy."""

        assert self.model is not None
        expected = set(self.model.state_dict())
        if set(policy_state) != expected:
            raise ValueError("evaluated policy state does not match the IQN model")
        fresh_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            "model": dict(policy_state),
            "target_model": deepcopy(dict(policy_state)),
            "optimizer": fresh_optimizer.state_dict(),
            "update_count": self.update_count,
            "rng": self._rng_state(),
            "policy_action_ids": self.policy_action_ids,
            **self._policy_anchor_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate
        self.update_count = int(state["update_count"])
        self._restore_rng(state.get("rng", {}))
        self._restore_policy_anchor(state)

    def _policy_anchor_state(self) -> dict[str, Mapping[str, torch.Tensor]]:
        if self.policy_anchor_model is None:
            return {}
        return {"policy_anchor_model": self.policy_anchor_model.state_dict()}

    def _restore_policy_anchor(self, state: Mapping[str, Any]) -> None:
        if not self.policy_anchor_weight:
            self.policy_anchor_model = None
            return
        assert self.model is not None
        self.policy_anchor_model = deepcopy(self.model).to(self.device).eval()
        if self.policy_anchor_checkpoint is not None:
            self._load_configured_policy_anchor()
            return
        anchor_state = state.get("policy_anchor_model", state["model"])
        self.policy_anchor_model.load_state_dict(anchor_state)
        for parameter in self.policy_anchor_model.parameters():
            parameter.requires_grad_(False)

    def _load_configured_policy_anchor(self) -> None:
        if self.policy_anchor_checkpoint is None:
            return
        if self.policy_anchor_model is None:
            raise RuntimeError("policy anchor model must exist before loading a checkpoint")
        if not self.policy_anchor_checkpoint.is_file():
            raise FileNotFoundError(
                f"policy anchor checkpoint does not exist: {self.policy_anchor_checkpoint}"
            )
        loaded = TorchCheckpointCodec().load(self.policy_anchor_checkpoint)
        learner_state = loaded.get("learner", loaded)
        if not isinstance(learner_state, Mapping) or not isinstance(
            learner_state.get("model"), Mapping
        ):
            raise ValueError("policy anchor checkpoint does not contain learner model state")
        self._validate_anchor_action_contract(learner_state)
        self.policy_anchor_model.load_state_dict(learner_state["model"], strict=True)
        for parameter in self.policy_anchor_model.parameters():
            parameter.requires_grad_(False)

    def _load_model_initialization(self) -> None:
        if self.model_initialization_checkpoint is None:
            return
        if not self.model_initialization_checkpoint.is_file():
            raise FileNotFoundError(
                "model initialization checkpoint does not exist: "
                f"{self.model_initialization_checkpoint}"
            )
        loaded = TorchCheckpointCodec().load(self.model_initialization_checkpoint)
        learner_state = loaded.get("learner", loaded)
        if not isinstance(learner_state, Mapping) or not isinstance(
            learner_state.get("model"), Mapping
        ):
            raise ValueError("model initialization checkpoint has no learner model state")
        self._validate_anchor_action_contract(learner_state)
        assert self.model is not None
        target_state = self.model.state_dict()
        source_state = learner_state["model"]
        exact = 0
        expanded = 0
        for name, target in target_state.items():
            source = self._initialization_tensor(source_state, name)
            if not isinstance(source, torch.Tensor):
                continue
            if source.shape == target.shape:
                target.copy_(source)
                exact += 1
            elif self._can_expand_initialization(name, source, target):
                target.zero_()
                slices = tuple(slice(0, size) for size in source.shape)
                target[slices].copy_(source)
                expanded += 1
        self.model.load_state_dict(target_state, strict=True)
        self.initialized_exact_tensors = exact
        self.initialized_expanded_tensors = expanded

    @staticmethod
    def _initialization_tensor(
        source_state: Mapping[str, Any], target_name: str
    ) -> torch.Tensor | None:
        direct = source_state.get(target_name)
        if isinstance(direct, torch.Tensor):
            return direct
        prefix = "encoder.encoder.frame."
        if not target_name.startswith(prefix):
            return None
        behavior_cloning_name = "encoder.encoder." + target_name.removeprefix(prefix)
        source = source_state.get(behavior_cloning_name)
        return source if isinstance(source, torch.Tensor) else None

    @staticmethod
    def _can_expand_initialization(name: str, source: torch.Tensor, target: torch.Tensor) -> bool:
        expandable = (
            "frame.track.0.weight",
            "frame.telemetry.0.0.weight",
            "frame.projection.0.weight",
        )
        return (
            name.endswith(expandable)
            and source.ndim == target.ndim
            and all(old <= new for old, new in zip(source.shape, target.shape, strict=True))
        )

    def _validate_anchor_action_contract(self, state: Mapping[str, Any]) -> None:
        saved = state.get("policy_action_ids")
        if saved is None:
            if self.policy_action_ids is not None:
                raise ValueError("policy anchor checkpoint has no matching action contract")
            return
        if not isinstance(saved, (list, tuple)) or tuple(saved) != self.policy_action_ids:
            raise ValueError("policy anchor checkpoint action contract does not match")
