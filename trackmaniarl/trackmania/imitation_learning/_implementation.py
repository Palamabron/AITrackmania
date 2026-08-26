"""Implementation for the modular behavior-cloning package."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from trackmaniarl.algorithms.execution import (
    ResolvedTorchExecution,
    TorchExecutionConfig,
    resolve_torch_execution,
)
from trackmaniarl.core.contracts import FeaturePipeline, ModelContract, ModelFactory, Policy
from trackmaniarl.models.backbones import project_hyperspherical_weights
from trackmaniarl.models.composite import FrameBatchAdapter
from trackmaniarl.models.temporal import GruTemporalCore, IdentityTemporalCore
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    load_demonstration,
    resample_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.encoders import LidarSensorEncoder, LidarSimbaSensorEncoder

RECOVERY_DATASET_FORMAT_V1 = "trackmaniarl-bc-recovery-v1"
RECOVERY_DATASET_FORMAT_V2 = "trackmaniarl-bc-recovery-v2"
RECOVERY_DATASET_FORMAT = "trackmaniarl-bc-recovery-v3"
SAMPLE_WEIGHT_KEY = "bc_sample_weight"
STUDENT_ACTION_KEY = "bc_student_action"
INTERVENTION_KEY = "bc_intervention"
STATE_ERROR_KEY = "bc_state_error"
ELITE_LAP_WEIGHT_TEMPERATURE_S = 0.35
MINIMUM_LAP_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class BehaviorCloningValidationBatch:
    loss: float
    loss_numerator: float
    loss_denominator: float
    correct: int
    total: int
    per_action_correct: torch.Tensor
    per_action_count: torch.Tensor
    transition_correct: int
    transition_count: int
    steering_correct: int
    steering_count: int
    steering_transition_correct: int
    steering_transition_count: int
    weighted_correct: float
    sample_weight_total: float
    intervention_correct: int
    intervention_count: int
    student_disagreement_correct: int
    student_disagreement_count: int


class _FactorizedActionHead(nn.Module):
    def __init__(self, input_dim: int, action_ids: tuple[int, ...]) -> None:
        super().__init__()
        canonical = torch.tensor(action_ids, dtype=torch.long)
        self.steering = nn.Linear(input_dim, 13)
        self.drive_mode = nn.Linear(input_dim, 6)
        self.steering_indices: torch.Tensor
        self.drive_mode_indices: torch.Tensor
        self.register_buffer("steering_indices", canonical // 6, persistent=False)
        self.register_buffer("drive_mode_indices", canonical % 6, persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        steering = self.steering(features).index_select(-1, self.steering_indices)
        drive_mode = self.drive_mode(features).index_select(-1, self.drive_mode_indices)
        return cast(torch.Tensor, steering + drive_mode)


class LidarBehaviorCloningModel(nn.Module):
    """Categorical policy over an explicit compact action set and frame history."""

    def __init__(
        self,
        *,
        action_ids: tuple[int, ...],
        telemetry_dim: int = 26,
        history_length: int = 1,
        spatial_bins: int = 12,
        burn_in: int = 0,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
        previous_action_conditioning: bool = False,
        previous_action_embedding_dim: int = 16,
        minimum_action_hold_steps: int = 1,
        switch_logit_margin: float = 0.0,
        masked_telemetry_indices: tuple[int, ...] = (),
        simba_backbone: Mapping[str, Any] | None = None,
        factorized_action_head: bool = False,
    ) -> None:
        super().__init__()
        self.action_ids = tuple(action_ids)
        select_brake_tap_actions(self.action_ids)
        self.action_count = len(self.action_ids)
        if previous_action_embedding_dim < 1 or minimum_action_hold_steps < 1:
            raise ValueError("previous-action policy dimensions must be positive")
        if switch_logit_margin < 0.0:
            raise ValueError("switch_logit_margin must be non-negative")
        self.previous_action_conditioning = previous_action_conditioning
        self.previous_action_start = self.action_count
        self.minimum_action_hold_steps = minimum_action_hold_steps
        self.switch_logit_margin = switch_logit_margin
        sensor: dict[str, Any] = {
            "telemetry_dim": telemetry_dim,
            "spatial_bins": spatial_bins,
            "lidar_channels": lidar_channels,
            "telemetry_group_dims": telemetry_group_dims,
            "hidden_dim": encoder_hidden_dim,
            "output_dim": encoder_output_dim,
            "masked_telemetry_indices": masked_telemetry_indices,
        }
        self.encoder: LidarSensorEncoder | LidarSimbaSensorEncoder
        self.temporal: IdentityTemporalCore | GruTemporalCore
        self.feedforward_history = simba_backbone is not None
        if self.feedforward_history:
            if burn_in:
                raise ValueError("feed-forward Simba history does not use burn-in")
            assert simba_backbone is not None
            self.encoder = LidarSimbaSensorEncoder(sensor, simba_backbone, history_length)
            self.temporal = IdentityTemporalCore(self.encoder.output_dim)
            self.burn_in = 0
        else:
            self.encoder = LidarSensorEncoder(**sensor)
            self.temporal = (
                IdentityTemporalCore(encoder_output_dim)
                if history_length == 1
                else GruTemporalCore(encoder_output_dim, encoder_output_dim)
            )
            self.burn_in = burn_in if history_length > 1 else 0
        self.previous_action_embedding = (
            nn.Embedding(self.action_count + 1, previous_action_embedding_dim)
            if previous_action_conditioning
            else None
        )
        head_input_dim = self.encoder.output_dim + (
            previous_action_embedding_dim if previous_action_conditioning else 0
        )
        self.head: nn.Linear | _FactorizedActionHead = (
            _FactorizedActionHead(head_input_dim, self.action_ids)
            if factorized_action_head
            else nn.Linear(head_input_dim, self.action_count)
        )

    def initial_policy_state(self, device: torch.device) -> Any:
        return self.temporal.initial_state(1, device)

    def policy_logits(
        self, observation: Mapping[str, torch.Tensor], state: Any
    ) -> tuple[torch.Tensor, Any]:
        if not self.feedforward_history and observation["lidar"].ndim == 4:
            return self(observation), state
        frames = {key: observation[key] for key in ("lidar", "lidar_mask", "telemetry")}
        if self.feedforward_history:
            features = self.encoder(frames)
        else:
            batch = FrameBatchAdapter.flatten(frames, sequence=False)
            features = self.encoder(cast(Any, batch.frames))
        encoded, next_state = self.temporal.step(features, state)
        return self._logits(encoded, observation), next_state

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        frames = {key: observation[key] for key in ("lidar", "lidar_mask", "telemetry")}
        if self.feedforward_history:
            return self._logits(self.encoder(frames), observation)
        sequence = observation["lidar"].ndim == 4
        batch = FrameBatchAdapter.flatten(frames, sequence=sequence)
        features = batch.restore(self.encoder(cast(Any, batch.frames)))
        encoded = self.temporal.unroll(features, self.burn_in)[:, -1]
        return self._logits(encoded, observation)

    def _logits(
        self, encoded: torch.Tensor, observation: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.previous_action_embedding is not None:
            previous_action = observation.get("previous_action")
            if previous_action is None:
                raise ValueError("previous_action is required by this behavior-cloning model")
            if previous_action.ndim == 2:
                previous_action = previous_action[:, -1]
            encoded = torch.cat(
                (encoded, self.previous_action_embedding(previous_action.long())), dim=-1
            )
        return cast(torch.Tensor, self.head(encoded))


class LidarBehaviorCloningModelFactory:
    model_contract = ModelContract.CATEGORICAL_POLICY

    def __init__(
        self,
        *,
        action_ids: tuple[int, ...],
        telemetry_dim: int = 26,
        history_length: int = 1,
        spatial_bins: int = 12,
        burn_in: int = 0,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
        previous_action_conditioning: bool = False,
        previous_action_embedding_dim: int = 16,
        minimum_action_hold_steps: int = 1,
        switch_logit_margin: float = 0.0,
        masked_telemetry_indices: tuple[int, ...] = (),
        simba_backbone: Mapping[str, Any] | None = None,
        factorized_action_head: bool = False,
    ) -> None:
        self.action_ids = tuple(action_ids)
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length
        self.spatial_bins = spatial_bins
        self.burn_in = burn_in
        self.lidar_channels = lidar_channels
        self.telemetry_group_dims = telemetry_group_dims
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.previous_action_conditioning = previous_action_conditioning
        self.previous_action_embedding_dim = previous_action_embedding_dim
        self.minimum_action_hold_steps = minimum_action_hold_steps
        self.switch_logit_margin = switch_logit_margin
        self.masked_telemetry_indices = masked_telemetry_indices
        self.simba_backbone = dict(simba_backbone) if simba_backbone is not None else None
        self.factorized_action_head = factorized_action_head

    def build(self) -> LidarBehaviorCloningModel:
        return LidarBehaviorCloningModel(
            action_ids=self.action_ids,
            telemetry_dim=self.telemetry_dim,
            history_length=self.history_length,
            spatial_bins=self.spatial_bins,
            burn_in=self.burn_in,
            lidar_channels=self.lidar_channels,
            telemetry_group_dims=self.telemetry_group_dims,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
            previous_action_conditioning=self.previous_action_conditioning,
            previous_action_embedding_dim=self.previous_action_embedding_dim,
            minimum_action_hold_steps=self.minimum_action_hold_steps,
            switch_logit_margin=self.switch_logit_margin,
            masked_telemetry_indices=self.masked_telemetry_indices,
            simba_backbone=self.simba_backbone,
            factorized_action_head=self.factorized_action_head,
        )


class BehaviorCloningPolicy:
    def __init__(self, model: LidarBehaviorCloningModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.previous_action = model.previous_action_start
        self.action_hold_steps = 0
        self.temporal_state = model.initial_policy_state(device)

    def reset_episode(self) -> None:
        self.previous_action = self.model.previous_action_start
        self.action_hold_steps = 0
        self.temporal_state = self.model.initial_policy_state(self.device)

    def act(self, observation: Mapping[str, torch.Tensor], *, deterministic: bool = False) -> int:
        del deterministic
        batched = {key: value.to(self.device).unsqueeze(0) for key, value in observation.items()}
        if self.model.previous_action_conditioning:
            batched["previous_action"] = torch.tensor(
                [self.previous_action], device=self.device, dtype=torch.long
            )
        with torch.inference_mode():
            logits, self.temporal_state = self.model.policy_logits(batched, self.temporal_state)
            logits = logits.squeeze(0)
        action = int(logits.argmax().item())
        if self.previous_action < self.model.action_count and action != self.previous_action:
            switch_margin = float(logits[action] - logits[self.previous_action])
            if (
                self.action_hold_steps < self.model.minimum_action_hold_steps
                or switch_margin < self.model.switch_logit_margin
            ):
                action = self.previous_action
        self.action_hold_steps = self.action_hold_steps + 1 if action == self.previous_action else 1
        self.previous_action = action
        return action


class BehaviorCloningLearner:
    accepted_model_contracts = frozenset({ModelContract.CATEGORICAL_POLICY})

    """Categorical learner used only by the offline ``trackmaniarl bc-train`` command."""

    def __init__(
        self,
        model: LidarBehaviorCloningModel | None = None,
        *,
        model_factory: ModelFactory | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.01,
        max_steps: int = 20_000,
        validation_interval: int = 100,
        early_stopping_patience: int = 30,
        lr_scheduler_factor: float = 0.3,
        lr_scheduler_patience: int = 5,
        min_learning_rate: float = 1e-6,
        gradient_clip_norm: float = 5.0,
        action_transition_weight: float = 1.0,
        class_weight_power: float = 0.5,
        focal_gamma: float = 0.0,
        steering_auxiliary_loss_weight: float = 0.0,
        horizontal_flip_augmentation: bool = False,
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if (
            learning_rate <= 0.0
            or weight_decay < 0.0
            or min_learning_rate < 0.0
            or gradient_clip_norm <= 0.0
            or action_transition_weight < 1.0
            or not 0.0 <= class_weight_power <= 1.0
            or focal_gamma < 0.0
            or steering_auxiliary_loss_weight < 0.0
        ):
            raise ValueError("behavior cloning optimizer parameters are invalid")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if (
            min(max_steps, validation_interval, early_stopping_patience, lr_scheduler_patience) < 1
            or not 0.0 < lr_scheduler_factor < 1.0
        ):
            raise ValueError("behavior cloning schedule parameters must be positive")
        self.model = model
        self.model_factory = model_factory
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.max_steps = max_steps
        self.validation_interval = validation_interval
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.min_learning_rate = min_learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.action_transition_weight = action_transition_weight
        self.class_weight_power = class_weight_power
        self.focal_gamma = focal_gamma
        self.steering_auxiliary_loss_weight = steering_auxiliary_loss_weight
        self.horizontal_flip_augmentation = horizontal_flip_augmentation
        self.execution = (
            TorchExecutionConfig(**execution)
            if isinstance(execution, Mapping)
            else execution or TorchExecutionConfig()
        )
        self.seed = seed
        self.device = torch.device("cpu")
        self.resolved_execution: ResolvedTorchExecution | None = None
        self.scaler: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self.dataset_fingerprint: str | None = None

    def setup(self, context: Mapping[str, Any]) -> None:
        seed = int(context.get("seed", self.seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        factory = self.model_factory or context.get("model_factory")
        if self.model is None:
            if factory is None:
                raise ValueError("BehaviorCloningLearner requires model_factory")
            self.model = factory.build()
        self.resolved_execution = resolve_torch_execution(self.execution)
        self.device = self.resolved_execution.torch_device
        self.model.to(self.device)
        self.scaler = cast(Any, torch.amp).GradScaler(
            self.device.type,
            enabled=self.resolved_execution.scaler_enabled,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.min_learning_rate,
        )

    def train_batch(
        self,
        observations: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> dict[str, float]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before training")
        self.model.train()
        targets = labels.to(self.device)
        weights = class_weights.to(self.device)
        with self._autocast():
            logits = self.model(_to_device(observations, self.device))
            loss = self._classification_loss(logits, targets, weights, observations)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        project_hyperspherical_weights(self.model)
        self.scaler.update()
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        transition_mask = self._transition_mask(observations, targets)
        transition_accuracy = self._masked_accuracy(logits, targets, transition_mask)
        recovery_metrics = self._recovery_metrics(logits, targets, observations)
        return {
            "loss": float(loss.detach()),
            "accuracy": float(accuracy.detach()),
            "transition_accuracy": transition_accuracy,
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": self.current_learning_rate(),
            **recovery_metrics,
        }

    def evaluate_batch(
        self,
        observations: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> BehaviorCloningValidationBatch:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before evaluation")
        self.model.eval()
        with torch.inference_mode():
            with self._autocast():
                logits = self.model(_to_device(observations, self.device))
                targets = labels.to(self.device)
                weights = class_weights.to(self.device)
                numerator, denominator = self._classification_loss_terms(
                    logits, targets, weights, observations
                )
                loss = numerator / denominator
            predicted = logits.argmax(dim=-1)
            correct_mask = predicted == targets
            correct = int(correct_mask.sum().item())
            action_count = logits.shape[-1]
            per_action_count = torch.bincount(targets, minlength=action_count)
            per_action_correct = torch.bincount(targets[correct_mask], minlength=action_count)
            transition_mask = self._transition_mask(observations, targets)
            transition_count = int(transition_mask.sum().item())
            transition_correct = int((correct_mask & transition_mask).sum().item())
            steering = self._steering_classes(logits.device)
            predicted_steering = steering[predicted]
            target_steering = steering[targets]
            steering_correct_mask = predicted_steering == target_steering
            steering_transition_mask = self._steering_transition_mask(
                observations, targets, steering
            )
            weighted_correct, sample_weight_total = self._weighted_validation_counts(
                correct_mask, targets, observations
            )
            intervention = self._validation_subset_counts(
                correct_mask, targets, observations.get(INTERVENTION_KEY)
            )
            student = observations.get(STUDENT_ACTION_KEY)
            disagreement = (
                None
                if student is None
                else (student.to(targets.device).long() < self.model_action_count)
                & (student.to(targets.device).long() != targets)
            )
            student_disagreement = self._validation_subset_counts(
                correct_mask, targets, disagreement
            )
        return BehaviorCloningValidationBatch(
            loss=float(loss),
            loss_numerator=float(numerator),
            loss_denominator=float(denominator),
            correct=correct,
            total=int(targets.numel()),
            per_action_correct=per_action_correct.cpu(),
            per_action_count=per_action_count.cpu(),
            transition_correct=transition_correct,
            transition_count=transition_count,
            steering_correct=int(steering_correct_mask.sum().item()),
            steering_count=int(targets.numel()),
            steering_transition_correct=int(
                (steering_correct_mask & steering_transition_mask).sum().item()
            ),
            steering_transition_count=int(steering_transition_mask.sum().item()),
            weighted_correct=weighted_correct,
            sample_weight_total=sample_weight_total,
            intervention_correct=intervention[0],
            intervention_count=intervention[1],
            student_disagreement_correct=student_disagreement[0],
            student_disagreement_count=student_disagreement[1],
        )

    def _weighted_validation_counts(
        self,
        correct: torch.Tensor,
        targets: torch.Tensor,
        observations: Mapping[str, torch.Tensor],
    ) -> tuple[float, float]:
        weights = self._sample_weights(observations, targets)
        return float((correct.float() * weights).sum()), float(weights.sum())

    def _autocast(self) -> Any:
        if self.resolved_execution is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before autocast")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.resolved_execution.precision]
        return torch.autocast(
            device_type=self.device.type,
            dtype=dtype,
            enabled=self.resolved_execution.precision != "float32",
        )

    def execution_manifest(self) -> Mapping[str, object]:
        if self.resolved_execution is None:
            return {
                "resolved": False,
                "requested_device": self.execution.device,
                "requested_precision": self.execution.precision,
            }
        return {"resolved": True, **self.resolved_execution.manifest()}

    def bind_dataset(self, fingerprint: str) -> None:
        if not fingerprint:
            raise ValueError("behavior-cloning dataset fingerprint must not be empty")
        self.dataset_fingerprint = fingerprint

    @staticmethod
    def _validation_subset_counts(
        correct: torch.Tensor,
        targets: torch.Tensor,
        subset: torch.Tensor | None,
    ) -> tuple[int, int]:
        mask = (
            torch.zeros_like(targets, dtype=torch.bool)
            if subset is None
            else subset.to(targets.device).bool()
        )
        return int((correct & mask).sum().item()), int(mask.sum().item())

    def _classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor,
        observations: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        numerator, denominator = self._classification_loss_terms(
            logits, targets, class_weights, observations
        )
        return numerator / denominator

    def _classification_loss_terms(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor,
        observations: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses = functional.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if self.steering_auxiliary_loss_weight:
            losses += self.steering_auxiliary_loss_weight * self._steering_loss(logits, targets)
        multipliers = self._sample_weights(observations, targets)
        transition_mask = self._transition_mask(observations, targets)
        multipliers = multipliers * torch.where(
            transition_mask,
            torch.full_like(multipliers, self.action_transition_weight),
            torch.ones_like(multipliers),
        )
        if self.focal_gamma:
            target_probability = logits.softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
            multipliers *= (1.0 - target_probability).pow(self.focal_gamma)
        denominator = (class_weights[targets] * multipliers).sum().clamp_min(1e-8)
        return (losses * multipliers).sum(), denominator

    def _steering_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        steering = self._steering_classes(logits.device)
        steering_bins = torch.unique(steering, sorted=True)
        grouped = []
        for steering_bin in steering_bins:
            selected = logits[:, steering == steering_bin]
            grouped.append(torch.logsumexp(selected, dim=-1) - np.log(selected.shape[-1]))
        steering_targets = torch.searchsorted(steering_bins, steering[targets])
        return functional.cross_entropy(
            torch.stack(grouped, dim=-1), steering_targets, reduction="none"
        )

    @staticmethod
    def _sample_weights(
        observations: Mapping[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        weights = observations.get(SAMPLE_WEIGHT_KEY)
        if weights is None:
            return torch.ones_like(targets, dtype=torch.float32)
        values = weights.to(targets.device, dtype=torch.float32).reshape(targets.shape)
        if not bool(torch.isfinite(values).all()) or bool((values <= 0.0).any()):
            raise ValueError("behavior-cloning sample weights must be finite and positive")
        return values

    def _recovery_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        observations: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        predictions = logits.argmax(dim=-1)
        sample_weights = self._sample_weights(observations, targets)
        weighted = (predictions == targets).float() * sample_weights
        metrics = {
            "weighted_accuracy": float(weighted.sum() / sample_weights.sum()),
            "sample_weight_mean": float(sample_weights.mean()),
        }
        metrics.update(self._subset_metrics(predictions, targets, observations))
        return metrics

    def _subset_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        observations: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        intervention = observations.get(INTERVENTION_KEY)
        intervention_mask = (
            torch.zeros_like(targets, dtype=torch.bool)
            if intervention is None
            else intervention.to(targets.device).bool()
        )
        student = observations.get(STUDENT_ACTION_KEY)
        disagreement_mask = (
            torch.zeros_like(targets, dtype=torch.bool)
            if student is None
            else (student.to(targets.device).long() < self.model_action_count)
            & (student.to(targets.device).long() != targets)
        )
        return {
            "intervention_accuracy": BehaviorCloningLearner._masked_accuracy(
                predictions, targets, intervention_mask
            ),
            "intervention_count": float(intervention_mask.sum()),
            "student_disagreement_accuracy": BehaviorCloningLearner._masked_accuracy(
                predictions, targets, disagreement_mask
            ),
            "student_disagreement_count": float(disagreement_mask.sum()),
        }

    def _transition_mask(
        self, observations: Mapping[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        previous = observations.get("expert_previous_action")
        if previous is None:
            previous = observations.get("previous_action")
        if previous is None:
            return torch.zeros_like(targets, dtype=torch.bool)
        previous = previous.to(targets.device).long()
        return (previous < self.model_action_count) & (previous != targets)

    def _steering_classes(self, device: torch.device) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before reading actions")
        _, actions = select_brake_tap_actions(self.model.action_ids)
        action_array = np.asarray(actions, dtype=np.float32)
        steering_bins = np.rint((action_array[:, 2] + 1.0) * 6.0).astype(np.int64)
        return torch.from_numpy(steering_bins).to(device)

    def _steering_transition_mask(
        self,
        observations: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        steering: torch.Tensor,
    ) -> torch.Tensor:
        previous = observations.get("expert_previous_action")
        if previous is None:
            return torch.zeros_like(targets, dtype=torch.bool)
        previous = previous.to(targets.device).long()
        valid = previous < self.model_action_count
        safe_previous = previous.clamp_max(self.model_action_count - 1)
        return valid & (steering[safe_previous] != steering[targets])

    @property
    def model_action_count(self) -> int:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before reading action count")
        return self.model.action_count

    @staticmethod
    def _masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        predictions = logits.argmax(dim=-1) if logits.ndim > 1 else logits
        accuracy = (predictions[mask] == targets[mask]).float().mean()
        return float(accuracy.detach())

    def policy(self) -> Policy:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before policy")
        self.model.eval()
        return BehaviorCloningPolicy(self.model, self.device)

    def step_scheduler(self, validation_loss: float) -> float:
        if self.scheduler is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before scheduling")
        self.scheduler.step(validation_loss)
        return self.current_learning_rate()

    def current_learning_rate(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before reading learning rate")
        return float(self.optimizer.param_groups[0]["lr"])

    def update(self, batch: Any) -> Mapping[str, float]:
        del batch
        raise RuntimeError("BehaviorCloningLearner only supports trackmaniarl bc-train")

    def validation_update(self, batch: Any) -> Mapping[str, float]:
        observations = batch.observations
        actions = batch.actions
        if not isinstance(observations, Mapping) or not isinstance(actions, torch.Tensor):
            raise TypeError("BC validation requires mapping observations and tensor actions")
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before validation")
        sequence = actions.ndim > 1
        labels = actions.long().reshape(-1).remainder(self.model.action_count)
        prepared = {
            key: value.reshape(-1, *value.shape[2:]) if sequence else value
            for key, value in observations.items()
            if isinstance(value, torch.Tensor)
        }
        if self.model.previous_action_conditioning and "previous_action" not in prepared:
            prepared["previous_action"] = torch.full_like(labels, self.model.previous_action_start)
        metrics = self.train_batch(prepared, labels, torch.ones(self.model.action_count))
        return {f"validation/{key}": value for key, value in metrics.items()}

    def state_dict(self) -> Mapping[str, Any]:
        if (
            self.model is None
            or self.optimizer is None
            or self.scheduler is None
            or self.scaler is None
        ):
            raise RuntimeError("BehaviorCloningLearner.setup must run before checkpointing")
        return {
            "schema_version": "trackmaniarl-bc-checkpoint-v2",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "policy_action_ids": self.model.action_ids,
            "dataset_fingerprint": self.dataset_fingerprint,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "accelerator": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            self.model is None
            or self.optimizer is None
            or self.scheduler is None
            or self.scaler is None
        ):
            raise RuntimeError("BehaviorCloningLearner.setup must run before restoring")
        saved_action_ids = state.get("policy_action_ids")
        if saved_action_ids is not None and tuple(saved_action_ids) != self.model.action_ids:
            raise ValueError("behavior-cloning checkpoint action contract does not match")
        saved_dataset = state.get("dataset_fingerprint")
        if (
            self.dataset_fingerprint is not None
            and saved_dataset is not None
            and saved_dataset != self.dataset_fingerprint
        ):
            raise ValueError("behavior-cloning checkpoint dataset fingerprint does not match")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        scheduler = state.get("scheduler")
        if scheduler is not None:
            self.scheduler.load_state_dict(scheduler)
        scaler = state.get("scaler")
        if not isinstance(scaler, Mapping):
            raise ValueError("checkpoint is missing gradient scaler state")
        self.scaler.load_state_dict(dict(scaler))
        rng = state.get("rng")
        if isinstance(rng, Mapping):
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            accelerator = rng["accelerator"]
            if torch.cuda.is_available() and accelerator:
                torch.cuda.set_rng_state_all(accelerator)


@dataclass(frozen=True, slots=True)
class BehaviorCloningLap:
    observations: tuple[Mapping[str, torch.Tensor], ...]
    labels: torch.Tensor
    quality_weight: float = 1.0
    source_id: str = ""


@dataclass(frozen=True, slots=True)
class RecoveryContract:
    map_uid: str
    geometry_sha256: str
    action_repeat_frames: int
    decision_interval_ms: float | None
    control_alignment: str

    def __post_init__(self) -> None:
        if not self.map_uid or not _is_sha256(self.geometry_sha256):
            raise ValueError("recovery map and geometry identity are invalid")
        if self.action_repeat_frames < 1:
            raise ValueError("recovery action repeat must be positive")
        if self.decision_interval_ms is not None and (
            not np.isfinite(self.decision_interval_ms) or self.decision_interval_ms <= 0.0
        ):
            raise ValueError("recovery decision interval must be finite and positive")
        if self.decision_interval_ms is not None and self.action_repeat_frames != 1:
            raise ValueError("recovery decision interval requires action repeat one")
        if self.control_alignment != "frame_start":
            raise ValueError("recovery controls must use frame_start alignment")

    @classmethod
    def from_demonstration(cls, demonstration: Demonstration) -> RecoveryContract:
        return cls(
            demonstration.map_uid,
            demonstration.geometry_sha256,
            demonstration.action_repeat_frames,
            demonstration.decision_interval_ms,
            demonstration.control_alignment,
        )


@dataclass(frozen=True, slots=True)
class RecoveryProvenance:
    contract: RecoveryContract
    source_demonstration_sha256: str
    source_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        hashes = (self.source_demonstration_sha256, self.source_checkpoint_sha256)
        if any(value is not None and not _is_sha256(value) for value in hashes):
            raise ValueError("recovery source hashes must be SHA-256 digests")

    @classmethod
    def from_demonstration(
        cls,
        demonstration: Demonstration,
        *,
        contract: RecoveryContract | None = None,
        source_checkpoint_sha256: str | None = None,
    ) -> RecoveryProvenance:
        return cls(
            contract or RecoveryContract.from_demonstration(demonstration),
            _demonstration_sha256(demonstration),
            source_checkpoint_sha256,
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _demonstration_sha256(demonstration: Demonstration) -> str:
    digest = hashlib.sha256()
    metadata = (
        demonstration.map_uid,
        demonstration.geometry_sha256,
        str(demonstration.action_repeat_frames),
        str(demonstration.decision_interval_ms),
        demonstration.control_alignment,
        f"{demonstration.finish_time_s:.12g}",
    )
    digest.update("\0".join(metadata).encode())
    for values in (demonstration.frames, demonstration.actions, demonstration.controls):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_behavior_cloning_recovery(
    path: str | Path,
    frames: np.ndarray,
    labels: np.ndarray,
    episode_starts: np.ndarray,
    action_ids: tuple[int, ...],
    *,
    provenance: RecoveryProvenance,
    sample_weights: np.ndarray | None = None,
    student_actions: np.ndarray | None = None,
    interventions: np.ndarray | None = None,
    state_errors: np.ndarray | None = None,
) -> Path:
    """Persist DAgger states with compact expert labels."""

    if frames.ndim != 2 or frames.shape[1] != 33:
        raise ValueError("recovery frames must have shape (steps, 33)")
    if not np.isfinite(frames).all():
        raise ValueError("recovery frames must be finite")
    if labels.shape != (len(frames),) or episode_starts.shape != (len(frames),):
        raise ValueError("recovery labels and episode starts must match frames")
    if len(frames) < 1 or not bool(episode_starts[0]):
        raise ValueError("recovery data must begin with an episode start")
    if np.any(labels < 0) or np.any(labels >= len(action_ids)):
        raise ValueError("recovery data contains an invalid compact action")
    _validate_recovery_metadata(
        len(frames),
        len(action_ids),
        sample_weights,
        student_actions,
        interventions,
        state_errors,
    )
    target = Path(path)
    if target.suffix.lower() != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    common = (
        frames.astype(np.float32, copy=False),
        labels.astype(np.int64, copy=False),
        episode_starts.astype(np.bool_, copy=False),
        np.asarray(action_ids, dtype=np.int64),
    )
    contract = provenance.contract
    np.savez_compressed(
        target,
        format=np.asarray(RECOVERY_DATASET_FORMAT),
        map_uid=np.asarray(contract.map_uid),
        geometry_sha256=np.asarray(contract.geometry_sha256),
        action_repeat_frames=np.asarray(contract.action_repeat_frames, dtype=np.int32),
        decision_interval_ms=np.asarray(contract.decision_interval_ms or 0.0, dtype=np.float64),
        control_alignment=np.asarray(contract.control_alignment),
        source_demonstration_sha256=np.asarray(provenance.source_demonstration_sha256),
        source_checkpoint_sha256=np.asarray(provenance.source_checkpoint_sha256 or ""),
        frames=common[0],
        labels=common[1],
        episode_starts=common[2],
        action_ids=common[3],
        sample_weight=(
            np.ones(len(frames), dtype=np.float32)
            if sample_weights is None
            else sample_weights.astype(np.float32, copy=False)
        ),
        student_action=(
            np.full(len(frames), len(action_ids), dtype=np.int64)
            if student_actions is None
            else student_actions.astype(np.int64, copy=False)
        ),
        intervention=(
            np.zeros(len(frames), dtype=np.bool_)
            if interventions is None
            else interventions.astype(np.bool_, copy=False)
        ),
        state_error=(
            np.zeros(len(frames), dtype=np.float32)
            if state_errors is None
            else state_errors.astype(np.float32, copy=False)
        ),
    )
    return target


def _validate_recovery_metadata(
    sample_count: int,
    action_count: int,
    sample_weights: np.ndarray | None,
    student_actions: np.ndarray | None,
    interventions: np.ndarray | None,
    state_errors: np.ndarray | None,
) -> None:
    if sample_weights is not None and (
        sample_weights.shape != (sample_count,)
        or not np.isfinite(sample_weights).all()
        or np.any(sample_weights <= 0.0)
    ):
        raise ValueError("recovery sample weights must be finite, positive, and match frames")
    if student_actions is not None and (
        student_actions.shape != (sample_count,)
        or np.any(student_actions < 0)
        or np.any(student_actions >= action_count)
    ):
        raise ValueError("recovery student actions must be compact actions matching frames")
    if interventions is not None and interventions.shape != (sample_count,):
        raise ValueError("recovery interventions must match frames")
    if state_errors is not None and (
        state_errors.shape != (sample_count,)
        or not np.isfinite(state_errors).all()
        or np.any(state_errors < 0.0)
    ):
        raise ValueError("recovery state errors must be finite, non-negative, and match frames")


def _load_recovery_metadata(
    data: Any, sample_count: int, action_count: int
) -> dict[str, np.ndarray]:
    required = {"sample_weight", "student_action", "intervention", "state_error"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"recovery archive is missing metadata {sorted(missing)}")
    weights = np.asarray(data["sample_weight"], dtype=np.float32)
    students = np.asarray(data["student_action"], dtype=np.int64)
    interventions = np.asarray(data["intervention"], dtype=np.bool_)
    state_errors = np.asarray(data["state_error"], dtype=np.float32)
    _validate_recovery_metadata(
        sample_count,
        action_count,
        weights,
        None if bool(np.all(students == action_count)) else students,
        interventions,
        state_errors,
    )
    return {
        SAMPLE_WEIGHT_KEY: weights,
        STUDENT_ACTION_KEY: students,
        INTERVENTION_KEY: interventions,
        STATE_ERROR_KEY: state_errors,
    }


def _attach_recovery_metadata(
    observation: dict[str, torch.Tensor], metadata: Mapping[str, np.ndarray], index: int
) -> None:
    observation[SAMPLE_WEIGHT_KEY] = torch.tensor(metadata[SAMPLE_WEIGHT_KEY][index])
    observation[STUDENT_ACTION_KEY] = torch.tensor(
        metadata[STUDENT_ACTION_KEY][index], dtype=torch.long
    )
    observation[INTERVENTION_KEY] = torch.tensor(
        metadata[INTERVENTION_KEY][index], dtype=torch.bool
    )
    observation[STATE_ERROR_KEY] = torch.tensor(metadata[STATE_ERROR_KEY][index])


def _attach_default_recovery_metadata(
    observation: dict[str, torch.Tensor], action_count: int
) -> None:
    observation[SAMPLE_WEIGHT_KEY] = torch.tensor(1.0)
    observation[STUDENT_ACTION_KEY] = torch.tensor(action_count, dtype=torch.long)
    observation[INTERVENTION_KEY] = torch.tensor(False)
    observation[STATE_ERROR_KEY] = torch.tensor(0.0)


def _load_recovery_provenance(data: Any, path: Path) -> RecoveryProvenance:
    required = {
        "map_uid",
        "geometry_sha256",
        "action_repeat_frames",
        "decision_interval_ms",
        "control_alignment",
        "source_demonstration_sha256",
        "source_checkpoint_sha256",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"recovery archive is missing provenance {sorted(missing)}: {path}")
    raw_action_repeat = data["action_repeat_frames"].item()
    if isinstance(raw_action_repeat, (bool, np.bool_)) or not isinstance(
        raw_action_repeat, (int, np.integer)
    ):
        raise ValueError(f"recovery action repeat must be an integer: {path}")
    interval_ms = float(data["decision_interval_ms"].item())
    checkpoint_sha256 = str(data["source_checkpoint_sha256"].item()) or None
    try:
        contract = RecoveryContract(
            map_uid=str(data["map_uid"].item()),
            geometry_sha256=str(data["geometry_sha256"].item()),
            action_repeat_frames=int(raw_action_repeat),
            decision_interval_ms=interval_ms or None,
            control_alignment=str(data["control_alignment"].item()),
        )
        return RecoveryProvenance(
            contract,
            str(data["source_demonstration_sha256"].item()),
            checkpoint_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"recovery archive has invalid provenance: {path}") from exc


def _validate_recovery_contract(
    actual: RecoveryContract,
    expected: RecoveryContract,
    path: Path,
) -> None:
    if actual.map_uid != expected.map_uid:
        raise ValueError(f"recovery map UID does not match the training map: {path}")
    if actual.geometry_sha256 != expected.geometry_sha256:
        raise ValueError(f"recovery geometry does not match the feature geometry: {path}")
    if actual.action_repeat_frames != expected.action_repeat_frames:
        raise ValueError(f"recovery action repeat does not match the environment: {path}")
    if actual.control_alignment != expected.control_alignment:
        raise ValueError(f"recovery control alignment does not match the dataset: {path}")
    actual_interval = actual.decision_interval_ms
    expected_interval = expected.decision_interval_ms
    if actual_interval is None and expected_interval is None:
        return
    if (
        actual_interval is None
        or expected_interval is None
        or not np.isclose(actual_interval, expected_interval, rtol=0.0, atol=0.05)
    ):
        raise ValueError(f"recovery decision interval does not match the environment: {path}")


def load_behavior_cloning_recovery(
    paths: Sequence[Path],
    pipeline: FeaturePipeline,
    action_ids: tuple[int, ...],
    *,
    expected_contract: RecoveryContract,
    expected_source_demonstration_sha256: frozenset[str],
    previous_action_conditioning: bool = False,
) -> list[BehaviorCloningLap]:
    """Rebuild feature histories for DAgger states and keep episodes separate."""

    if not expected_source_demonstration_sha256 or any(
        not _is_sha256(value) for value in expected_source_demonstration_sha256
    ):
        raise ValueError("expected recovery source hashes must be non-empty SHA-256 digests")
    resolved_paths = tuple(path.resolve() for path in paths)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("behavior-cloning recovery paths must be unique")
    laps: list[BehaviorCloningLap] = []
    for path in resolved_paths:
        with np.load(path, allow_pickle=False) as data:
            if "format" not in data.files:
                raise ValueError(f"recovery archive is missing its format marker: {path}")
            format_name = str(data["format"].item())
            if format_name in {RECOVERY_DATASET_FORMAT_V1, RECOVERY_DATASET_FORMAT_V2}:
                raise ValueError(
                    f"recovery archive predates map and timing provenance; regenerate it: {path}"
                )
            if format_name != RECOVERY_DATASET_FORMAT:
                raise ValueError(f"unsupported behavior-cloning recovery format: {path}")
            provenance = _load_recovery_provenance(data, path)
            _validate_recovery_contract(provenance.contract, expected_contract, path)
            if provenance.source_demonstration_sha256 not in expected_source_demonstration_sha256:
                raise ValueError(
                    f"recovery source demonstration is not present in --demo inputs: {path}"
                )
            required = {"action_ids", "frames", "labels", "episode_starts"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"recovery archive is missing data {sorted(missing)}: {path}")
            stored_ids = tuple(int(value) for value in data["action_ids"].tolist())
            if stored_ids != action_ids:
                raise ValueError(f"recovery action IDs do not match the model: {path}")
            frames = np.asarray(data["frames"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            starts = np.asarray(data["episode_starts"], dtype=np.bool_)
            metadata = _load_recovery_metadata(data, len(frames), len(action_ids))
        if frames.ndim != 2 or frames.shape[1] != 33:
            raise ValueError(f"recovery frames have an invalid shape: {path}")
        if not np.isfinite(frames).all():
            raise ValueError(f"recovery frames contain non-finite values: {path}")
        if labels.shape != (len(frames),) or starts.shape != (len(frames),):
            raise ValueError(f"recovery labels and episode starts do not match frames: {path}")
        if np.any(labels < 0) or np.any(labels >= len(action_ids)):
            raise ValueError(f"recovery data contains an invalid compact action: {path}")
        boundaries = np.flatnonzero(starts)
        if not len(boundaries) or boundaries[0] != 0:
            raise ValueError(f"recovery data does not begin with an episode: {path}")
        for episode, (start, stop) in enumerate(
            zip(boundaries, [*boundaries[1:], len(frames)], strict=True)
        ):
            reset = getattr(pipeline, "reset_episode", None)
            if callable(reset):
                reset()
            observations = []
            previous_action = len(action_ids)
            for offset, (frame, label) in enumerate(
                zip(frames[start:stop], labels[start:stop], strict=True)
            ):
                observation = dict(
                    _clone_mapping_observation(pipeline.transform_observation(frame))
                )
                observation["expert_previous_action"] = torch.tensor(
                    previous_action, dtype=torch.long
                )
                if previous_action_conditioning:
                    observation["previous_action"] = torch.tensor(previous_action, dtype=torch.long)
                _attach_recovery_metadata(observation, metadata, int(start) + offset)
                observations.append(observation)
                previous_action = int(label)
            laps.append(
                BehaviorCloningLap(
                    tuple(observations),
                    torch.from_numpy(labels[start:stop].copy()),
                    source_id=f"{path.resolve()}#episode-{episode}",
                )
            )
    return laps


def _clone_mapping_observation(observation: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(observation, Mapping):
        raise TypeError("behavior cloning recovery requires mapping lidar observations")
    return {key: value.detach().clone() for key, value in observation.items()}


def load_behavior_cloning_laps(
    paths: Sequence[Path],
    pipeline: FeaturePipeline,
    action_ids: tuple[int, ...],
    *,
    expected_action_repeat_frames: int | None = None,
    expected_decision_interval_ms: float | None = None,
    action_lead_ms: float = 0.0,
    aggregate_controls: bool = False,
    previous_action_conditioning: bool = False,
) -> list[BehaviorCloningLap]:
    """Convert full demonstration laps into compact supervised examples."""

    mapping = {action: index for index, action in enumerate(action_ids)}
    demonstrations = [(path, load_demonstration(path)) for path in paths]
    best_finish_time_s = min(demo.finish_time_s for _, demo in demonstrations)
    laps: list[BehaviorCloningLap] = []
    for path, demo in demonstrations:
        _validate_demonstration_contract(
            demo,
            pipeline,
            path,
            expected_action_repeat_frames,
            expected_decision_interval_ms,
        )
        validate_recording_quality(demo)
        frames, actions = resample_demonstration(
            demo,
            expected_decision_interval_ms,
            action_lead_ms=action_lead_ms,
            aggregate_controls=aggregate_controls,
        )
        lap_weight = float(
            np.clip(
                np.exp((best_finish_time_s - demo.finish_time_s) / ELITE_LAP_WEIGHT_TEMPERATURE_S),
                MINIMUM_LAP_WEIGHT,
                1.0,
            )
        )
        reset = getattr(pipeline, "reset_episode", None)
        if callable(reset):
            reset()
        labels = []
        observations = []
        previous_action = len(action_ids)
        for frame, action in zip(frames[:-1], actions, strict=True):
            source_action = int(action)
            if source_action not in mapping:
                raise ValueError(
                    f"demo {path} contains action {source_action} outside compact action IDs"
                )
            observation = pipeline.transform_observation(frame)
            if not isinstance(observation, Mapping):
                raise TypeError("behavior cloning requires mapping lidar observations")
            prepared = {key: value.detach().clone() for key, value in observation.items()}
            prepared["expert_previous_action"] = torch.tensor(previous_action, dtype=torch.long)
            _attach_default_recovery_metadata(prepared, len(action_ids))
            prepared[SAMPLE_WEIGHT_KEY] = torch.tensor(lap_weight, dtype=torch.float32)
            if previous_action_conditioning:
                prepared["previous_action"] = torch.tensor(previous_action, dtype=torch.long)
            observations.append(prepared)
            label = mapping[source_action]
            labels.append(label)
            previous_action = label
        laps.append(
            BehaviorCloningLap(
                tuple(observations),
                torch.tensor(labels, dtype=torch.long),
                quality_weight=lap_weight,
                source_id=str(path.resolve()),
            )
        )
    if len(laps) < 3:
        raise ValueError("behavior cloning requires at least three complete demonstration laps")
    return laps


def _validate_demonstration_contract(
    demonstration: Demonstration,
    pipeline: FeaturePipeline,
    path: Path,
    expected_action_repeat_frames: int | None,
    expected_decision_interval_ms: float | None,
) -> None:
    geometry = getattr(pipeline, "geometry", None)
    if geometry is not None:
        if demonstration.map_uid != geometry.map_uid:
            raise ValueError(
                f"demo {path} map UID {demonstration.map_uid!r} does not match "
                f"feature geometry {geometry.map_uid!r}"
            )
        if demonstration.geometry_sha256 != geometry.sha256:
            raise ValueError(f"demo {path} was recorded against a different geometry asset")
    if expected_decision_interval_ms is not None:
        recorded_interval_ms = demonstration.decision_interval_ms
        if recorded_interval_ms is not None and not np.isclose(
            recorded_interval_ms,
            expected_decision_interval_ms,
            rtol=0.0,
            atol=0.05,
        ):
            raise ValueError(
                f"demo {path} decision interval {recorded_interval_ms:g}ms does not match "
                f"environment decision interval {expected_decision_interval_ms:g}ms"
            )
        return
    if (
        expected_action_repeat_frames is not None
        and demonstration.action_repeat_frames != expected_action_repeat_frames
    ):
        raise ValueError(
            f"demo {path} action repeat {demonstration.action_repeat_frames} does not match "
            f"environment action repeat {expected_action_repeat_frames}"
        )


def split_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], seed: int
) -> tuple[list[BehaviorCloningLap], list[BehaviorCloningLap]]:
    """Split complete laps into an 80/20 deterministic train/validation partition."""

    generator = torch.Generator().manual_seed(seed)
    if len(laps) < 2:
        raise ValueError("behavior-cloning split requires at least two complete episodes")
    order = torch.randperm(len(laps), generator=generator).tolist()
    validation_count = max(1, round(len(laps) * 0.2))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    elite_index = max(range(len(laps)), key=lambda index: laps[index].quality_weight)
    if elite_index in validation_indices:
        replacement = training_indices[0]
        validation_indices[validation_indices.index(elite_index)] = replacement
        training_indices[0] = elite_index
    validation = [laps[index] for index in validation_indices]
    training = [laps[index] for index in training_indices]
    return training, validation


def augment_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], action_ids: tuple[int, ...]
) -> list[BehaviorCloningLap]:
    """Add a reflected copy of each local-frame demonstration lap."""

    mapping = _horizontal_flip_action_indices(action_ids)
    reflected = [
        BehaviorCloningLap(
            tuple(
                _horizontal_flip_conditioned_observation(observation, mapping)
                for observation in lap.observations
            ),
            mapping[lap.labels],
            quality_weight=lap.quality_weight,
            source_id=f"{lap.source_id}#horizontal-reflection",
        )
        for lap in laps
    ]
    return [*laps, *reflected]


def _horizontal_flip_conditioned_observation(
    observation: Mapping[str, torch.Tensor], mapping: torch.Tensor
) -> dict[str, torch.Tensor]:
    reflected = horizontal_flip_observation(observation)
    for key in ("expert_previous_action", "previous_action", STUDENT_ACTION_KEY):
        previous_action = observation.get(key)
        if previous_action is not None:
            index = int(previous_action)
            reflected[key] = (
                mapping[index].clone()
                if index < len(mapping)
                else torch.tensor(len(mapping), dtype=torch.long)
            )
    for key in (SAMPLE_WEIGHT_KEY, INTERVENTION_KEY, STATE_ERROR_KEY):
        value = observation.get(key)
        if value is not None:
            reflected[key] = value.clone()
    return reflected


def horizontal_flip_observation(
    observation: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reflect the v59 local TrackMania observation across the car forward axis."""

    lidar = observation["lidar"]
    telemetry = observation["telemetry"]
    if lidar.shape[-2] != 8 or telemetry.shape[-1] not in {46, 49}:
        raise ValueError("horizontal flip requires the 8-channel, 46- or 49-feature observation")
    reflected_lidar = lidar.clone()
    reflected_lidar[..., 0, :] = -lidar[..., 2, :]
    reflected_lidar[..., 1, :] = lidar[..., 3, :]
    reflected_lidar[..., 2, :] = -lidar[..., 0, :]
    reflected_lidar[..., 3, :] = lidar[..., 1, :]
    reflected_lidar[..., 4, :] = -lidar[..., 4, :]
    reflected_lidar[..., 5, :] = lidar[..., 5, :]
    reflected_telemetry = telemetry.clone()
    reflected_telemetry[..., 6] = -telemetry[..., 6]
    reflected_telemetry[..., 10] = telemetry[..., 11]
    reflected_telemetry[..., 11] = telemetry[..., 10]
    reflected_telemetry[..., 12] = telemetry[..., 13]
    reflected_telemetry[..., 13] = telemetry[..., 12]
    offset = 3 if telemetry.shape[-1] == 49 else 0
    if offset:
        reflected_telemetry[..., 17] = -telemetry[..., 17]
    lateral_indices = (18, 19, 22, 29, 31, 32, 39, 41)
    for index in lateral_indices:
        reflected_telemetry[..., index + offset] = -telemetry[..., index + offset]
    reflected_telemetry[..., 34 + offset] = -telemetry[..., 36 + offset]
    reflected_telemetry[..., 35 + offset] = telemetry[..., 37 + offset]
    reflected_telemetry[..., 36 + offset] = -telemetry[..., 34 + offset]
    reflected_telemetry[..., 37 + offset] = telemetry[..., 35 + offset]
    reflected = {key: value.clone() for key, value in observation.items()}
    reflected["lidar"] = reflected_lidar
    reflected["telemetry"] = reflected_telemetry
    return reflected


def _horizontal_flip_action_indices(action_ids: tuple[int, ...]) -> torch.Tensor:
    _, table = select_brake_tap_actions(action_ids)
    mirrored: list[int] = []
    for control in table:
        match = next(
            (
                index
                for index, candidate in enumerate(table)
                if np.array_equal(candidate[:2], control[:2])
                and np.isclose(candidate[2], -control[2])
            ),
            None,
        )
        if match is None:
            raise ValueError("horizontal flip requires left-right paired compact actions")
        mirrored.append(match)
    return torch.tensor(mirrored, dtype=torch.long)


def flatten_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], indices: torch.Tensor | None = None
) -> tuple[list[Mapping[str, torch.Tensor]], torch.Tensor]:
    observations = [observation for lap in laps for observation in lap.observations]
    labels = torch.cat([lap.labels for lap in laps])
    if indices is None:
        return observations, labels
    return [observations[int(index)] for index in indices], labels[indices]


def class_weights(labels: torch.Tensor, action_count: int, *, power: float = 0.5) -> torch.Tensor:
    if not 0.0 <= power <= 1.0:
        raise ValueError("class weight power must be in [0, 1]")
    counts = torch.bincount(labels, minlength=action_count).float()
    observed = counts > 0
    if not bool(observed.any()):
        raise ValueError("behavior cloning labels must not be empty")
    weights = torch.ones_like(counts)
    weights[observed] = counts[observed].pow(-power)
    weights[observed] /= weights[observed].mean()
    return weights.clamp(0.5, 3.0)


def collate_behavior_cloning(
    observations: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("behavior cloning batch must not be empty")
    return {
        key: torch.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy tensor state before the next optimizer update mutates it."""

    return deepcopy(dict(state))


def _to_device(
    observations: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in observations.items()}
