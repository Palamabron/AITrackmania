"""Behavior-cloning learner entry points."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Unpack, cast

import numpy as np
import torch

from trackmaniarl.algorithms.execution import (
    ResolvedTorchExecution,
    TorchExecutionConfig,
    resolve_torch_execution,
)
from trackmaniarl.core.contracts import ModelContract, ModelFactory, Policy
from trackmaniarl.models.backbones import project_hyperspherical_weights
from trackmaniarl.trackmania.imitation_learning._learner_checkpoint import (
    CheckpointComponents,
    capture_checkpoint,
    restore_checkpoint,
)
from trackmaniarl.trackmania.imitation_learning._learner_config import (
    LearnerConfiguration,
    LearnerOptions,
    learner_configuration,
)
from trackmaniarl.trackmania.imitation_learning._learner_metrics import (
    BehaviorCloningValidationBatch,
    ClassificationBatch,
    LossConfiguration,
    RecoveryMetricInputs,
    ValidationInputs,
    classification_loss_terms,
    masked_accuracy,
    recovery_metrics,
    sample_weights,
    steering_classes,
    steering_loss,
    to_device,
    transition_mask,
    validation_batch,
)
from trackmaniarl.trackmania.imitation_learning.model import (
    BehaviorCloningPolicy,
    LidarBehaviorCloningModel,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BehaviorCloningLearner:
    """Categorical learner used only by the offline ``trackmaniarl bc-train`` command."""

    accepted_model_contracts = frozenset({ModelContract.CATEGORICAL_POLICY})

    model_factory: ModelFactory | None
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    max_steps: int
    validation_interval: int
    early_stopping_patience: int
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    min_learning_rate: float
    gradient_clip_norm: float
    action_transition_weight: float
    class_weight_power: float
    focal_gamma: float
    steering_auxiliary_loss_weight: float
    horizontal_flip_augmentation: bool
    execution: TorchExecutionConfig
    seed: int

    def __init__(
        self,
        model: LidarBehaviorCloningModel | None = None,
        **options: Unpack[LearnerOptions],
    ) -> None:
        self.model = model
        self._apply_configuration(learner_configuration(**options))
        self._initialize_runtime()

    def _apply_configuration(self, configuration: LearnerConfiguration) -> None:
        for name in configuration.__dataclass_fields__:
            setattr(self, name, getattr(configuration, name))

    def _initialize_runtime(self) -> None:
        self.device = torch.device("cpu")
        self.resolved_execution: ResolvedTorchExecution | None = None
        self.scaler: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self.dataset_fingerprint: str | None = None

    def setup(self, context: Mapping[str, Any]) -> None:
        seed = int(context.get("seed", self.seed))
        _seed_everything(seed)
        self.model = self._resolve_model(context)
        self.resolved_execution = resolve_torch_execution(self.execution)
        self.device = self.resolved_execution.torch_device
        self.model.to(self.device)
        self._setup_optimizer()

    def _resolve_model(self, context: Mapping[str, Any]) -> LidarBehaviorCloningModel:
        if self.model is not None:
            return self.model
        factory = self.model_factory or context.get("model_factory")
        if factory is None:
            raise ValueError("BehaviorCloningLearner requires model_factory")
        return cast(LidarBehaviorCloningModel, factory.build())

    def _setup_optimizer(self) -> None:
        assert self.model is not None
        assert self.resolved_execution is not None
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
        model, _ = self._training_components()
        model.train()
        with self._autocast():
            batch = self._forward_batch(observations, labels, class_weights)
            loss = self._classification_loss(batch)
        gradient_norm = self._optimize(loss)
        return self._training_metrics(batch, loss, gradient_norm)

    def _training_components(
        self,
    ) -> tuple[LidarBehaviorCloningModel, torch.optim.Optimizer]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before training")
        return self.model, self.optimizer

    def _forward_batch(
        self,
        observations: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> ClassificationBatch:
        assert self.model is not None
        targets = labels.to(self.device)
        weights = class_weights.to(self.device)
        logits = self.model(to_device(observations, self.device))
        return ClassificationBatch(logits, targets, weights, observations)

    def _optimize(self, loss: torch.Tensor) -> torch.Tensor:
        model, optimizer = self._training_components()
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
        self.scaler.step(optimizer)
        project_hyperspherical_weights(model)
        self.scaler.update()
        return gradient_norm

    def _training_metrics(
        self, batch: ClassificationBatch, loss: torch.Tensor, gradient_norm: torch.Tensor
    ) -> dict[str, float]:
        accuracy = (batch.logits.argmax(dim=-1) == batch.targets).float().mean()
        transitions = self._transition_mask(batch.observations, batch.targets)
        inputs = RecoveryMetricInputs(
            batch.logits, batch.targets, batch.observations, self.model_action_count
        )
        return {
            "loss": float(loss.detach()),
            "accuracy": float(accuracy.detach()),
            "transition_accuracy": masked_accuracy(batch.logits, batch.targets, transitions),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": self.current_learning_rate(),
            **recovery_metrics(inputs),
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
        with torch.inference_mode(), self._autocast():
            batch = self._forward_batch(observations, labels, class_weights)
            numerator, denominator = self._classification_loss_terms(batch)
        inputs = ValidationInputs(
            batch, numerator, denominator, self._steering_classes(batch.logits.device)
        )
        return validation_batch(inputs)

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

    def _classification_loss(self, batch: ClassificationBatch) -> torch.Tensor:
        numerator, denominator = self._classification_loss_terms(batch)
        return numerator / denominator

    def _classification_loss_terms(
        self, batch: ClassificationBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        configuration = LossConfiguration(
            self.label_smoothing,
            self.steering_auxiliary_loss_weight,
            self.action_transition_weight,
            self.focal_gamma,
        )
        return classification_loss_terms(
            batch,
            configuration,
            self._steering_classes(batch.logits.device),
        )

    def _steering_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return steering_loss(logits, targets, self._steering_classes(logits.device))

    @staticmethod
    def _sample_weights(
        observations: Mapping[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        return sample_weights(observations, targets)

    def _transition_mask(
        self, observations: Mapping[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        return transition_mask(observations, targets, self.model_action_count)

    def _steering_classes(self, device: torch.device) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before reading actions")
        return steering_classes(self.model.action_ids, device)

    @property
    def model_action_count(self) -> int:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before reading action count")
        return self.model.action_count

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
        return capture_checkpoint(self._checkpoint_components())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restore_checkpoint(self._checkpoint_components(), state)

    def _checkpoint_components(self) -> CheckpointComponents:
        if (
            self.model is None
            or self.optimizer is None
            or self.scheduler is None
            or self.scaler is None
        ):
            raise RuntimeError("BehaviorCloningLearner.setup must run before checkpointing")
        return CheckpointComponents(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.dataset_fingerprint,
        )


__all__ = ["BehaviorCloningLearner", "BehaviorCloningValidationBatch"]
