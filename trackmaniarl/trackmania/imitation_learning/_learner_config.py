from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, Unpack, cast

from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.contracts import ModelFactory


class LearnerOptions(TypedDict, total=False):
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
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class LearnerConfiguration:
    model_factory: ModelFactory | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.01
    max_steps: int = 20_000
    validation_interval: int = 100
    early_stopping_patience: int = 30
    lr_scheduler_factor: float = 0.3
    lr_scheduler_patience: int = 5
    min_learning_rate: float = 1e-6
    gradient_clip_norm: float = 5.0
    action_transition_weight: float = 1.0
    class_weight_power: float = 0.5
    focal_gamma: float = 0.0
    steering_auxiliary_loss_weight: float = 0.0
    horizontal_flip_augmentation: bool = False
    execution: TorchExecutionConfig = field(default_factory=TorchExecutionConfig)
    seed: int = 0

    @classmethod
    def from_options(cls, options: LearnerOptions) -> LearnerConfiguration:
        values = cast(dict[str, Any], dict(options))
        execution = values.get("execution")
        if isinstance(execution, Mapping):
            values["execution"] = TorchExecutionConfig(**execution)
        elif execution is None:
            values["execution"] = TorchExecutionConfig()
        configuration = cls(**values)
        validate_learner_configuration(configuration)
        return configuration


def validate_learner_configuration(configuration: LearnerConfiguration) -> None:
    _validate_optimizer(configuration)
    _validate_schedule(configuration)


def _validate_optimizer(configuration: LearnerConfiguration) -> None:
    values = (
        configuration.learning_rate <= 0.0,
        configuration.weight_decay < 0.0,
        configuration.min_learning_rate < 0.0,
        configuration.gradient_clip_norm <= 0.0,
        configuration.action_transition_weight < 1.0,
        not 0.0 <= configuration.class_weight_power <= 1.0,
        configuration.focal_gamma < 0.0,
        configuration.steering_auxiliary_loss_weight < 0.0,
    )
    if any(values):
        raise ValueError("behavior cloning optimizer parameters are invalid")
    if not 0.0 <= configuration.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")


def _validate_schedule(configuration: LearnerConfiguration) -> None:
    counts = (
        configuration.max_steps,
        configuration.validation_interval,
        configuration.early_stopping_patience,
        configuration.lr_scheduler_patience,
    )
    if min(counts) < 1 or not 0.0 < configuration.lr_scheduler_factor < 1.0:
        raise ValueError("behavior cloning schedule parameters must be positive")


def learner_configuration(**options: Unpack[LearnerOptions]) -> LearnerConfiguration:
    return LearnerConfiguration.from_options(options)
