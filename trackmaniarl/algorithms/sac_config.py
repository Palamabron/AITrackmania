from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from trackmaniarl.algorithms.execution import TorchExecutionConfig


class SACOptions(TypedDict, total=False):
    model_factory: Any | None
    target_tau: float
    learning_rate: float
    entropy_coefficient: float
    target_entropy: float | None
    learn_entropy_coefficient: bool
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class SACConfig:
    model_factory: Any | None = None
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.2
    target_entropy: float | None = None
    learn_entropy_coefficient: bool = True
    execution: TorchExecutionConfig | Mapping[str, Any] | None = None
    seed: int = 0

    def validate(self) -> None:
        values = (self.target_tau, self.learning_rate, self.entropy_coefficient)
        if min(values) <= 0.0 or self.target_tau > 1.0:
            raise ValueError("target_tau, learning_rate, and entropy_coefficient must be positive")


class REDQOptions(TypedDict, total=False):
    model_factory: Any | None
    target_tau: float
    learning_rate: float
    entropy_coefficient: float
    target_subset_size: int
    policy_update_interval: int
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class REDQConfig:
    model_factory: Any | None = None
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.2
    target_subset_size: int = 2
    policy_update_interval: int = 20
    execution: TorchExecutionConfig | Mapping[str, Any] | None = None
    seed: int = 0

    def validate(self) -> None:
        values = (self.target_tau, self.learning_rate, self.entropy_coefficient)
        if min(values) <= 0.0 or self.target_tau > 1.0:
            raise ValueError("target_tau, learning_rate, and entropy_coefficient must be positive")
        if self.target_subset_size < 1 or self.policy_update_interval < 1:
            raise ValueError("target_subset_size and policy_update_interval must be positive")


class TQCOptions(SACOptions, total=False):
    top_quantiles_to_drop_per_critic: int


@dataclass(frozen=True, slots=True)
class TQCConfig(SACConfig):
    top_quantiles_to_drop_per_critic: int = 2

    def validate(self) -> None:
        SACConfig.validate(self)
        if self.top_quantiles_to_drop_per_critic < 0:
            raise ValueError("top_quantiles_to_drop_per_critic must be non-negative")


class DiscreteSACOptions(SACOptions, total=False):
    q_clip_epsilon: float
    entropy_penalty_coefficient: float


@dataclass(frozen=True, slots=True)
class DiscreteSACConfig(SACConfig):
    q_clip_epsilon: float = 0.5
    entropy_penalty_coefficient: float = 0.5

    def validate(self) -> None:
        SACConfig.validate(self)
        if min(self.q_clip_epsilon, self.entropy_penalty_coefficient) < 0.0:
            raise ValueError(
                "SD-SAC clipping and entropy penalty coefficients must be non-negative"
            )
