from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from trackmaniarl.algorithms.execution import TorchExecutionConfig


class PPOOptions(TypedDict, total=False):
    model_factory: Any | None
    learning_rate: float
    clip_epsilon: float
    value_clip_epsilon: float
    gae_lambda: float
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    update_epochs: int
    minibatch_size: int
    target_kl: float | None
    normalize_observations: bool
    observation_clip: float
    normalize_rewards: bool
    reward_clip: float
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class PPOConfig:
    model_factory: Any | None = None
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    target_kl: float | None = 0.02
    normalize_observations: bool = True
    observation_clip: float = 10.0
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    execution: TorchExecutionConfig | Mapping[str, Any] | None = None
    seed: int = 0

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.max_gradient_norm <= 0.0:
            raise ValueError("learning_rate and max_gradient_norm must be positive")
        if not 0.0 < self.clip_epsilon < 1.0 or not 0.0 < self.value_clip_epsilon < 1.0:
            raise ValueError("PPO clipping epsilons must be between zero and one")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("Invalid PPO GAE coefficient")
        if min(self.entropy_coefficient, self.value_coefficient) < 0.0:
            raise ValueError("Invalid PPO loss coefficient")
        self._validate_schedule()

    def _validate_schedule(self) -> None:
        if self.update_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("Invalid PPO update schedule")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("Invalid PPO update schedule")
        if self.observation_clip <= 0.0 or self.reward_clip <= 0.0:
            raise ValueError("PPO normalization clips must be positive")
