from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.algorithms.value_based.objectives import ValueObjective


class DiscreteValueOptions(TypedDict, total=False):
    model_factory: Any | None
    learning_rate: float
    fraction_learning_rate: float
    target_update_interval: int
    target_tau: float
    gradient_clip_norm: float
    fraction_gradient_clip_norm: float
    burn_in: int
    exploration_epsilon: float
    policy_action_ids: tuple[int, ...] | None
    online_quantile_distortion: str
    evaluation_quantile_distortion: str
    upper_cvar_alpha: float
    value_rescaling: bool
    adaptive_gradient_clipper: Any | None
    diagnostics_interval_updates: int
    objectives: Sequence[ValueObjective]
    action_selector: Any | None
    model_initialization_checkpoint: str | Path | None
    warm_start_submodules: tuple[str, ...]
    warm_start_required_tensors: tuple[str, ...]
    freeze_warm_start_during_offline_pretraining: bool
    base_dir: str | Path
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


@dataclass(frozen=True, slots=True)
class DiscreteValueConfig:
    model_factory: Any | None = None
    learning_rate: float = 1e-4
    fraction_learning_rate: float = 1e-7
    target_update_interval: int = 1_000
    target_tau: float = 0.0
    gradient_clip_norm: float = 10.0
    fraction_gradient_clip_norm: float = 10.0
    burn_in: int = 0
    exploration_epsilon: float = 0.1
    policy_action_ids: tuple[int, ...] | None = None
    online_quantile_distortion: str = "neutral"
    evaluation_quantile_distortion: str = "neutral"
    upper_cvar_alpha: float = 0.25
    value_rescaling: bool = False
    adaptive_gradient_clipper: Any | None = None
    diagnostics_interval_updates: int = 100
    objectives: Sequence[ValueObjective] = ()
    action_selector: Any | None = None
    model_initialization_checkpoint: str | Path | None = None
    warm_start_submodules: tuple[str, ...] = ("encoder", "temporal")
    warm_start_required_tensors: tuple[str, ...] = ()
    freeze_warm_start_during_offline_pretraining: bool = False
    base_dir: str | Path = "."
    execution: TorchExecutionConfig | Mapping[str, Any] | None = None
    seed: int = 0

    def validate(self) -> None:
        if min(self.learning_rate, self.fraction_learning_rate) <= 0.0:
            raise ValueError("optimizer learning rates must be positive")
        if self.target_update_interval < 1 or not 0.0 <= self.target_tau <= 1.0:
            raise ValueError("target update configuration is invalid")
        if self.diagnostics_interval_updates < 1:
            raise ValueError("diagnostics interval must be positive")
        if min(self.gradient_clip_norm, self.fraction_gradient_clip_norm) <= 0.0:
            raise ValueError("gradient clips must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be between zero and one")
        self._validate_actions()

    def _validate_actions(self) -> None:
        action_ids = self.policy_action_ids
        if action_ids is None:
            return
        if not action_ids or len(set(action_ids)) != len(action_ids) or min(action_ids) < 0:
            raise ValueError("policy_action_ids must contain unique non-negative actions")
