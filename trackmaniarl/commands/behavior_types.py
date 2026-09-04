from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.trackmania.imitation_learning import BehaviorCloningLap


@dataclass(frozen=True, slots=True)
class _BehaviorCloningSelection:
    minimum_loss: float = float("inf")
    checkpoint_score: float = float("-inf")
    checkpoint_loss: float = float("inf")
    checkpoint_state: dict[str, Any] | None = None
    stale_validations: int = 0


@dataclass(frozen=True, slots=True)
class _BehaviorTrainingRequest:
    run: Any
    training: list[BehaviorCloningLap]
    validation: list[BehaviorCloningLap]
    resume: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _BehaviorData:
    train_observations: Mapping[str, torch.Tensor]
    train_labels: torch.Tensor
    validation_observations: Mapping[str, torch.Tensor]
    validation_labels: torch.Tensor
    weights: torch.Tensor
    generator: torch.Generator


@dataclass(frozen=True, slots=True)
class _BehaviorCheckpoints:
    best: Path
    latest: Path


@dataclass(slots=True)
class _BehaviorRuntime:
    run: Any
    data: _BehaviorData
    checkpoints: _BehaviorCheckpoints
    selection: _BehaviorCloningSelection = field(default_factory=_BehaviorCloningSelection)
    best_step: int = 0
    start_step: int = 1
