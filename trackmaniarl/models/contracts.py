"""Typed contracts shared by composable value models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from trackmaniarl.core.pytree import PyTree


class ValueRepresentation(StrEnum):
    SCALAR = "scalar"
    FIXED_QUANTILE = "fixed_quantile"
    IMPLICIT_QUANTILE = "implicit_quantile"


class ValuePhase(StrEnum):
    TRAIN = "train"
    TARGET = "target"
    EVALUATE = "evaluate"


class RiskDistortion(StrEnum):
    NEUTRAL = "neutral"
    UPPER_CVAR = "upper_cvar"


@dataclass(frozen=True, slots=True)
class RiskSpec:
    distortion: RiskDistortion = RiskDistortion.NEUTRAL
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("risk alpha must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ValueSupport:
    """Quantile locations and integration masses for one feature tensor."""

    points: torch.Tensor
    weights: torch.Tensor
    boundaries: torch.Tensor | None = None
    entropy: torch.Tensor | None = None

    def detached_points(self) -> ValueSupport:
        return ValueSupport(
            points=self.points.detach(),
            weights=self.weights,
            boundaries=self.boundaries,
            entropy=self.entropy,
        )

    def detached(self) -> ValueSupport:
        return ValueSupport(
            points=self.points.detach(),
            weights=self.weights.detach(),
            boundaries=(self.boundaries.detach() if self.boundaries is not None else None),
            entropy=self.entropy.detach() if self.entropy is not None else None,
        )


@dataclass(frozen=True, slots=True)
class FractionLossContext:
    support: ValueSupport
    boundary_values: torch.Tensor
    midpoint_values: torch.Tensor
    valid: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class AuxiliaryLoss:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


@runtime_checkable
class SensorEncoder(Protocol):
    output_dim: int

    def __call__(self, frames: PyTree) -> torch.Tensor: ...


@runtime_checkable
class TemporalCore(Protocol):
    input_dim: int
    output_dim: int

    def unroll(self, features: torch.Tensor, burn_in: int) -> torch.Tensor: ...

    def initial_state(self, batch_size: int, device: torch.device) -> PyTree: ...

    def step(self, feature: torch.Tensor, state: PyTree) -> tuple[torch.Tensor, PyTree]: ...


@runtime_checkable
class ValueHead(Protocol):
    representation: ValueRepresentation
    action_count: int

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor: ...

    def evaluate_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor: ...


ValueEvaluator = Callable[[torch.Tensor, ValueSupport, torch.Tensor], torch.Tensor]


@runtime_checkable
class ValueStrategy(Protocol):
    required_representation: ValueRepresentation

    def support(
        self,
        features: torch.Tensor,
        phase: ValuePhase,
        generator: torch.Generator | None,
    ) -> ValueSupport: ...

    def expectation(
        self,
        values: torch.Tensor,
        support: ValueSupport,
        risk: RiskSpec,
    ) -> torch.Tensor: ...

    def regression_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        support: ValueSupport,
    ) -> torch.Tensor: ...

    def auxiliary_parameters(self) -> tuple[nn.Parameter, ...]: ...

    def auxiliary_loss(self, context: FractionLossContext) -> AuxiliaryLoss | None: ...
