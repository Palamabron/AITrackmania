"""Random fractions used by IQN."""

from __future__ import annotations

import torch
from torch import nn

from trackmaniarl.models.contracts import (
    AuxiliaryLoss,
    FractionLossContext,
    RiskSpec,
    ValuePhase,
    ValueRepresentation,
    ValueSupport,
)
from trackmaniarl.models.strategies._common import (
    SupportSampling,
    UniformSupportSpec,
    quantile_huber_loss,
    uniform_support,
    weighted_expectation,
)


class RandomQuantileStrategy(nn.Module):
    required_representation = ValueRepresentation.IMPLICIT_QUANTILE

    def __init__(
        self,
        train_quantile_count: int = 64,
        target_quantile_count: int = 64,
        evaluation_quantile_count: int = 32,
    ) -> None:
        super().__init__()
        if min(train_quantile_count, target_quantile_count, evaluation_quantile_count) < 2:
            raise ValueError("IQN quantile counts must be at least two")
        self.train_quantile_count = train_quantile_count
        self.target_quantile_count = target_quantile_count
        self.evaluation_quantile_count = evaluation_quantile_count

    def support(
        self,
        features: torch.Tensor,
        phase: ValuePhase,
        generator: torch.Generator | None,
    ) -> ValueSupport:
        count = {
            ValuePhase.TRAIN: self.train_quantile_count,
            ValuePhase.TARGET: self.target_quantile_count,
            ValuePhase.EVALUATE: self.evaluation_quantile_count,
        }[phase]
        sampling = (
            SupportSampling.MIDPOINTS if phase is ValuePhase.EVALUATE else SupportSampling.RANDOM
        )
        return uniform_support(features, UniformSupportSpec(count, sampling, generator))

    def expectation(
        self, values: torch.Tensor, support: ValueSupport, risk: RiskSpec
    ) -> torch.Tensor:
        return weighted_expectation(values, support, risk)

    def regression_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        support: ValueSupport,
    ) -> torch.Tensor:
        return quantile_huber_loss(predictions.float(), targets.float(), support.points)

    def auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def auxiliary_loss(self, context: FractionLossContext) -> AuxiliaryLoss | None:
        del context
        return None
