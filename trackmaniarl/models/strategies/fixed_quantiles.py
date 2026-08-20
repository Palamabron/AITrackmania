"""Fixed uniform fractions used by QR-DQN."""

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
    quantile_huber_loss,
    uniform_support,
    weighted_expectation,
)


class FixedQuantileStrategy(nn.Module):
    required_representation = ValueRepresentation.FIXED_QUANTILE

    def __init__(self, quantile_count: int = 32) -> None:
        super().__init__()
        if quantile_count < 2:
            raise ValueError("quantile_count must be at least two")
        self.quantile_count = quantile_count

    def support(
        self,
        features: torch.Tensor,
        phase: ValuePhase,
        generator: torch.Generator | None,
    ) -> ValueSupport:
        del phase, generator
        return uniform_support(features, self.quantile_count, random=False, generator=None)

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
