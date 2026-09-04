"""Scalar Q-value representation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from trackmaniarl.models.contracts import (
    AuxiliaryLoss,
    FractionLossContext,
    RiskSpec,
    ValuePhase,
    ValueRepresentation,
    ValueSupport,
)


class ScalarValueStrategy(nn.Module):
    required_representation = ValueRepresentation.SCALAR

    def support(
        self,
        features: torch.Tensor,
        phase: ValuePhase,
        generator: torch.Generator | None,
    ) -> ValueSupport:
        del phase, generator
        shape = (*features.shape[:-1], 1)
        point = torch.full(shape, 0.5, device=features.device, dtype=torch.float32)
        weight = torch.ones_like(point)
        boundaries = torch.cat([torch.zeros_like(point), torch.ones_like(point)], dim=-1)
        return ValueSupport(point, weight, boundaries)

    def expectation(
        self, values: torch.Tensor, support: ValueSupport, risk: RiskSpec
    ) -> torch.Tensor:
        del support, risk
        return values.squeeze(-2)

    def regression_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        support: ValueSupport,
    ) -> torch.Tensor:
        del support
        return F.smooth_l1_loss(predictions.squeeze(-1), targets.squeeze(-1), reduction="none")

    def auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def auxiliary_loss(self, context: FractionLossContext) -> AuxiliaryLoss | None:
        del context
        return None
