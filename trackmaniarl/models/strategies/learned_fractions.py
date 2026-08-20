"""Learned quantile fractions used by FQF."""

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
from trackmaniarl.models.strategies._common import quantile_huber_loss, weighted_expectation


class LearnedFractionStrategy(nn.Module):
    required_representation = ValueRepresentation.IMPLICIT_QUANTILE

    def __init__(
        self,
        feature_dim: int,
        fraction_count: int = 32,
        entropy_coefficient: float = 1e-3,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or fraction_count < 2 or entropy_coefficient < 0.0:
            raise ValueError("FQF dimensions must be positive and entropy non-negative")
        self.feature_dim = feature_dim
        self.fraction_count = fraction_count
        self.entropy_coefficient = entropy_coefficient
        self.proposal = nn.Linear(feature_dim, fraction_count)
        nn.init.zeros_(self.proposal.weight)
        nn.init.zeros_(self.proposal.bias)

    def support(
        self,
        features: torch.Tensor,
        phase: ValuePhase,
        generator: torch.Generator | None,
    ) -> ValueSupport:
        del phase, generator
        logits = self.proposal(features.detach()).float()
        masses = torch.softmax(logits, dim=-1)
        zero = torch.zeros_like(masses[..., :1])
        one = torch.ones_like(masses[..., :1])
        boundaries = torch.cat([zero, masses[..., :-1].cumsum(dim=-1), one], dim=-1)
        points = 0.5 * (boundaries[..., :-1] + boundaries[..., 1:])
        entropy = -(masses * masses.clamp_min(1e-8).log()).sum(dim=-1)
        return ValueSupport(points, masses, boundaries, entropy)

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
        return quantile_huber_loss(predictions.float(), targets.float(), support.points.detach())

    def auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.proposal.parameters())

    def auxiliary_loss(self, context: FractionLossContext) -> AuxiliaryLoss | None:
        boundaries = context.support.boundaries
        entropy = context.support.entropy
        if boundaries is None or entropy is None:
            raise ValueError("FQF auxiliary loss requires boundaries and entropy")
        internal = boundaries[..., 1:-1]
        boundary_values = context.boundary_values.detach().float()
        midpoint_values = context.midpoint_values.detach().float()
        gradient = 2.0 * boundary_values - midpoint_values[..., :-1] - midpoint_values[..., 1:]
        per_position = (internal * gradient).sum(dim=-1) - self.entropy_coefficient * entropy
        if context.valid is None:
            loss = per_position.mean()
        else:
            valid = context.valid.to(per_position.dtype)
            loss = (per_position * valid).sum() / valid.sum().clamp_min(1.0)
        masses = context.support.weights.detach()
        metrics = {
            "loss/fraction": loss.detach(),
            "fraction/entropy": entropy.detach().mean(),
            "fraction/effective_count": entropy.detach().exp().mean(),
            "fraction/min_mass": masses.amin(),
            "fraction/max_mass": masses.amax(),
            "fraction/wasserstein_gradient_mean": gradient.detach().abs().mean(),
            "fraction/wasserstein_gradient_max": gradient.detach().abs().amax(),
        }
        return AuxiliaryLoss(loss, metrics)
