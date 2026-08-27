"""Numerical helpers shared by value-distribution strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from trackmaniarl.models.contracts import RiskDistortion, RiskSpec, ValueSupport


class SupportSampling(StrEnum):
    MIDPOINTS = "midpoints"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class UniformSupportSpec:
    count: int
    sampling: SupportSampling
    generator: torch.Generator | None = None


def quantile_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    fractions: torch.Tensor,
) -> torch.Tensor:
    """Return one quantile-Huber loss per leading feature position."""

    delta = targets.unsqueeze(-2) - predictions.unsqueeze(-1)
    absolute = delta.abs()
    huber = torch.where(absolute <= 1.0, 0.5 * delta.square(), absolute - 0.5)
    weights = torch.abs(fractions.unsqueeze(-1) - (delta.detach() < 0).float())
    return (weights * huber).mean(dim=(-2, -1))


def weighted_expectation(
    values: torch.Tensor,
    support: ValueSupport,
    risk: RiskSpec,
) -> torch.Tensor:
    weights = support.weights
    if risk.distortion is RiskDistortion.UPPER_CVAR:
        if support.boundaries is None:
            raise ValueError("upper CVaR requires quantile interval boundaries")
        lower = 1.0 - risk.alpha
        left = support.boundaries[..., :-1]
        right = support.boundaries[..., 1:]
        weights = (right - torch.maximum(left, torch.full_like(left, lower))).clamp_min(0.0)
        weights = weights / risk.alpha
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=-2)


def uniform_support(
    features: torch.Tensor,
    spec: UniformSupportSpec,
) -> ValueSupport:
    leading = features.shape[:-1]
    dtype = torch.float32
    device = features.device
    boundaries = torch.linspace(0.0, 1.0, spec.count + 1, device=device, dtype=dtype)
    boundaries = boundaries.expand(*leading, -1)
    if spec.sampling is SupportSampling.RANDOM:
        points = torch.rand(
            *leading, spec.count, device=device, dtype=dtype, generator=spec.generator
        )
    else:
        points = 0.5 * (boundaries[..., :-1] + boundaries[..., 1:])
    weights = torch.full((*leading, spec.count), 1.0 / spec.count, device=device, dtype=dtype)
    return ValueSupport(points=points, weights=weights, boundaries=boundaries)
