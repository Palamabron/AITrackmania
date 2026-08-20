"""Target transforms shared by discrete value representations."""

from __future__ import annotations

import torch

_VALUE_RESCALING_EPSILON = 1e-3


def rescale_value(value: torch.Tensor) -> torch.Tensor:
    return value.sign() * ((value.abs() + 1.0).sqrt() - 1.0) + _VALUE_RESCALING_EPSILON * value


def inverse_rescale_value(value: torch.Tensor) -> torch.Tensor:
    epsilon = _VALUE_RESCALING_EPSILON
    inner = (1.0 + 4.0 * epsilon * (value.abs() + 1.0 + epsilon)).sqrt() - 1.0
    return value.sign() * ((inner / (2.0 * epsilon)).square() - 1.0)


def bootstrap_target(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    target_values: torch.Tensor,
    *,
    rescale: bool,
) -> torch.Tensor:
    if not rescale:
        return rewards.unsqueeze(-1) + discounts.unsqueeze(-1) * target_values
    unscaled = rewards.unsqueeze(-1) + discounts.unsqueeze(-1) * inverse_rescale_value(
        target_values
    )
    return rescale_value(unscaled)
