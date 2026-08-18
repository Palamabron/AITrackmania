"""Optional stateful optimization utilities for learner experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GradientClipStats:
    norm: float
    ema_norm: float
    coefficient: float

    @property
    def clipped(self) -> bool:
        return self.coefficient < 1.0


class AdaptiveGradientClipper(nn.Module):
    """Clip gradient spikes relative to a checkpointable EMA of prior norms."""

    ema_norm: torch.Tensor
    step_count: torch.Tensor

    def __init__(
        self,
        decay: float = 0.995,
        warmup_steps: int = 100,
        clip_factor: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("gradient EMA decay must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("gradient clip warmup must be non-negative")
        if clip_factor <= 0.0:
            raise ValueError("gradient clip factor must be positive")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.clip_factor = clip_factor
        self.register_buffer("ema_norm", torch.tensor(float("nan"), dtype=torch.float64))
        self.register_buffer("step_count", torch.zeros((), dtype=torch.int64))

    def forward(self, parameters: Iterable[nn.Parameter]) -> GradientClipStats:
        trainable = [parameter for parameter in parameters if parameter.grad is not None]
        if not trainable:
            return GradientClipStats(norm=0.0, ema_norm=self._ema_value(), coefficient=1.0)
        norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
        current = float(norm.detach())
        self._update_ema(current)
        threshold = self._ema_value() * self.clip_factor
        coefficient = 1.0
        if int(self.step_count) > self.warmup_steps and current > threshold:
            coefficient = threshold / current
            for parameter in trainable:
                assert parameter.grad is not None
                parameter.grad.mul_(coefficient)
        return GradientClipStats(
            norm=current,
            ema_norm=self._ema_value(),
            coefficient=coefficient,
        )

    def _update_ema(self, current: float) -> None:
        if torch.isnan(self.ema_norm):
            self.ema_norm.fill_(current)
        else:
            self.ema_norm.mul_(self.decay).add_(current * (1.0 - self.decay))
        self.step_count.add_(1)

    def _ema_value(self) -> float:
        return 0.0 if torch.isnan(self.ema_norm) else float(self.ema_norm)
