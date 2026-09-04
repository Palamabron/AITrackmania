"""Reusable neural-network backbones for reinforcement-learning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Unpack, cast

import torch
from torch import nn
from torch.nn import functional as F


def _unit_norm(value: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return F.normalize(value, dim=dim, eps=1e-8)


class HypersphericalLinear(nn.Module):
    """Linear projection whose effective weight rows stay on the unit sphere."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("hyperspherical dimensions must be positive")
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.scale = nn.Parameter(torch.ones(output_dim))
        nn.init.orthogonal_(self.weight)
        self.project_weights_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = _unit_norm(self.weight, dim=1)
        return F.linear(value, weight) * self.scale

    @torch.no_grad()
    def project_weights_(self) -> None:
        self.weight.copy_(_unit_norm(self.weight, dim=1))


class SimbaV2Block(nn.Module):
    """Hyperspherical inverted-bottleneck residual block."""

    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0 or expansion <= 0:
            raise ValueError("SimbaV2 block dimensions must be positive")
        expanded_dim = hidden_dim * expansion
        self.expand = HypersphericalLinear(hidden_dim, expanded_dim)
        self.project = HypersphericalLinear(expanded_dim, hidden_dim)
        self.mix_logit = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        transformed = self.project(F.relu(self.expand(value)))
        mix = self.mix_logit.sigmoid()
        return _unit_norm(torch.lerp(value, transformed, mix))

    def project_weights_(self) -> None:
        self.expand.project_weights_()
        self.project.project_weights_()


class _SimbaKwargs(TypedDict, total=False):
    block_count: int
    expansion: int
    input_shift: float


@dataclass(frozen=True, slots=True)
class SimbaV2Options:
    block_count: int = 2
    expansion: int = 4
    input_shift: float = 1.0


class SimbaV2Backbone(nn.Module):
    """SimbaV2-style backbone with normalized inputs, weights and features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        **kwargs: Unpack[_SimbaKwargs],
    ) -> None:
        super().__init__()
        options = SimbaV2Options(**kwargs)
        if input_dim <= 0 or hidden_dim <= 0 or options.block_count < 0:
            raise ValueError("SimbaV2 backbone dimensions are invalid")
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.input_shift = options.input_shift
        self.input_projection = HypersphericalLinear(input_dim + 1, hidden_dim)
        self.blocks = nn.ModuleList(
            SimbaV2Block(hidden_dim, options.expansion) for _ in range(options.block_count)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.input_dim:
            raise ValueError(f"expected final input dimension {self.input_dim}")
        shift = value.new_full((*value.shape[:-1], 1), self.input_shift)
        hidden = _unit_norm(torch.cat((value, shift), dim=-1))
        hidden = _unit_norm(self.input_projection(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def project_weights_(self) -> None:
        self.input_projection.project_weights_()
        for block in self.blocks:
            cast(SimbaV2Block, block).project_weights_()


def project_hyperspherical_weights(module: nn.Module) -> None:
    """Re-project every hyperspherical layer after an optimizer update."""

    for child in module.modules():
        if isinstance(child, HypersphericalLinear):
            child.project_weights_()
