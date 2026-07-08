"""ResidualMLP, SimbaV2, and squashed_logprob."""

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tmrl.custom.models.shared.nn_utils import SiLU

# ---------------------------------------------------------------------------
# Residual MLP blocks
# ---------------------------------------------------------------------------


class ResidualMLPBlock(nn.Module):
    """Pre-activation residual block: Linear→LN→SiLU→Linear→LN, out = x + scale·block(x).

    Scale = 1/√num_blocks prevents activation growth in deep stacks.
    Second linear is zero-initialised so each block starts as near-identity.
    """

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = SiLU()
        self.scale = scale
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.ln1(self.linear1(x)))
        out = self.ln2(self.linear2(out))
        return cast(torch.Tensor, x + self.scale * self.act(out))


def residual_mlp_backbone(input_dim: int, hidden_dim: int, num_blocks: int) -> nn.Module:
    """Build: input_proj -> num_blocks ResidualMLPBlock layers. Output dim = hidden_dim."""
    scale = 1.0 / max(1, num_blocks) ** 0.5
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        SiLU(),
    ]
    for _ in range(num_blocks):
        layers.append(ResidualMLPBlock(hidden_dim, scale=scale))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# SimbaV2 blocks
# ---------------------------------------------------------------------------


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return cast(torch.Tensor, x / (x.norm(dim=dim, keepdim=True) + eps))


class HypersphericalLinear(nn.Module):
    """Linear layer with unit-norm weight rows and a learnable per-output scaler.

    Call ``project_weights()`` after each optimizer step to re-normalise rows.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scaler = nn.Parameter(torch.ones(out_features))
        nn.init.orthogonal_(self.weight)
        self.project_weights()

    @torch.no_grad()
    def project_weights(self):
        self.weight.copy_(_l2_normalize(self.weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight) * self.scaler


class SimbaV2Block(nn.Module):
    """SimbaV2 residual block: inverted-bottleneck MLP + LERP + L2-norm (paper Eq. 11-12)."""

    def __init__(self, dim: int, expand_ratio: int = 4):
        super().__init__()
        inner_dim = dim * expand_ratio
        self.up_proj = HypersphericalLinear(dim, inner_dim)
        self.down_proj = HypersphericalLinear(inner_dim, dim)
        self.alpha = nn.Parameter(torch.full((dim,), 0.5))

    def project_weights(self):
        self.up_proj.project_weights()
        self.down_proj.project_weights()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_tilde = _l2_normalize(self.down_proj(torch.relu(self.up_proj(h))))
        alpha = self.alpha.sigmoid()
        return _l2_normalize((1.0 - alpha) * h + alpha * h_tilde)


class SimbaV2Backbone(nn.Module):
    """SimbaV2 backbone: shift + L2-norm input → hyperspherical residual blocks.

    Call ``project_weights()`` after each optimizer step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, c_shift: float = 1.0):
        super().__init__()
        self.c_shift = c_shift
        self.input_proj = HypersphericalLinear(input_dim + 1, hidden_dim)
        self.blocks = nn.ModuleList([SimbaV2Block(hidden_dim) for _ in range(num_blocks)])

    @torch.no_grad()
    def project_weights(self):
        self.input_proj.project_weights()
        for block in self.blocks:
            block.project_weights()  # type: ignore[operator]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.full((*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype)
        x = _l2_normalize(torch.cat([x, shift], dim=-1))
        x = _l2_normalize(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return x


def simba_v2_backbone(
    input_dim: int,
    hidden_dim: int,
    num_blocks: int,
    c_shift: float = 1.0,
) -> SimbaV2Backbone:
    """Build a SimbaV2 backbone. Output dim = hidden_dim (features on unit hypersphere)."""
    return SimbaV2Backbone(input_dim, hidden_dim, num_blocks, c_shift=c_shift)


# ---------------------------------------------------------------------------
# Squashed Gaussian log-prob (SAC appendix C)
# ---------------------------------------------------------------------------

_LOG2 = float(np.log(2.0))
_SQUASH_CLAMP = 20.0


def squashed_logprob(pi_distribution: Normal, pi_action: torch.Tensor) -> torch.Tensor:
    """Log-prob of a Gaussian policy corrected for tanh squashing (SAC appendix C).

    Numerically stable form:  logp -= 2*(log2 - a - softplus(-2*a))
    Pre-tanh action is clamped to +/-20 to prevent -inf / NaN at large magnitudes.
    """
    logp = pi_distribution.log_prob(pi_action).sum(axis=-1)
    a = pi_action.clamp(-_SQUASH_CLAMP, _SQUASH_CLAMP)
    logp -= (2.0 * (_LOG2 - a - F.softplus(-2.0 * a))).sum(dim=1)
    return cast(torch.Tensor, logp)
