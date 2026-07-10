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
        """Initialize the residual block.

        Args:
            dim: Width of the block; input and output dimensions are equal.
            scale: Residual scaling factor applied to the block output before
                adding back to the skip connection.  Set to ``1/√num_blocks``
                by ``residual_mlp_backbone`` to keep activation magnitudes stable
                in deep stacks.
        """
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = SiLU()
        self.scale = scale
        self._init_weights()

    def _init_weights(self):
        """Initialize block weights.

        ``linear1`` uses orthogonal init (gain=1) for stable gradient flow.
        ``linear2`` is zero-initialized so each block starts as near-identity,
        which aids convergence of deep residual networks.
        """
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-activation residual transformation.

        Args:
            x: Input tensor of shape ``(B, dim)``.

        Returns:
            Output tensor of shape ``(B, dim)``.
        """
        out = self.act(self.ln1(self.linear1(x)))
        out = self.ln2(self.linear2(out))
        return cast(torch.Tensor, x + self.scale * self.act(out))


def residual_mlp_backbone(input_dim: int, hidden_dim: int, num_blocks: int) -> nn.Module:
    """Build a linear input projection followed by stacked residual MLP blocks.

    The per-block scale is set to ``1/√num_blocks`` so that residual
    contributions shrink proportionally with depth, keeping feature magnitudes
    roughly constant throughout the stack.

    Args:
        input_dim: Dimensionality of the input feature vector.
        hidden_dim: Width of the hidden layers and the output.
        num_blocks: Number of ``ResidualMLPBlock`` layers in the stack.

    Returns:
        ``nn.Sequential`` of Linear -> LayerNorm -> SiLU -> [ResidualMLPBlock x num_blocks].
        Output shape: ``(B, hidden_dim)``.
    """
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
    """Normalize a tensor to unit L2 norm along a given dimension.

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to normalize. Defaults to -1 (last dim).
        eps: Small constant added to the norm to prevent division by zero.

    Returns:
        Tensor with the same shape as ``x``, with unit L2 norm along ``dim``.
    """
    return cast(torch.Tensor, x / (x.norm(dim=dim, keepdim=True) + eps))


class HypersphericalLinear(nn.Module):
    """Linear layer with unit-norm weight rows and a learnable per-output scaler.

    Call ``project_weights()`` after each optimizer step to re-normalise rows.
    """

    def __init__(self, in_features: int, out_features: int):
        """Initialize the layer and project weights onto the unit hypersphere.

        Args:
            in_features: Input dimensionality.
            out_features: Output dimensionality (number of unit-norm weight rows).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scaler = nn.Parameter(torch.ones(out_features))
        nn.init.orthogonal_(self.weight)
        self.project_weights()

    @torch.no_grad()
    def project_weights(self):
        """Re-normalize weight rows to unit L2 norm.

        Must be called after each optimizer step to maintain the hyperspherical
        constraint on the weight matrix.  Decorated with ``@torch.no_grad()``
        to avoid accumulating gradient history during projection.
        """
        self.weight.copy_(_l2_normalize(self.weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a scaled cosine-similarity projection.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Tensor of shape ``(..., out_features)`` — dot product with unit-norm
            rows, scaled element-wise by the learnable ``scaler`` parameter.
        """
        return nn.functional.linear(x, self.weight) * self.scaler


class SimbaV2Block(nn.Module):
    """SimbaV2 residual block: inverted-bottleneck MLP + LERP + L2-norm (paper Eq. 11-12)."""

    def __init__(self, dim: int, expand_ratio: int = 4):
        """Initialize the SimbaV2 residual block.

        Args:
            dim: Dimensionality of the block (input, inter-block, and output).
            expand_ratio: Width multiplier for the inner bottleneck dimension.
                Defaults to 4 (``inner_dim = dim * expand_ratio``).
        """
        super().__init__()
        inner_dim = dim * expand_ratio
        self.up_proj = HypersphericalLinear(dim, inner_dim)
        self.down_proj = HypersphericalLinear(inner_dim, dim)
        self.alpha = nn.Parameter(torch.full((dim,), 0.5))

    def project_weights(self):
        """Re-normalize hyperspherical weight rows in both projection layers.

        Delegates to ``up_proj.project_weights()`` and
        ``down_proj.project_weights()``.  Must be called after every optimizer
        step.
        """
        self.up_proj.project_weights()
        self.down_proj.project_weights()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the SimbaV2 inverted-bottleneck residual step (paper Eq. 11-12).

        Args:
            h: Input tensor of shape ``(B, dim)``, assumed to lie on the unit
                hypersphere (output of a prior SimbaV2Block or input projection).

        Returns:
            Tensor of shape ``(B, dim)`` on the unit hypersphere.
        """
        h_tilde = _l2_normalize(self.down_proj(torch.relu(self.up_proj(h))))
        alpha = self.alpha.sigmoid()
        return _l2_normalize((1.0 - alpha) * h + alpha * h_tilde)


class SimbaV2Backbone(nn.Module):
    """SimbaV2 backbone: shift + L2-norm input → hyperspherical residual blocks.

    Call ``project_weights()`` after each optimizer step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, c_shift: float = 1.0):
        """Initialize the SimbaV2 backbone.

        Args:
            input_dim: Dimensionality of the raw input feature vector.
            hidden_dim: Width of the hidden representation (and output).
            num_blocks: Number of ``SimbaV2Block`` layers.
            c_shift: Constant appended to the input before L2-normalization.
                Shifts the origin so the zero vector maps to a well-defined
                point on the hypersphere rather than being undefined after
                normalization.
        """
        super().__init__()
        self.c_shift = c_shift
        self.input_proj = HypersphericalLinear(input_dim + 1, hidden_dim)
        self.blocks = nn.ModuleList([SimbaV2Block(hidden_dim) for _ in range(num_blocks)])

    @torch.no_grad()
    def project_weights(self):
        """Re-normalize all hyperspherical weight rows throughout the backbone.

        Delegates to ``input_proj`` and each ``SimbaV2Block``.  Must be called
        after every optimizer step.
        """
        self.input_proj.project_weights()
        for block in self.blocks:
            block.project_weights()  # type: ignore[operator]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through the SimbaV2 backbone.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Feature tensor of shape ``(B, hidden_dim)`` on the unit hypersphere.
        """
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
    """Build a SimbaV2 backbone.

    A convenience factory around ``SimbaV2Backbone``.  After each optimizer
    step, call ``backbone.project_weights()`` to maintain the hyperspherical
    weight constraint.

    Args:
        input_dim: Dimensionality of the raw input feature vector.
        hidden_dim: Width of the hidden representation (and output).
        num_blocks: Number of ``SimbaV2Block`` residual layers.
        c_shift: Constant appended before L2-normalization of the input.
            Defaults to 1.0.

    Returns:
        A ``SimbaV2Backbone`` instance.  Output dim = ``hidden_dim`` (features
        lie on the unit hypersphere).
    """
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
