"""Shared neural network building blocks.

Single source of truth for all reusable NN components:
- Basic utilities (mlp, combined_shape, count_vars)
- Gym obs-space helpers (_obs_dim, _cat_obs, …)
- Conv dimension helpers (num_flat_features, conv2d_out_dims)
- EfficientNetV2 architecture and factory functions
- FrozenEfficientNetEncoder
- Residual MLP blocks
- SimbaV2 blocks
- squashed_logprob: SAC appendix-C log-prob correction
"""

from math import floor, sqrt
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Conv2d

from tmrl.util import prod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -20  # Expanded for high-variance exploration (SAC literature)
EPSILON = 1e-7

# ---------------------------------------------------------------------------
# Basic NN utilities
# ---------------------------------------------------------------------------


def combined_shape(length, shape=None):
    """Return shape tuple combining length with an optional inner shape."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create an MLP as nn.Sequential from a list of layer sizes."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module: nn.Module) -> int:
    """Total number of trainable parameters in a module."""
    return sum(int(np.prod(p.shape)) for p in module.parameters())


# ---------------------------------------------------------------------------
# Obs-space utilities
# ---------------------------------------------------------------------------


def obs_dim(observation_space) -> int:
    """Total flat dimension of an observation space (tuple of spaces or Box)."""
    try:
        return sum(prod(s for s in space.shape) for space in observation_space)
    except TypeError:
        return prod(observation_space.shape)


def cat_obs(obs, tuple_obs: bool) -> torch.Tensor:
    """Concatenate obs tensors along last dim; flatten for single-tensor obs."""
    if tuple_obs:
        return torch.cat(obs, -1)
    return torch.flatten(obs, start_dim=1)


def ensure_float(t: torch.Tensor) -> torch.Tensor:
    """Cast to float32 only when needed (avoids redundant copies for fp16/bf16)."""
    if t.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return t
    return t.float()


def obs_spaces_list(observation_space) -> list:
    """Convert an observation space to a flat list of sub-spaces."""
    if hasattr(observation_space, "spaces"):
        return list(observation_space.spaces)
    return list(observation_space)


def vector_dim_except(observation_space, image_index: int) -> int:
    """Flat dimension of all obs components except the one at *image_index*."""
    try:
        spaces = list(observation_space)
    except TypeError:
        return prod(observation_space.shape)
    return sum(
        prod(space.shape) if hasattr(space, "shape") else int(space)
        for i, space in enumerate(spaces)
        if i != image_index
    )


def cat_obs_except_image(obs, image_index: int) -> torch.Tensor:
    """Concatenate all obs tensors except ``obs[image_index]`` on the last dim."""
    parts = [obs[i] for i in range(len(obs)) if i != image_index]
    if any(t.dim() > 2 for t in parts):
        parts = [t.view(t.size(0), -1) for t in parts]
    return ensure_float(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# Conv dimension helpers
# ---------------------------------------------------------------------------


def num_flat_features(x: torch.Tensor) -> int:
    """Number of flat features in a tensor (excluding the batch dimension)."""
    num_features = 1
    for s in x.size()[1:]:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer: Conv2d, h_in: int, w_in: int) -> tuple[int, int]:
    """Output (H, W) of a Conv2d layer given input spatial dimensions."""

    def _get(attr, idx):
        v = getattr(conv_layer, attr)
        return int(v[idx]) if isinstance(v, tuple) else int(v)

    pad_h, pad_w = _get("padding", 0), _get("padding", 1)
    dil_h, dil_w = _get("dilation", 0), _get("dilation", 1)
    str_h, str_w = _get("stride", 0), _get("stride", 1)
    ker_h, ker_w = _get("kernel_size", 0), _get("kernel_size", 1)

    h_out = floor((h_in + 2 * pad_h - dil_h * (ker_h - 1) - 1) / str_h + 1)
    w_out = floor((w_in + 2 * pad_w - dil_w * (ker_w - 1) - 1) / str_w + 1)
    return h_out, w_out


# ---------------------------------------------------------------------------
# EfficientNet building blocks
# ---------------------------------------------------------------------------


def _make_divisible(v, divisor, min_value=None):
    """Round v to nearest multiple of divisor (allow at most 10% reduction)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


SiLU = nn.SiLU


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention layer."""

    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution block (with optional SE)."""

    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


def conv_3x3_bn(inp, oup, stride):
    """3×3 conv + BN + SiLU."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


def conv_dw_3x3_bn(inp, oup, stride):
    """Depthwise 3×3 + pointwise 1×1 stem (fewer FLOPs than conv_3x3_bn)."""
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        SiLU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


def conv_1x1_bn(inp, oup):
    """1×1 conv + BN + SiLU."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


# ---------------------------------------------------------------------------
# EfficientNetV2
# ---------------------------------------------------------------------------


class EffNetV2(nn.Module):
    """EfficientNetV2-style CNN.

    Args:
        cfgs: Block config list — each entry is [expand_ratio, channels, num_blocks, stride, use_se].
        nb_channels_in: Input channels (default 3 for RGB).
        dim_output: Output embedding dimension.
        width_mult: Channel width multiplier.
        use_dw_stem: Use depthwise-separable first conv (faster, slightly fewer params).
    """

    def __init__(
        self,
        cfgs,
        nb_channels_in: int = 3,
        dim_output: int = 1,
        width_mult: float = 1.0,
        use_dw_stem: bool = False,
    ):
        super().__init__()
        self.cfgs = cfgs
        input_channel = _make_divisible(24 * width_mult, 8)
        stem = (
            conv_dw_3x3_bn(nb_channels_in, input_channel, 2)
            if use_dw_stem
            else conv_3x3_bn(nb_channels_in, input_channel, 2)
        )
        layers = [stem]
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(MBConv(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, dim_output)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_xs(**kwargs) -> EffNetV2:
    """EfficientNetV2-XS: 8 blocks, ~5× faster than S."""
    cfgs = [
        [1, 16, 1, 1, 0],
        [4, 32, 2, 2, 0],
        [4, 48, 2, 2, 0],
        [4, 96, 3, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_s(**kwargs) -> EffNetV2:
    """EfficientNetV2-S."""
    cfgs = [
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs) -> EffNetV2:
    """EfficientNetV2-M."""
    cfgs = [
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs) -> EffNetV2:
    """EfficientNetV2-L."""
    cfgs = [
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs) -> EffNetV2:
    """EfficientNetV2-XL."""
    cfgs = [
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


_EFFNET_VARIANTS = {
    "xs": effnetv2_xs,
    "s": effnetv2_s,
}


class FrozenEfficientNetEncoder(nn.Module):
    """EfficientNetV2 image encoder — frozen (default) or trainable.

    frozen=True: all parameters have requires_grad=False; encoder stays in train
    mode so BatchNorm uses batch statistics (running stats are never calibrated
    when the encoder starts from random init).
    frozen=False: trains end-to-end with the rest of the model.

    Supported variants: "xs" (8 blocks, fast), "s" (40 blocks).
    use_dw_stem=True: depthwise-separable first conv.
    """

    def __init__(
        self,
        nb_channels_in: int = 4,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        frozen: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self._frozen = frozen
        factory = _EFFNET_VARIANTS.get(variant)
        if factory is None:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}', choose from {list(_EFFNET_VARIANTS)}"
            )
        self._encoder = factory(
            nb_channels_in=nb_channels_in,
            dim_output=embed_dim,
            width_mult=width_mult,
            use_dw_stem=use_dw_stem,
        )
        if frozen:
            for p in self._encoder.parameters():
                p.requires_grad = False
            self._encoder.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self._encoder.train()
        else:
            self._encoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.5:
            x = x / 255.0
        if self._frozen:
            with torch.no_grad():
                return cast(torch.Tensor, self._encoder(x))
        return cast(torch.Tensor, self._encoder(x))


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
    """Build: input_proj → num_blocks × ResidualMLPBlock. Output dim = hidden_dim."""
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
    return x / (x.norm(dim=dim, keepdim=True) + eps)


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

    Numerically stable form:  logp -= 2·(log2 − a − softplus(−2a))
    Pre-tanh action is clamped to ±20 to prevent −inf / NaN at large magnitudes.
    """
    logp = pi_distribution.log_prob(pi_action).sum(axis=-1)
    a = pi_action.clamp(-_SQUASH_CLAMP, _SQUASH_CLAMP)
    logp -= (2.0 * (_LOG2 - a - F.softplus(-2.0 * a))).sum(dim=1)
    return logp
