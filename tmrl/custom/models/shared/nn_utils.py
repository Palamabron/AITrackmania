"""Constants, basic NN utilities, obs-space helpers, and conv helpers."""

from math import floor
from typing import cast  # noqa: F401 — kept for potential downstream use

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 — kept for potential downstream use
from torch.nn import Conv2d

from tmrl.util import prod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -20  # Expanded for high-variance exploration (SAC literature)
EPSILON = 1e-7

# Alias so effnet.py and backbones.py can import it from here
SiLU = nn.SiLU

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
        return int(prod(observation_space.shape))


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
        return int(prod(observation_space.shape))
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
# EfficientNet helper — logically grouped here; imported by effnet.py
# ---------------------------------------------------------------------------


def _make_divisible(v, divisor, min_value=None):
    """Round v to nearest multiple of divisor (allow at most 10% reduction)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
