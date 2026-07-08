"""Shared helpers and utilities for TMRL training agents."""

import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import SGD, Adam, AdamW


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed for NumPy, PyTorch, and Python stdlib ``random``.
            Must be supplied explicitly; use ``MAIN_CONFIG.environment.seed``
            (or equivalent) at the call site so seeding is driven by the
            validated config rather than a hard-coded default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _amp_enabled(device: str | None, mixed_precision: bool) -> bool:
    """Return True if mixed-precision (AMP) is enabled and device supports it."""
    return mixed_precision and torch.cuda.is_available() and str(device).startswith("cuda")


def _tensor_to_scalar(value: torch.Tensor | float) -> float:
    """Convert a tensor or scalar to a Python float.

    Used in train() return dicts to avoid GPU sync in training_offline.

    Args:
        value: A single-element tensor, multi-element tensor (returns mean), or float.

    Returns:
        Python float.
    """
    if isinstance(value, torch.Tensor):
        t = value
        return t.item() if t.numel() == 1 else float(t.mean().item())
    return float(value)


def _amp_dtype(mixed_precision_dtype: str = "bfloat16") -> torch.dtype:
    """Return torch dtype for mixed precision (bfloat16 or float16)."""
    return torch.bfloat16 if mixed_precision_dtype.lower() == "bfloat16" else torch.float16


def amp_setup(
    device: str | None, mixed_precision: bool, mixed_precision_dtype: str
) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    """One-call AMP setup from config fields.

    Returns:
        Tuple of (use_amp, amp_dtype, grad_scaler).  When *mixed_precision* is
        False or the device is CPU the scaler is created in disabled mode and
        ``use_amp`` is False so callers can branch cheaply.
    """
    use_amp = _amp_enabled(device, mixed_precision)
    dtype = _amp_dtype(mixed_precision_dtype)
    use_scaler = use_amp and dtype != torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    return use_amp, dtype, scaler


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with 0 in a floating-point tensor; pass-through for integer tensors."""
    if t.is_floating_point():
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def sanitize_obs(obs: torch.Tensor | tuple) -> torch.Tensor | tuple:
    """Sanitize observation (tuple of tensors or single tensor): NaN/Inf -> 0."""
    if isinstance(obs, torch.Tensor):
        return sanitize_tensor(obs)
    return tuple(sanitize_tensor(t) if isinstance(t, torch.Tensor) else t for t in obs)


def _compute_n_step_return_and_bootstrap_mask(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float, n_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute n-step returns and bootstrap mask for 1D reward/done tensors.

    Implements "break before adding reward at done step" semantics: rewards
    after a done step are not included in the n-step return.

    Args:
        rewards: 1D tensor of rewards (will be flattened).
        dones: 1D tensor of done flags (will be flattened).
        gamma: Discount factor.
        n_steps: Number of steps for n-step return.

    Returns:
        Tuple of (n_step_returns, bootstrap_mask). bootstrap_mask is 1 where
        the transition is not done and we bootstrap from the value function.
        Both tensors are shaped (batch, 1) for safe broadcasting with quantile
        matrices in distributional RL algorithms like TQC/IQN.

    Note:
        The continue_mask is shifted right by one position so the terminal step's
        own reward is included in the return. A step's reward should count if the
        episode was alive *before* that step.
    """
    rewards = rewards.reshape(-1)
    dones = dones.reshape(-1)
    batch_size = rewards.shape[0]

    if n_steps <= 1:
        ones = torch.ones_like(rewards)
        return rewards, ones

    step_offsets = torch.arange(n_steps, device=rewards.device)
    start_indices = torch.arange(batch_size, device=rewards.device).unsqueeze(1)
    all_indices = start_indices + step_offsets.unsqueeze(0)
    valid = all_indices < batch_size
    clamped_indices = all_indices.clamp(max=batch_size - 1)

    reward_windows = rewards[clamped_indices]
    done_windows = dones[clamped_indices]
    not_done_windows = (done_windows != 1.0).to(rewards.dtype)
    continue_mask = torch.cumprod(not_done_windows, dim=1) * valid.to(rewards.dtype)
    discounted = torch.pow(
        torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype), step_offsets
    )
    ones = torch.ones((batch_size, 1), device=rewards.device, dtype=rewards.dtype)
    reward_mask = torch.cat([ones, continue_mask[:, :-1]], dim=1) * valid.to(rewards.dtype)
    n_step_returns = (reward_windows * discounted.unsqueeze(0) * reward_mask).sum(dim=1)
    bootstrap_not_done = continue_mask[:, n_steps - 1]
    return n_step_returns.unsqueeze(-1), bootstrap_not_done.unsqueeze(-1)


def clip_model_weights(model: torch.nn.Module, max_value: float = 1.0) -> None:
    """Clip all model parameters to [-max_value, max_value].

    Args:
        model: Model whose parameters to clip.
        max_value: Maximum absolute value for parameters.
    """
    for param in model.parameters():
        param.data.clamp_(-max_value, max_value)


def autocast_context(
    use_mixed_precision: bool, amp_dtype: torch.dtype
) -> torch.autocast | nullcontext:
    """Return an autocast context manager if mixed precision is enabled.

    Args:
        use_mixed_precision: Whether to enable automatic mixed precision.
        amp_dtype: Data type for AMP (e.g., torch.float16, torch.bfloat16).

    Returns:
        Autocast context manager if enabled, otherwise nullcontext (no-op).
    """
    if use_mixed_precision:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    return nullcontext()


def polyak_update(model: torch.nn.Module, model_target: torch.nn.Module, polyak: float) -> None:
    """Polyak-average (soft-update) target network parameters.

    Args:
        model: Source model (online network).
        model_target: Target model to update.
        polyak: Polyak averaging coefficient (0-1). Target = polyak * target + (1-polyak) * source.
    """
    with torch.no_grad():
        for p, p_targ in zip(model.parameters(), model_target.parameters(), strict=True):
            p_targ.data.mul_(polyak).add_(p.data, alpha=(1 - polyak))


def _make_optimizer(
    params,
    optimizer_name: str,
    lr: float,
    *,
    weight_decay: float | None = None,
    eps: float | None = None,
    betas: tuple[float, ...] | None = None,
) -> torch.optim.Optimizer:
    """Build an Adam/AdamW/SGD optimizer from a name string and optional kwargs.

    Args:
        params: Iterable of parameters to optimize.
        optimizer_name: One of "adam", "adamw", "sgd" (case-insensitive).
            Unknown names log a warning and fall back to SGD.
        lr: Learning rate.
        weight_decay: L2 regularisation. Omitted when None.
        eps: Adam epsilon. Ignored for SGD and when None.
        betas: Adam beta coefficients. Ignored for SGD and when None.

    Returns:
        Configured optimizer instance.
    """
    name = optimizer_name.lower()
    if name == "adam":
        cls: type[Adam] | type[AdamW] | type[SGD] = Adam
    elif name == "adamw":
        cls = AdamW
    else:
        if name != "sgd":
            logger.warning("Unknown optimizer '{}', defaulting to SGD", name)
        cls = SGD
    kwargs: dict[str, Any] = {"lr": lr}
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay
    if eps is not None and name in ("adam", "adamw"):
        kwargs["eps"] = eps
    if betas is not None and name in ("adam", "adamw"):
        kwargs["betas"] = tuple(betas)
    return cls(params, **kwargs)


def project_simbav2_weights(model: torch.nn.Module) -> None:
    """Re-project HypersphericalLinear weights after an optimizer step (SimbaV2).

    Args:
        model: Model containing SimbaV2Backbone modules to project.

    Note:
        SimbaV2 uses hyperspherical linear layers that must be re-projected
        to the unit sphere after gradient updates to maintain constraints.
    """
    from tmrl.custom.models.shared.blocks import SimbaV2Backbone

    for m in model.modules():
        if isinstance(m, SimbaV2Backbone):
            m.project_weights()
