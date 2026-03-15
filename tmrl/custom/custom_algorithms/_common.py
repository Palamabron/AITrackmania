"""Shared helpers and utilities for TMRL training agents."""

import numpy as np
import torch

import tmrl.config as cfg


def set_seed(seed: int = cfg.SEED) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed for NumPy and PyTorch. Defaults to config SEED.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _amp_enabled(device: str | None) -> bool:
    """Return True if mixed-precision (AMP) is enabled and device supports it."""
    use_mp = bool(cfg.ALG_CONFIG.get("MIXED_PRECISION", False))
    return use_mp and torch.cuda.is_available() and str(device).startswith("cuda")


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


def _amp_dtype() -> torch.dtype:
    """Return torch dtype for mixed precision (bfloat16 or float16)."""
    return (
        torch.bfloat16
        if str(cfg.ALG_CONFIG.get("MIXED_PRECISION_DTYPE", "float16")).lower() == "bfloat16"
        else torch.float16
    )


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
    # Shift continue_mask right by one so the terminal step's own reward is included:
    # a step's reward should count if the episode was alive *before* that step.
    ones = torch.ones((batch_size, 1), device=rewards.device, dtype=rewards.dtype)
    reward_mask = torch.cat([ones, continue_mask[:, :-1]], dim=1) * valid.to(rewards.dtype)
    n_step_returns = (reward_windows * discounted.unsqueeze(0) * reward_mask).sum(dim=1)
    bootstrap_not_done = continue_mask[:, n_steps - 1]
    # Enforce (batch, 1) for safe broadcasting with quantile matrices in TQC backup
    return n_step_returns.unsqueeze(-1), bootstrap_not_done.unsqueeze(-1)
