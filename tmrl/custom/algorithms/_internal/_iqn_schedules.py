"""IQN epsilon schedules and quantile loss helper functions."""

import math

import torch
from einops import rearrange


def epsilon_cosine_schedule(
    step: float,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.005,
    t0: float = 50000.0,
    tmult: float = 1.5,
    decay: float = 0.8,
    initial_amplitude: float = 0.1,
    floor_frac: float = 0.0,
    floor_steps: int = 0,
    mode: str = "cosine",
) -> float:
    """Epsilon schedule for a given step (for plotting/debugging).

    Args:
        step: Current training step.
        epsilon_start: Initial epsilon value.
        epsilon_end: Minimum epsilon floor value.
        t0: Initial cycle length.
        tmult: Cycle length multiplier (>1 for expanding cycles).
        decay: Amplitude decay factor per cycle.
        initial_amplitude: Initial oscillation amplitude.
        floor_frac: Fraction of cycle spent at floor (0-1).
        floor_steps: Explicit floor duration in steps (overrides floor_frac if >0).
        mode: Schedule mode - "cosine" or "ramp".

    Returns:
        Epsilon value for the given step.

    Note:
        Mode options:
        - "cosine": Damped sinusoid (full wave per cycle, peak->trough->peak).
        - "ramp": Half-cosine (peak->trough per cycle, then floor).
    """
    min_eps = epsilon_end
    floor_frac = max(0.0, min(1.0, floor_frac))
    floor_steps = max(0, floor_steps)

    if step <= 0.0:
        return epsilon_start

    if tmult <= 1.0:
        cycle_num = int(step // t0)
        step_in_cycle = step - cycle_num * t0
        cycle_length = t0
    else:
        ratio = 1.0 + step * (tmult - 1.0) / t0
        cycle_num = int(math.log(ratio) / math.log(tmult)) if ratio > 1.0 else 0
        cum_start = t0 * (tmult**cycle_num - 1.0) / (tmult - 1.0) if cycle_num > 0 else 0.0
        step_in_cycle = step - cum_start
        cycle_length = t0 * (tmult**cycle_num)

    if floor_steps > 0:
        floor_duration = min(floor_steps, cycle_length)
    else:
        floor_duration = floor_frac * cycle_length
    cosine_length = max(1e-9, cycle_length - floor_duration)

    if step_in_cycle >= cosine_length:
        return min_eps

    if mode == "ramp":
        current_amplitude = max(0.0, epsilon_start - min_eps) * (decay**cycle_num)
        angle = math.pi * (step_in_cycle / cosine_length)
    else:
        current_amplitude = max(0.0, initial_amplitude) * (decay**cycle_num)
        phase = step_in_cycle / cosine_length
        angle = 2.0 * math.pi * phase

    return min_eps + 0.5 * current_amplitude * (1.0 + math.cos(angle))


def epsilon_linear_schedule(
    step: float,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.005,
    decay_steps: float = 500000.0,
) -> float:
    """Linear decay from epsilon_start to epsilon_end (floor) over decay_steps.

    Args:
        step: Current training step.
        epsilon_start: Initial epsilon value.
        epsilon_end: Final epsilon value (floor).
        decay_steps: Number of steps for full decay.

    Returns:
        Epsilon value for the given step.
    """
    if step <= 0.0:
        return epsilon_start
    if step >= decay_steps:
        return epsilon_end
    frac = step / decay_steps
    return epsilon_start + (epsilon_end - epsilon_start) * frac


def epsilon_cosine_anneal_schedule(
    step: float,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.005,
    decay_steps: float = 500000.0,
) -> float:
    """Cosine annealing (single period) from epsilon_start to epsilon_end over decay_steps.

    Args:
        step: Current training step.
        epsilon_start: Initial epsilon value.
        epsilon_end: Final epsilon value.
        decay_steps: Number of steps for full annealing.

    Returns:
        Epsilon value for the given step.
    """
    if step <= 0.0:
        return epsilon_start
    if step >= decay_steps:
        return epsilon_end
    frac = min(1.0, step / decay_steps)
    return epsilon_end + 0.5 * (epsilon_start - epsilon_end) * (1.0 + math.cos(math.pi * frac))


def _quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
    is_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile Huber loss for IQN.

    Args:
        current_quantiles: (batch, N_tau, n_actions_selected=1) or (batch, N_tau).
        target_quantiles:  (batch, N_tau_prime).
        tau: (batch, N_tau) quantile fractions for current.
        kappa: Huber threshold.
        is_weights: (batch,) importance sampling weights for PER (optional).

    Returns:
        Scalar loss (sum over N_tau_prime, mean over N_tau, then mean over batch).
    """
    if current_quantiles.dim() == 2:
        current_quantiles = current_quantiles.unsqueeze(-1)
    if target_quantiles.dim() == 2:
        target_quantiles = target_quantiles.unsqueeze(1)

    delta = target_quantiles - current_quantiles
    abs_delta = delta.abs()

    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )

    tau_expanded = rearrange(tau, "b n -> b n 1")
    weight = torch.abs(tau_expanded - (delta.detach() < 0).float())
    # Sum over N_tau_prime (target), then mean over N_tau (current) — Dabney et al. 2018.
    per_sample_loss = (weight * huber).sum(dim=-1).mean(dim=-1)

    if is_weights is not None:
        # IS weights are already normalized in training_offline.py, so apply directly
        per_sample_loss = per_sample_loss * is_weights.squeeze()

    return per_sample_loss.mean()


def _munchausen_bonus_from_q(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    tau: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Compute clipped Munchausen log-policy bonus for selected actions.

    Args:
        q_values: Q-values for all actions (batch, n_actions).
        actions: Selected actions (batch,).
        tau: Temperature parameter for policy extraction.
        clip_min: Minimum clip value for log-policy.
        clip_max: Maximum clip value for log-policy.

    Returns:
        Clipped log-policy bonus for selected actions (batch,).
    """
    logits = q_values / tau
    log_policy = torch.log_softmax(logits, dim=-1)
    log_pi_a = log_policy.gather(1, actions.view(-1, 1)).squeeze(1)
    return log_pi_a.clamp(min=clip_min, max=clip_max)
