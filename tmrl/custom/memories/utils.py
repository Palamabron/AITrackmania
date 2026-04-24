"""Replay memory utilities: FoG resampling and horizontal-flip action helpers."""

from __future__ import annotations

import numpy as np

# Steering is at index 2 in [gas, brake, steer] (control_gamepad / send_control).
ACTION_STEER_INDEX = 2
_DISCRETE_STEER_BINS: int | None = None


def configure_discrete_steer_bins(n_steer: int) -> None:
    """Set runtime steering-bin count used for discrete action mirroring."""
    n_steer_int = int(n_steer)
    _set_discrete_steer_bins(n_steer_int if n_steer_int > 0 else None)


def _set_discrete_steer_bins(n_steer: int | None) -> None:
    global _DISCRETE_STEER_BINS
    _DISCRETE_STEER_BINS = n_steer


def fog_recency_resample(
    indices: tuple[int, ...] | list[int],
    buffer_len: int,
    decay_temperature: float = 3.0,
) -> tuple[int, ...]:
    """Forget-and-Grow (FoG) replay decay: resample *indices* with exponential recency bias.

    Each index *i* (representing its position in the circular buffer) is assigned a
    sampling weight  ``exp(decay_temperature * i / buffer_len)``.  More recent
    transitions (higher *i*) therefore receive exponentially higher probability,
    discouraging the agent from anchoring on the earliest, low-quality experiences
    ("primacy bias").

    Args:
        indices: Raw indices produced by the base ``sample_indices()`` call.
        buffer_len: Current number of valid items in the replay buffer.
        decay_temperature: Controls the steepness of the recency curve.
            Higher values forget old data more aggressively.  0 = uniform (no FoG).

    Returns:
        A tuple of resampled indices with the same length as *indices*.
    """
    if decay_temperature <= 0.0 or buffer_len <= 1 or len(indices) == 0:
        return tuple(indices)

    idx_arr = np.asarray(indices, dtype=np.int64)
    log_weights = decay_temperature * (idx_arr.astype(np.float64) / buffer_len)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    if total <= 0:
        return tuple(indices)
    probs = weights / total
    resampled = np.random.choice(idx_arr, size=len(idx_arr), replace=True, p=probs)
    return tuple(int(x) for x in resampled)


def _is_discrete_action(action) -> bool:
    """True when action is a scalar integer (DQN discrete index)."""
    arr = np.asarray(action)
    return arr.ndim == 0 and np.issubdtype(arr.dtype, np.integer)


def _hflip_discrete_action(action_idx, n_steer: int | None = None):
    """Mirror the steering component inside a composite discrete action index.

    Index layout: steer_idx * (n_gas * n_brake) + gas_idx * n_brake + brake_idx.
    Mirroring steering: new_steer = n_steer - 1 - steer_idx (symmetric about 0).

    Args:
        action_idx: Scalar integer action index.
        n_steer: Number of steer bins for composite discrete control. If omitted,
            uses bins configured via ``configure_discrete_steer_bins``.
    """
    from tmrl.custom.tm.utils.discrete_control import BRAKE_TAP_TABLE_N_BRAKE, BRAKE_TAP_TABLE_N_GAS

    n_steer = _DISCRETE_STEER_BINS if n_steer is None else int(n_steer)
    if n_steer is None:
        raise RuntimeError(
            "Discrete action flip requested without a configured steer-bin count. "
            "Set discrete_n_steer_bins in memory construction."
        )
    idx = int(action_idx)
    gas_brake = BRAKE_TAP_TABLE_N_GAS * BRAKE_TAP_TABLE_N_BRAKE
    steer_idx = idx // gas_brake
    remainder = idx % gas_brake
    return np.int64((n_steer - 1 - steer_idx) * gas_brake + remainder)


def _hflip_action(action):
    """Negate the steering component (index 2) of an action array.

    Action order is [gas, brake, steer] (control_gamepad / send_control).
    Do NOT negate index 0 (gas) or index 1 (brake); only steer (index 2) flips.
    For discrete (DQN) actions, mirrors the steer index within the composite action.
    """
    if _is_discrete_action(action):
        if _DISCRETE_STEER_BINS is None:
            raise RuntimeError(
                "Discrete action flip requested before steer bins were configured. "
                "Pass discrete_n_steer_bins from config to memory constructors."
            )
        return _hflip_discrete_action(action, _DISCRETE_STEER_BINS)
    action_arr = np.array(action, dtype=np.float32, copy=True)
    if len(action_arr) >= 3:
        action_arr[ACTION_STEER_INDEX] *= -1.0
    return action_arr
