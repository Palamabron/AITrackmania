"""
Discrete action space that maps to continuous gamepad control.

Keeps the gamepad API (and thus crash/vibration feedback) unchanged:
the policy outputs a discrete action index; we map it to [forward, backward, steer]
and pass that to the existing control_gamepad(), so vibrations on guardrail hit still work.
"""

import numpy as np


# Default bins: steer (e.g. 5), gas (3), brake (2) -> 5*3*2 = 30 actions
DEFAULT_N_STEER = 5   # left, left-mid, center, right-mid, right
DEFAULT_N_GAS = 3     # 0, half, full
DEFAULT_N_BRAKE = 2   # no brake, brake


def build_discrete_to_continuous(
    n_steer: int = DEFAULT_N_STEER,
    n_gas: int = DEFAULT_N_GAS,
    n_brake: int = DEFAULT_N_BRAKE,
) -> tuple[int, list[np.ndarray]]:
    """
    Build discrete action space size and mapping from index to continuous control.

    Control is [forward (gas), backward (brake), steer] in [0,1], [0,1], [-1,1] respectively.
    Action index = steer_idx * (n_gas * n_brake) + gas_idx * n_brake + brake_idx.

    Returns:
        n_actions: total number of discrete actions.
        table: list of length n_actions; table[i] is np.array([gas, brake, steer]).
    """
    n_actions = n_steer * n_gas * n_brake
    table = []
    for si in range(n_steer):
        steer = np.linspace(-1.0, 1.0, n_steer)[si]
        for gi in range(n_gas):
            gas = np.linspace(0.0, 1.0, n_gas)[gi] if n_gas > 1 else 1.0
            for bi in range(n_brake):
                brake = np.linspace(0.0, 1.0, n_brake)[bi] if n_brake > 1 else 0.0
                table.append(np.array([gas, brake, steer], dtype=np.float32))
    return n_actions, table


def discrete_index_to_control(
    action_index: int,
    table: list[np.ndarray],
) -> np.ndarray:
    """
    Map a single discrete action index to continuous control [forward, backward, steer].

    Same format as expected by send_control / control_gamepad.
    """
    return table[action_index].copy()


def discrete_indices_to_control_batch(
    action_indices: np.ndarray,
    table: list[np.ndarray],
) -> np.ndarray:
    """Map a batch of discrete indices to (batch, 3) continuous controls."""
    return np.array([table[int(i)] for i in action_indices], dtype=np.float32)
