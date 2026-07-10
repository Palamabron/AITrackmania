"""
Discrete action space that maps to continuous gamepad control.

Keeps the gamepad API (and thus crash/vibration feedback) unchanged:
the policy outputs a discrete action index; we map it to [forward, backward, steer]
and pass that to the existing control_gamepad(), so vibrations on guardrail hit still work.

Supports two presets:
  - Legacy (30 actions): 5 steer x 3 gas x 2 brake
  - Composite (78 actions): 13 steer x 2 gas x 3 brake (off / full / 0.01s tap)
"""

import numpy as np

# Legacy default bins: steer (5), gas (3), brake (2) -> 30 actions
DEFAULT_N_STEER = 5
DEFAULT_N_GAS = 3
DEFAULT_N_BRAKE = 2

# Default composite layout: 13 steer x 2 gas x 3 brake = 78 actions
BRAKE_TAP_TABLE_N_STEER = 13
BRAKE_TAP_TABLE_N_GAS = 2
BRAKE_TAP_TABLE_N_BRAKE = 3

BRAKE_TAP_SENTINEL = -1.0
BRAKE_TAP_DURATION_S = 0.01
BRAKE_TAP_MATCH_PENALTY = 2.0


def build_discrete_to_continuous(
    n_steer: int = DEFAULT_N_STEER,
    n_gas: int = DEFAULT_N_GAS,
    n_brake: int = DEFAULT_N_BRAKE,
) -> tuple[int, list[np.ndarray]]:
    """
    Build discrete action space size and mapping from index to continuous control.

    Control is [forward (gas), backward (brake), steer] in [0,1], [0,1], [-1,1] respectively.
    Action index = steer_idx * (n_gas * n_brake) + gas_idx * n_brake + brake_idx.

    Args:
        n_steer: Number of steering bins. Steers span [-1.0, 1.0] linearly.
        n_gas: Number of gas bins. Gas spans [0.0, 1.0]; if n_gas=1, fixed at 1.0.
        n_brake: Number of brake bins. Brake spans [0.0, 1.0]; if n_brake=1, fixed at 0.0.

    Returns:
        n_actions: total number of discrete actions.
        table: list of length n_actions; table[i] is np.array([gas, brake, steer]).
    """
    n_actions = n_steer * n_gas * n_brake
    steer_values = np.linspace(-1.0, 1.0, n_steer)
    gas_values = np.linspace(0.0, 1.0, n_gas) if n_gas > 1 else np.array([1.0])
    brake_values = np.linspace(0.0, 1.0, n_brake) if n_brake > 1 else np.array([0.0])
    table = []
    for si in range(n_steer):
        for gi in range(n_gas):
            for bi in range(n_brake):
                table.append(
                    np.array([gas_values[gi], brake_values[bi], steer_values[si]], dtype=np.float32)
                )
    return n_actions, table


def build_brake_tap_action_table(
    n_steer: int = BRAKE_TAP_TABLE_N_STEER,
    n_gas: int = BRAKE_TAP_TABLE_N_GAS,
) -> tuple[int, list[np.ndarray]]:
    """Build composite discrete table: n_steer x n_gas x 3 brake modes.

    Brake modes:
      0 = no brake (0.0)
      1 = full brake (1.0)
      2 = 0.01 s brake tap (encoded as BRAKE_TAP_SENTINEL)

    The sentinel is detected by send_control to fire a timed pulse.

    Args:
        n_steer: Number of steering bins (default 13). Steers span [-1.0, 1.0] linearly.
        n_gas: Number of gas bins (default 2). Gas is 0.0 (gi=0) or 1.0 (gi=1).

    Returns:
        n_actions: total discrete actions (default 78).
        table: list[np.array([gas, brake, steer])].
    """
    brake_values = np.array([0.0, 1.0, BRAKE_TAP_SENTINEL], dtype=np.float32)
    steer_values = np.linspace(-1.0, 1.0, n_steer, dtype=np.float32)
    n_brake = len(brake_values)
    n_actions = n_steer * n_gas * n_brake
    table: list[np.ndarray] = []
    for si in range(n_steer):
        for gi in range(n_gas):
            gas = float(gi)
            for bi in range(n_brake):
                table.append(np.array([gas, brake_values[bi], steer_values[si]], dtype=np.float32))
    return n_actions, table


def is_brake_tap(control: np.ndarray) -> bool:
    """Return True if the control vector encodes a 0.01 s brake tap.

    Args:
        control: np.array([gas, brake, steer]) as produced by
            ``build_brake_tap_action_table``.  The brake element is checked
            against BRAKE_TAP_SENTINEL (-1.0).

    Returns:
        True if the brake element equals BRAKE_TAP_SENTINEL.
    """
    return float(control[1]) == BRAKE_TAP_SENTINEL


def discrete_index_to_control(
    action_index: int,
    table: list[np.ndarray],
) -> np.ndarray:
    """Map a single discrete action index to continuous control [forward, backward, steer].

    Same format as expected by send_control / control_gamepad.

    Args:
        action_index: Integer index into ``table``, in [0, len(table)).
        table: Action table built by ``build_discrete_to_continuous`` or
            ``build_brake_tap_action_table``.

    Returns:
        np.array([gas, brake, steer], dtype=float32) — a copy, not a view.
    """
    return table[action_index].copy()


def discrete_indices_to_control_batch(
    action_indices: np.ndarray,
    table: list[np.ndarray],
) -> np.ndarray:
    """Map a batch of discrete indices to a (batch, 3) continuous control array.

    Args:
        action_indices: 1-D integer array of shape (batch,) with values in
            [0, len(table)).
        table: Action table built by ``build_discrete_to_continuous`` or
            ``build_brake_tap_action_table``.

    Returns:
        np.ndarray of shape (batch, 3) and dtype float32, each row being
        [gas, brake, steer].
    """
    return np.array([table[int(i)] for i in action_indices], dtype=np.float32)


def continuous_control_to_discrete_index(
    control: np.ndarray,
    table: list[np.ndarray],
) -> int:
    """
    Map continuous [gas, brake, steer] to the nearest discrete action index.

    Used when replay contains continuous actions (e.g. player runs) but the
    agent expects discrete indices (e.g. IQN). Brake tap sentinel is treated
    as far from any continuous value so we match to off or full brake only.

    Args:
        control: 1-D array-like [gas, brake, steer] with gas/brake in [0, 1]
            and steer in [-1, 1].
        table: Action table built by ``build_discrete_to_continuous`` or
            ``build_brake_tap_action_table``.

    Returns:
        Integer index of the nearest table entry under squared Euclidean
        distance, with BRAKE_TAP_SENTINEL entries penalised by
        BRAKE_TAP_MATCH_PENALTY so they are never selected from continuous
        inputs.
    """
    c = np.asarray(control, dtype=np.float32).flat
    gas, brake, steer = float(c[0]), float(c[1]), float(c[2])
    best_i = 0
    best_d2 = np.inf
    for i, t in enumerate(table):
        tg, tb, ts = float(t[0]), float(t[1]), float(t[2])
        # Brake: continuous [0,1] vs table 0, 1, or BRAKE_TAP_SENTINEL
        # Tap sentinel: never match from continuous; use large distance
        b_d2 = BRAKE_TAP_MATCH_PENALTY if tb == BRAKE_TAP_SENTINEL else (brake - tb) ** 2
        d2 = (gas - tg) ** 2 + b_d2 + (steer - ts) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def continuous_control_to_discrete_indices_batch(
    controls: np.ndarray,
    table: list[np.ndarray],
) -> np.ndarray:
    """Map a (batch, 3) continuous control array to (batch,) discrete indices.

    Applies ``continuous_control_to_discrete_index`` row-wise.  A 1-D input is
    treated as a single control vector and returns a 0-D array.

    Args:
        controls: np.ndarray of shape (batch, 3) or (3,) with dtype float32.
            Each row is [gas, brake, steer].
        table: Action table built by ``build_discrete_to_continuous`` or
            ``build_brake_tap_action_table``.

    Returns:
        np.ndarray of shape (batch,) or scalar with dtype int64 containing
        the nearest discrete action index for each control vector.
    """
    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim == 1:
        return np.array(continuous_control_to_discrete_index(controls, table), dtype=np.int64)
    idx_list = [
        continuous_control_to_discrete_index(controls[i], table) for i in range(controls.shape[0])
    ]
    return np.array(idx_list, dtype=np.int64)
