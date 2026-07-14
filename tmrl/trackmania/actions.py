"""Discrete TrackMania action encodings.

These helpers are independent of a particular game-control backend: an
environment factory can translate the returned ``[gas, brake, steer]`` vectors
to OpenPlanet, a gamepad, or a test environment.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_STEER = 5
DEFAULT_N_GAS = 3
DEFAULT_N_BRAKE = 2
BRAKE_TAP_TABLE_N_STEER = 13
BRAKE_TAP_TABLE_N_GAS = 2
BRAKE_TAP_SENTINEL = -1.0
BRAKE_TAP_DURATION_S = 0.01
BRAKE_TAP_MATCH_PENALTY = 2.0


def build_discrete_to_continuous(
    n_steer: int = DEFAULT_N_STEER,
    n_gas: int = DEFAULT_N_GAS,
    n_brake: int = DEFAULT_N_BRAKE,
) -> tuple[int, list[np.ndarray]]:
    """Build a table mapping discrete action IDs to ``[gas, brake, steer]``."""
    if min(n_steer, n_gas, n_brake) < 1:
        raise ValueError("Each action dimension must contain at least one bin.")
    steer_values = np.linspace(-1.0, 1.0, n_steer, dtype=np.float32)
    gas_values = (
        np.linspace(0.0, 1.0, n_gas, dtype=np.float32)
        if n_gas > 1
        else np.array([1.0], dtype=np.float32)
    )
    brake_values = (
        np.linspace(0.0, 1.0, n_brake, dtype=np.float32)
        if n_brake > 1
        else np.array([0.0], dtype=np.float32)
    )
    table = [
        np.array([gas, brake, steer], dtype=np.float32)
        for steer in steer_values
        for gas in gas_values
        for brake in brake_values
    ]
    return len(table), table


def build_brake_tap_action_table(
    n_steer: int = BRAKE_TAP_TABLE_N_STEER,
    n_gas: int = BRAKE_TAP_TABLE_N_GAS,
) -> tuple[int, list[np.ndarray]]:
    """Build a discrete table with off, full-brake, and timed-brake modes."""
    if n_steer < 1 or n_gas < 1:
        raise ValueError("Each action dimension must contain at least one bin.")
    brakes = np.array([0.0, 1.0, BRAKE_TAP_SENTINEL], dtype=np.float32)
    steers = np.linspace(-1.0, 1.0, n_steer, dtype=np.float32)
    table = [
        np.array([float(gas), brake, steer], dtype=np.float32)
        for steer in steers
        for gas in range(n_gas)
        for brake in brakes
    ]
    return len(table), table


def is_brake_tap(control: np.ndarray) -> bool:
    """Return whether a table control requests a timed brake tap."""
    return float(control[1]) == BRAKE_TAP_SENTINEL


def discrete_index_to_control(action_index: int, table: list[np.ndarray]) -> np.ndarray:
    """Return a copy of the continuous control for one action ID."""
    return table[action_index].copy()


def discrete_indices_to_control_batch(
    action_indices: np.ndarray, table: list[np.ndarray]
) -> np.ndarray:
    """Map a batch of action IDs to an array of controls."""
    return np.asarray([table[int(index)] for index in action_indices], dtype=np.float32)


def continuous_control_to_discrete_index(control: np.ndarray, table: list[np.ndarray]) -> int:
    """Return the nearest action ID, preserving exact brake-tap controls."""
    gas, brake, steer = np.asarray(control, dtype=np.float32).reshape(-1)[:3]
    candidate = np.array([gas, brake, steer], dtype=np.float32)
    for index, entry in enumerate(table):
        if np.array_equal(np.asarray(entry, dtype=np.float32), candidate):
            return index
    distances = []
    for entry in table:
        table_gas, table_brake, table_steer = entry
        brake_distance = (
            BRAKE_TAP_MATCH_PENALTY
            if table_brake == BRAKE_TAP_SENTINEL
            else float((brake - table_brake) ** 2)
        )
        distances.append(
            float((gas - table_gas) ** 2 + brake_distance + (steer - table_steer) ** 2)
        )
    return int(np.argmin(distances))


def continuous_control_to_discrete_indices_batch(
    controls: np.ndarray, table: list[np.ndarray]
) -> np.ndarray:
    """Map a control batch to nearest discrete action IDs."""
    values = np.asarray(controls, dtype=np.float32)
    if values.ndim == 1:
        return np.asarray(continuous_control_to_discrete_index(values, table), dtype=np.int64)
    return np.asarray(
        [continuous_control_to_discrete_index(control, table) for control in values], dtype=np.int64
    )
