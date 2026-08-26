"""Discrete TrackMania action encodings.

These helpers are independent of a particular game-control backend: an
environment factory can translate the returned ``[gas, brake, steer]`` vectors
to OpenPlanet, a gamepad, or a test environment.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_N_STEER = 5
DEFAULT_N_GAS = 3
DEFAULT_N_BRAKE = 2
BRAKE_TAP_TABLE_N_STEER = 13
BRAKE_TAP_TABLE_N_GAS = 2
BRAKE_TAP_SENTINEL = -1.0
BRAKE_TAP_DURATION_S = 0.01
BRAKE_TAP_MATCH_PENALTY = 2.0


class TrackmaniaActionSelector:
    """TrackMania-aware weighted and steering-neighbor exploration."""

    def __init__(
        self,
        action_ids: tuple[int, ...] | None = None,
        *,
        minimum_action_hold_steps: int = 1,
        exploration_hold_steps: int = 1,
        switch_q_margin: float = 0.0,
        global_exploration_probability: float = 0.15,
    ) -> None:
        if min(minimum_action_hold_steps, exploration_hold_steps) < 1 or switch_q_margin < 0.0:
            raise ValueError("action stabilization parameters are invalid")
        if not 0.0 <= global_exploration_probability <= 1.0:
            raise ValueError("global exploration probability must be inside [0, 1]")
        self.action_ids = action_ids
        self.weights = torch.from_numpy(select_brake_tap_exploration_weights(action_ids))
        self.minimum_action_hold_steps = minimum_action_hold_steps
        self.exploration_hold_steps = exploration_hold_steps
        self.switch_q_margin = switch_q_margin
        self.global_exploration_probability = global_exploration_probability
        self._previous_action: int | None = None
        self._hold_steps = 0
        self._exploration_steps_remaining = 0

    def reset_episode(self) -> None:
        self._previous_action = None
        self._hold_steps = 0
        self._exploration_steps_remaining = 0

    def select(
        self,
        q_values: torch.Tensor,
        greedy: torch.Tensor,
        *,
        deterministic: bool,
        epsilon: float,
    ) -> torch.Tensor:
        if self.exploration_hold_steps > 1:
            return self._select_with_exploration_hold(
                q_values,
                greedy,
                deterministic=deterministic,
                epsilon=epsilon,
            )
        selected = greedy
        if not deterministic and epsilon:
            explore = torch.rand(greedy.shape, device=greedy.device) < epsilon
            random = self._exploration_action(q_values, greedy)
            selected = torch.where(explore, random, greedy)
        return self._stabilize(q_values, selected)

    def _select_with_exploration_hold(
        self,
        q_values: torch.Tensor,
        greedy: torch.Tensor,
        *,
        deterministic: bool,
        epsilon: float,
    ) -> torch.Tensor:
        if greedy.numel() != 1:
            raise ValueError("exploration action holding requires a single-policy batch")
        if self._exploration_steps_remaining:
            if self._previous_action is None:
                raise RuntimeError("exploration hold is missing its previous action")
            self._exploration_steps_remaining -= 1
            self._hold_steps += 1
            return greedy.new_tensor([self._previous_action]).reshape(greedy.shape)
        explore = bool(
            not deterministic and epsilon and torch.rand((), device=greedy.device) < epsilon
        )
        if not explore:
            return self._stabilize(q_values, greedy)
        selected = self._exploration_action(q_values, greedy)
        self._previous_action = int(selected.item())
        self._hold_steps = 1
        self._exploration_steps_remaining = self.exploration_hold_steps - 1
        return selected

    def _stabilize(self, q_values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if selected.numel() != 1:
            if self.minimum_action_hold_steps > 1 or self.switch_q_margin:
                raise ValueError("action stabilization requires a single-policy batch")
            return selected
        candidate = int(selected.item())
        previous = self._previous_action
        if previous is None:
            self._previous_action = candidate
            self._hold_steps = 1
            return selected
        if candidate == previous:
            self._hold_steps += 1
            return selected
        advantage = float(q_values.reshape(-1, q_values.shape[-1])[0, candidate]) - float(
            q_values.reshape(-1, q_values.shape[-1])[0, previous]
        )
        margin_blocks_switch = self.switch_q_margin > 0.0 and advantage < self.switch_q_margin
        if self._hold_steps < self.minimum_action_hold_steps or margin_blocks_switch:
            self._hold_steps += 1
            return selected.new_tensor([previous]).reshape(selected.shape)
        self._previous_action = candidate
        self._hold_steps = 1
        return selected

    def _exploration_action(self, q_values: torch.Tensor, greedy: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(q_values.device)
        global_action = torch.multinomial(weights, greedy.numel(), replacement=True).reshape(
            greedy.shape
        )
        if self.action_ids is not None:
            return global_action.to(greedy.dtype)
        modes_per_steering = 6
        steering_bins = q_values.shape[-1] // modes_per_steering
        if steering_bins * modes_per_steering != q_values.shape[-1]:
            return global_action.to(greedy.dtype)
        steering = greedy // modes_per_steering
        mode = greedy % modes_per_steering
        delta = torch.randint(-1, 2, greedy.shape, device=greedy.device, dtype=greedy.dtype)
        neighboring = (steering + delta).clamp(0, steering_bins - 1) * 6 + mode
        change_mode = (
            torch.rand(greedy.shape, device=greedy.device) < self.global_exploration_probability
        )
        return torch.where(change_mode, global_action, neighboring).to(greedy.dtype)


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


def build_brake_tap_exploration_weights() -> np.ndarray:
    """Return a throttle-biased exploratory distribution for the discrete action table."""

    _, table = build_brake_tap_action_table()
    weights = []
    for gas, brake, steer in table:
        mode_weight = {
            (1.0, 0.0): 8.0,
            (1.0, BRAKE_TAP_SENTINEL): 3.0,
            (1.0, 1.0): 2.0,
            (0.0, 0.0): 1.0,
            (0.0, BRAKE_TAP_SENTINEL): 0.5,
            (0.0, 1.0): 2.0,
        }[(float(gas), float(brake))]
        steering_weight = 1.0 - 0.6 * abs(float(steer))
        weights.append(mode_weight * steering_weight)
    return np.asarray(weights, dtype=np.float32)


def select_brake_tap_actions(action_ids: tuple[int, ...] | None) -> tuple[int, list[np.ndarray]]:
    """Select a validated compact subset from the canonical brake-tap table."""

    count, table = build_brake_tap_action_table()
    if action_ids is None:
        return count, table
    if not action_ids or len(set(action_ids)) != len(action_ids):
        raise ValueError("compact action IDs must be non-empty and unique")
    if any(action < 0 or action >= count for action in action_ids):
        raise ValueError(f"compact action IDs must be inside [0, {count})")
    return len(action_ids), [table[action] for action in action_ids]


def select_brake_tap_exploration_weights(action_ids: tuple[int, ...] | None) -> np.ndarray:
    """Return exploration weights aligned with a compact action subset."""

    weights = build_brake_tap_exploration_weights()
    return weights if action_ids is None else weights[list(action_ids)]


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
