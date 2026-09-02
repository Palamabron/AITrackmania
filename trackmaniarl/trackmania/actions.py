"""Discrete TrackMania action encodings.

These helpers are independent of a particular game-control backend: an
environment factory can translate the returned ``[gas, brake, steer]`` vectors
to OpenPlanet, a gamepad, or a test environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import torch

from trackmaniarl.core.contracts import ActionSelectionRequest, PolicyMode

BRAKE_TAP_TABLE_N_STEER = 13
BRAKE_TAP_TABLE_N_GAS = 2
BRAKE_TAP_SENTINEL = -1.0
BRAKE_TAP_DURATION_S = 0.01
BRAKE_TAP_MATCH_PENALTY = 2.0


@dataclass(frozen=True, slots=True)
class _Selection:
    q_values: torch.Tensor
    greedy: torch.Tensor
    request: ActionSelectionRequest


@dataclass(frozen=True, slots=True)
class TrackmaniaActionSelectorConfig:
    action_ids: tuple[int, ...] | None = None
    minimum_action_hold_steps: int = 1
    exploration_hold_steps: int = 1
    switch_q_margin: float = 0.0
    global_exploration_probability: float = 0.15
    exploration_weights_preset: str = "throttle_biased"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        return cls(**dict(values))


def _selector_config(
    config: TrackmaniaActionSelectorConfig | Mapping[str, Any] | None,
) -> TrackmaniaActionSelectorConfig:
    if config is None:
        return TrackmaniaActionSelectorConfig()
    if isinstance(config, TrackmaniaActionSelectorConfig):
        return config
    return TrackmaniaActionSelectorConfig.from_mapping(config)


class TrackmaniaActionSelector:
    """TrackMania-aware weighted and steering-neighbor exploration."""

    _previous_action: int | None

    def __init__(
        self,
        config: TrackmaniaActionSelectorConfig | Mapping[str, Any] | None = None,
    ) -> None:
        config = _selector_config(config)
        _validate_stabilization(
            config.minimum_action_hold_steps,
            config.exploration_hold_steps,
            config.switch_q_margin,
        )
        _validate_probability(config.global_exploration_probability)
        self._configure(config)
        self.reset_episode()

    def _configure(self, config: TrackmaniaActionSelectorConfig) -> None:
        self.action_ids = config.action_ids
        self.weights = torch.from_numpy(
            select_exploration_weights(config.action_ids, config.exploration_weights_preset)
        )
        self.minimum_action_hold_steps = config.minimum_action_hold_steps
        self.exploration_hold_steps = config.exploration_hold_steps
        self.switch_q_margin = config.switch_q_margin
        self.global_exploration_probability = config.global_exploration_probability

    def reset_episode(self) -> None:
        self._previous_action = None
        self._hold_steps = 0
        self._exploration_steps_remaining = 0

    def select(
        self,
        q_values: torch.Tensor,
        greedy: torch.Tensor,
        request: ActionSelectionRequest,
    ) -> torch.Tensor:
        selection = _Selection(q_values, greedy, request)
        if self.exploration_hold_steps > 1:
            return self._select_with_exploration_hold(selection)
        selected = greedy
        if request.mode is PolicyMode.ONLINE and request.epsilon:
            explore = torch.rand(greedy.shape, device=greedy.device) < request.epsilon
            random = self._exploration_action(q_values, greedy)
            selected = torch.where(explore, random, greedy)
        return self._stabilize(q_values, selected)

    def _select_with_exploration_hold(self, request: _Selection) -> torch.Tensor:
        q_values, greedy = request.q_values, request.greedy
        if greedy.numel() != 1:
            raise ValueError("exploration action holding requires a single-policy batch")
        if self._exploration_steps_remaining:
            return self._held_exploration_action(greedy)
        explore = self._should_explore(request)
        if not explore:
            return self._stabilize(q_values, greedy)
        selected = self._exploration_action(q_values, greedy)
        self._start_exploration_hold(selected)
        return selected

    def _held_exploration_action(self, greedy: torch.Tensor) -> torch.Tensor:
        if self._previous_action is None:
            raise RuntimeError("exploration hold is missing its previous action")
        self._exploration_steps_remaining -= 1
        self._hold_steps += 1
        return greedy.new_tensor([self._previous_action]).reshape(greedy.shape)

    @staticmethod
    def _should_explore(request: _Selection) -> bool:
        selection = request.request
        return bool(
            selection.mode is PolicyMode.ONLINE
            and selection.epsilon
            and torch.rand((), device=request.greedy.device) < selection.epsilon
        )

    def _start_exploration_hold(self, selected: torch.Tensor) -> None:
        self._previous_action = int(selected.item())
        self._hold_steps = 1
        self._exploration_steps_remaining = self.exploration_hold_steps - 1

    def _stabilize(self, q_values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if selected.numel() != 1:
            self._validate_batch_stabilization()
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
        if self._switch_is_blocked(q_values, candidate, previous):
            self._hold_steps += 1
            return selected.new_tensor([previous]).reshape(selected.shape)
        self._previous_action = candidate
        self._hold_steps = 1
        return selected

    def _validate_batch_stabilization(self) -> None:
        if self.minimum_action_hold_steps > 1 or self.switch_q_margin:
            raise ValueError("action stabilization requires a single-policy batch")

    def _switch_is_blocked(self, q_values: torch.Tensor, candidate: int, previous: int) -> bool:
        values = q_values.reshape(-1, q_values.shape[-1])[0]
        advantage = float(values[candidate]) - float(values[previous])
        margin_blocks = self.switch_q_margin > 0.0 and advantage < self.switch_q_margin
        return self._hold_steps < self.minimum_action_hold_steps or margin_blocks

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


def _validate_stabilization(minimum: int, exploration: int, margin: float) -> None:
    if min(minimum, exploration) < 1 or margin < 0.0:
        raise ValueError("action stabilization parameters are invalid")


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("global exploration probability must be inside [0, 1]")


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


# Marginals measured on 10 expert keyboard laps (36.6-37.4 s): gas on 83%, brake 5.5%,
# steering exactly 0 / +1 / -1 on 35% / 37% / 28% of frames, never intermediate, never a tap.
EXPERT_KEYBOARD_MODE_WEIGHTS: dict[tuple[float, float], float] = {
    (1.0, 0.0): 0.83 * 0.94,
    (1.0, 1.0): 0.83 * 0.05,
    (1.0, BRAKE_TAP_SENTINEL): 0.83 * 0.01,
    (0.0, 0.0): 0.17 * 0.94,
    (0.0, 1.0): 0.17 * 0.05,
    (0.0, BRAKE_TAP_SENTINEL): 0.17 * 0.01,
}


def _expert_steering_weight(steer: float) -> float:
    if steer == 0.0:
        return 0.35
    if abs(steer) == 1.0:
        return 0.31
    return 0.005


def build_expert_keyboard_exploration_weights() -> np.ndarray:
    """Return exploration weights shaped like the expert's action marginals."""

    _, table = build_brake_tap_action_table()
    return np.asarray(
        [
            EXPERT_KEYBOARD_MODE_WEIGHTS[(float(gas), float(brake))]
            * _expert_steering_weight(float(steer))
            for gas, brake, steer in table
        ],
        dtype=np.float32,
    )


EXPLORATION_WEIGHT_PRESETS = {
    "throttle_biased": build_brake_tap_exploration_weights,
    "expert_keyboard": build_expert_keyboard_exploration_weights,
}


def select_exploration_weights(action_ids: tuple[int, ...] | None, preset: str) -> np.ndarray:
    """Return the preset exploration weights aligned with a compact action subset."""

    builder = EXPLORATION_WEIGHT_PRESETS.get(preset)
    if builder is None:
        raise ValueError(
            f"unknown exploration weights preset {preset!r}; "
            f"choose one of {sorted(EXPLORATION_WEIGHT_PRESETS)}"
        )
    weights = builder()
    return weights if action_ids is None else weights[list(action_ids)]


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
