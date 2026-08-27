"""Geometry-based, deterministic TrackMania progress reward."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

import trackmaniarl.trackmania.reward_components as reward_components
import trackmaniarl.trackmania.reward_step as reward_step
from trackmaniarl.trackmania.reward_config import (
    RewardConfig,
    RewardTrajectory,
    build_reward_trajectory,
    validate_reward_config,
)
from trackmaniarl.trackmania.reward_types import (
    AdvanceRequest,
    CollisionRequest,
    PaceValues,
    TerminalReason,
    TerminalRequest,
    TransitionInput,
)
from trackmaniarl.trackmania.reward_types import (
    RewardResult as RewardResult,
)


class TrajectoryReward:
    """Time-trial reward with potential-based trajectory shaping."""

    def __init__(self, trajectory: np.ndarray, config: RewardConfig | None = None) -> None:
        config = config or RewardConfig()
        reward_trajectory = build_reward_trajectory(trajectory)
        validate_reward_config(config, len(reward_trajectory.points))
        self._set_trajectory(reward_trajectory)
        self._set_progress_config(config)
        self._set_reward_config(config)
        self._set_reference_config(config)
        self._initialize_state()

    def _set_trajectory(self, trajectory: RewardTrajectory) -> None:
        self.points = trajectory.points
        self._segment_directions = trajectory.segment_directions
        self._cumulative_distance = np.r_[0.0, np.cumsum(trajectory.segment_lengths)]

    def _set_progress_config(self, config: RewardConfig) -> None:
        self.crash_distance = config.crash_distance
        self.finish_progress = config.finish_progress
        self.no_progress_steps = config.no_progress_steps
        self.slow_progress_window_steps = config.slow_progress_window_steps
        self.minimum_progress_per_window_m = config.minimum_progress_per_window_m
        self.minimum_finish_steps = config.minimum_finish_steps
        self.nearest_forward_points = config.nearest_forward_points
        self.nearest_backward_points = config.nearest_backward_points
        self.max_time_delta_s = config.max_time_delta_s
        self.maximum_race_time_s = config.maximum_race_time_s

    def _set_reward_config(self, config: RewardConfig) -> None:
        self.terminal_failure_penalty = config.terminal_failure_penalty
        self.collision_penalty = config.collision_penalty
        self.collision_cooldown_s = config.collision_cooldown_s
        self.time_penalty_per_second = config.time_penalty_per_second
        self.progress_reward_full_lap = config.progress_reward_full_lap
        self.finish_reward = config.finish_reward
        self.potential_progress_weight = config.potential_progress_weight
        self.max_projected_speed_mps = config.max_projected_speed_mps
        self.velocity_to_mps_scale = config.velocity_to_mps_scale
        self.projected_velocity_scale = config.projected_velocity_scale
        self.projected_speed_bonus_scale = config.projected_speed_bonus_scale
        self.steering_delta_penalty = config.steering_delta_penalty
        self.reward_gamma = config.reward_gamma

    def _set_reference_config(self, config: RewardConfig) -> None:
        self.time_attack_target_s = config.time_attack_target_s
        self.time_attack_bonus_scale = config.time_attack_bonus_scale
        self.time_attack_linear_scale = config.time_attack_linear_scale
        self.pace_profile = config.pace_profile
        self.pace_reward_scale = config.pace_reward_scale
        self.pace_debt_clip_s = config.pace_debt_clip_s

    def _initialize_state(self) -> None:
        self._index = 0
        self._step = 0
        self._last_progress_step = 0
        self._progress_history: deque[tuple[int, float]] = deque()
        self._previous_potential: float | None = None
        self._previous_race_time_s: float | None = None
        self._previous_steering = 0.0
        self._last_penalized_collision_s: float | None = None
        self._previous_time_debt_s: float | None = None
        self._nearest_distance_m = 0.0
        self._accepted_progress_delta_m = 0.0
        self._window_progress_m = 0.0
        self._previous_position: np.ndarray | None = None
        self._reachable_progress_m = 0.0

    @classmethod
    def from_file(cls, path: str | Path, config: RewardConfig | None = None) -> TrajectoryReward:
        source = Path(path)
        values = np.load(source) if source.suffix == ".npy" else np.loadtxt(source, delimiter=",")
        return cls(values, config)

    def reset(
        self,
        position: np.ndarray | None = None,
        *,
        velocity: np.ndarray | None = None,
        race_time_ms: float | None = None,
    ) -> None:
        self._reset_state(race_time_ms)
        if velocity is not None:
            self._vector3("velocity", velocity)
        if position is not None:
            self._initialize_position(position)
        self._previous_time_debt_s = self._time_debt(race_time_ms)[1]

    def _reset_state(self, race_time_ms: float | None) -> None:
        self._index = 0
        self._step = 0
        self._last_progress_step = 0
        self._progress_history.clear()
        self._previous_potential = None
        self._previous_race_time_s = self._race_time_s(race_time_ms)
        self._previous_steering = 0.0
        self._last_penalized_collision_s = None
        self._nearest_distance_m = 0.0
        self._accepted_progress_delta_m = 0.0
        self._window_progress_m = 0.0
        self._previous_position = None
        self._reachable_progress_m = 0.0

    def _initialize_position(self, position: np.ndarray) -> None:
        point = self._vector3("position", position)
        self._index, _ = self._nearest_point(point)
        self._previous_potential = self._potential()
        self._previous_position = point
        self._reachable_progress_m = self.progress_m

    @property
    def progress_m(self) -> float:
        """Distance reached along the recorded centre line in metres."""

        return float(self._cumulative_distance[self._index])

    @property
    def progress_pct(self) -> float:
        """Monotonic centre-line completion percentage in the current episode."""

        return 100.0 * self.progress_m / max(1.0, float(self._cumulative_distance[-1]))

    def step(self, transition: TransitionInput) -> RewardResult:
        return reward_step.score_transition(self, transition)

    def _apply_collision(self, request: CollisionRequest) -> RewardResult:
        return reward_components.apply_collision(self, request)

    def _nearest_point(self, point: np.ndarray) -> tuple[int, float]:
        return reward_components.nearest_point(self, point)

    def _bounded_advance(self, request: AdvanceRequest) -> int:
        return reward_components.bounded_advance(self, request)

    def _potential(self) -> float:
        return reward_components.potential(self)

    def _progress_reward(self, previous_index: int) -> float:
        return reward_components.progress_reward(self, previous_index)

    def _projected_velocity_mps(self, velocity: np.ndarray | None) -> float:
        return reward_components.projected_velocity_mps(self, velocity)

    def _steering_delta_reward(self, steering: float | None) -> float:
        return reward_components.steering_delta_reward(self, steering)

    def _time_attack_terminal_reward(self, race_time_s: float | None) -> float:
        return reward_components.time_attack_terminal_reward(self, race_time_s)

    def _below_progress_threshold(self) -> bool:
        return reward_components.below_progress_threshold(self)

    def _time_debt(
        self,
        race_time_ms: float | None,
        terminal_reason: TerminalReason | None = None,
    ) -> tuple[float, float]:
        return reward_components.time_debt(self, race_time_ms, terminal_reason)

    def _pace_reward(
        self,
        race_time_ms: float | None,
        terminal_reason: TerminalReason | None = None,
    ) -> PaceValues:
        return reward_components.pace_reward(self, race_time_ms, terminal_reason)

    def _with_pace(self, result: RewardResult, pace: PaceValues) -> RewardResult:
        return reward_components.with_pace(self, result, pace)

    def _path_tangent(self) -> np.ndarray:
        return reward_components.path_tangent(self)

    def _time_reward(self, race_time_ms: float | None) -> tuple[float, float]:
        return reward_components.time_reward(self, race_time_ms)

    @staticmethod
    def _race_time_s(race_time_ms: float | None) -> float | None:
        return reward_components.race_time_s(race_time_ms)

    @staticmethod
    def _vector3(name: str, value: np.ndarray) -> np.ndarray:
        return reward_components.vector3(name, value)

    def _terminal(self, request: TerminalRequest) -> RewardResult:
        return reward_components.terminal(self, request)
