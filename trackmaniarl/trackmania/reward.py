"""Geometry-based, deterministic TrackMania progress reward."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from trackmaniarl.trackmania.pace import ReferencePaceProfile


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    terminated: bool
    reason: str | None
    time_reward: float = 0.0
    pbrs_reward: float = 0.0
    progress_reward: float = 0.0
    projected_velocity_reward: float = 0.0
    projected_speed_reward: float = 0.0
    steering_delta_reward: float = 0.0
    time_attack_terminal_reward: float = 0.0
    pace_reward: float = 0.0
    terminal_reward: float = 0.0
    collision_reward: float = 0.0
    collided: bool = False
    collision_detected: bool = False
    potential_progress: float = 0.0
    projected_velocity_mps: float = 0.0
    projected_velocity_ratio: float = 0.0
    reference_time_s: float = 0.0
    time_debt_s: float = 0.0
    nearest_distance_m: float = 0.0
    accepted_progress_delta_m: float = 0.0
    window_progress_m: float = 0.0
    steps_since_progress: int = 0


class TrajectoryReward:
    """Time-trial reward with potential-based trajectory shaping."""

    def __init__(
        self,
        trajectory: np.ndarray,
        *,
        crash_distance: float = 25.0,
        finish_progress: float = 0.995,
        no_progress_steps: int = 200,
        slow_progress_window_steps: int = 80,
        minimum_progress_per_window_m: float = 2.0,
        terminal_failure_penalty: float = 1.0,
        collision_penalty: float = 0.05,
        collision_cooldown_s: float = 0.0,
        minimum_finish_steps: int = 50,
        nearest_forward_points: int = 500,
        nearest_backward_points: int = 10,
        time_penalty_per_second: float = 0.1,
        max_time_delta_s: float = 1.0,
        maximum_race_time_s: float | None = None,
        progress_reward_full_lap: float = 10.0,
        finish_reward: float = 30.0,
        potential_progress_weight: float = 2.0,
        max_projected_speed_mps: float = 100.0,
        velocity_to_mps_scale: float = 0.001,
        projected_velocity_scale: float = 0.0,
        projected_speed_bonus_scale: float = 0.0,
        steering_delta_penalty: float = 0.0,
        time_attack_target_s: float | None = None,
        time_attack_bonus_scale: float = 0.0,
        time_attack_linear_scale: float = 0.0,
        pace_profile: ReferencePaceProfile | None = None,
        pace_reward_scale: float = 0.0,
        pace_debt_clip_s: float = 10.0,
        reward_gamma: float = 0.995,
    ) -> None:
        points = np.asarray(trajectory, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            raise ValueError("trajectory must have shape (points >= 2, coordinates >= 3)")
        if not np.isfinite(points).all():
            raise ValueError("trajectory points must be finite")
        self.points = points[:, :3]
        segment_directions = np.diff(self.points, axis=0)
        segment_lengths = np.linalg.norm(segment_directions, axis=1)
        if np.any(segment_lengths <= 0.0):
            raise ValueError("trajectory must not contain adjacent duplicate points")
        unit_directions = segment_directions / segment_lengths[:, None]
        if len(unit_directions) > 1 and np.any(
            np.linalg.norm(unit_directions[:-1] + unit_directions[1:], axis=1) <= 1.0e-6
        ):
            raise ValueError("trajectory must not contain a zero-length local tangent")
        self._segment_directions = unit_directions
        reward_limits = (
            crash_distance,
            finish_progress,
            minimum_progress_per_window_m,
            terminal_failure_penalty,
            collision_penalty,
            collision_cooldown_s,
            time_penalty_per_second,
            max_time_delta_s,
            progress_reward_full_lap,
            finish_reward,
            potential_progress_weight,
            max_projected_speed_mps,
            velocity_to_mps_scale,
            projected_velocity_scale,
            projected_speed_bonus_scale,
            steering_delta_penalty,
            time_attack_bonus_scale,
            time_attack_linear_scale,
            pace_reward_scale,
            pace_debt_clip_s,
            reward_gamma,
        )
        optional_limits = (maximum_race_time_s, time_attack_target_s)
        if not all(np.isfinite(value) for value in reward_limits) or not all(
            value is None or np.isfinite(value) for value in optional_limits
        ):
            raise ValueError("reward limits must be finite")
        if crash_distance <= 0.0:
            raise ValueError("crash_distance must be positive")
        if not 0.0 < finish_progress <= 1.0:
            raise ValueError("finish_progress must be in (0, 1]")
        self.crash_distance = crash_distance
        self.finish_progress = finish_progress
        if no_progress_steps < 1 or slow_progress_window_steps < 2:
            raise ValueError("progress timeout windows must be positive")
        if (
            minimum_progress_per_window_m < 0.0
            or terminal_failure_penalty < 0.0
            or minimum_finish_steps < 1
            or collision_penalty < 0.0
            or collision_cooldown_s < 0.0
            or nearest_forward_points < 1
            or nearest_backward_points < 0
            or time_penalty_per_second < 0.0
            or max_time_delta_s <= 0.0
            or progress_reward_full_lap < 0.0
            or finish_reward < 0.0
            or potential_progress_weight < 0.0
            or max_projected_speed_mps <= 0.0
            or velocity_to_mps_scale <= 0.0
            or projected_velocity_scale < 0.0
            or projected_speed_bonus_scale < 0.0
            or steering_delta_penalty < 0.0
            or time_attack_bonus_scale < 0.0
            or time_attack_linear_scale < 0.0
            or pace_reward_scale < 0.0
            or pace_debt_clip_s <= 0.0
            or not 0.0 <= reward_gamma <= 1.0
        ):
            raise ValueError("reward limits must be non-negative")
        if time_attack_target_s is not None and time_attack_target_s <= 0.0:
            raise ValueError("time_attack_target_s must be positive")
        if maximum_race_time_s is not None and maximum_race_time_s <= 0.0:
            raise ValueError("maximum_race_time_s must be positive")
        if (time_attack_bonus_scale or time_attack_linear_scale) and time_attack_target_s is None:
            raise ValueError("time-attack reward scales require time_attack_target_s")
        if pace_reward_scale and pace_profile is None:
            raise ValueError("pace_reward_scale requires a pace_profile")
        if pace_profile is not None and len(pace_profile.reference_times_s) != len(self.points):
            raise ValueError("pace profile length must match trajectory length")
        self.no_progress_steps = no_progress_steps
        self.slow_progress_window_steps = slow_progress_window_steps
        self.minimum_progress_per_window_m = minimum_progress_per_window_m
        self.terminal_failure_penalty = terminal_failure_penalty
        self.collision_penalty = collision_penalty
        self.collision_cooldown_s = collision_cooldown_s
        self.minimum_finish_steps = minimum_finish_steps
        self.nearest_forward_points = nearest_forward_points
        self.nearest_backward_points = nearest_backward_points
        self.time_penalty_per_second = time_penalty_per_second
        self.max_time_delta_s = max_time_delta_s
        self.maximum_race_time_s = maximum_race_time_s
        self.progress_reward_full_lap = progress_reward_full_lap
        self.finish_reward = finish_reward
        self.potential_progress_weight = potential_progress_weight
        self.max_projected_speed_mps = max_projected_speed_mps
        self.velocity_to_mps_scale = velocity_to_mps_scale
        self.projected_velocity_scale = projected_velocity_scale
        self.projected_speed_bonus_scale = projected_speed_bonus_scale
        self.steering_delta_penalty = steering_delta_penalty
        self.time_attack_target_s = time_attack_target_s
        self.time_attack_bonus_scale = time_attack_bonus_scale
        self.time_attack_linear_scale = time_attack_linear_scale
        self.pace_profile = pace_profile
        self.pace_reward_scale = pace_reward_scale
        self.pace_debt_clip_s = pace_debt_clip_s
        self.reward_gamma = reward_gamma
        self._cumulative_distance = np.r_[0.0, np.cumsum(segment_lengths)]
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
    def from_file(cls, path: str | Path, **kwargs: Any) -> TrajectoryReward:
        source = Path(path)
        values = np.load(source) if source.suffix == ".npy" else np.loadtxt(source, delimiter=",")
        return cls(values, **kwargs)

    def reset(
        self,
        position: np.ndarray | None = None,
        *,
        velocity: np.ndarray | None = None,
        race_time_ms: float | None = None,
    ) -> None:
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
        if velocity is not None:
            self._vector3("velocity", velocity)
        if position is not None:
            point = self._vector3("position", position)
            self._index, _ = self._nearest_point(point)
            self._previous_potential = self._potential()
            self._previous_position = point
            self._reachable_progress_m = self.progress_m
        self._previous_time_debt_s = self._time_debt(race_time_ms)[1]

    @property
    def progress_m(self) -> float:
        """Distance reached along the recorded centre line in metres."""

        return float(self._cumulative_distance[self._index])

    @property
    def progress_pct(self) -> float:
        """Monotonic centre-line completion percentage in the current episode."""

        return 100.0 * self.progress_m / max(1.0, float(self._cumulative_distance[-1]))

    def step(
        self,
        position: np.ndarray,
        *,
        finish_ui_active: bool,
        velocity: np.ndarray | None = None,
        race_time_ms: float | None = None,
        collision: bool = False,
        steering: float | None = None,
    ) -> RewardResult:
        """Score a transition from time, geometric progress, and velocity."""

        race_time_s = self._race_time_s(race_time_ms)
        time_reward, elapsed_s = self._time_reward(race_time_ms)
        point = self._vector3("position", position)
        if velocity is not None:
            self._vector3("velocity", velocity)
        if steering is not None and not np.isfinite(steering):
            raise ValueError("steering must be finite")
        nearest, nearest_distance = self._nearest_point(point)
        previous_index = self._index
        if nearest_distance <= self.crash_distance:
            self._index = max(
                self._index,
                self._bounded_advance(
                    nearest,
                    point,
                    elapsed_s,
                    has_race_time=race_time_ms is not None,
                ),
            )
        else:
            self._previous_position = point
        self._step += 1
        progress_m = float(self._cumulative_distance[self._index])
        previous_progress_m = float(self._cumulative_distance[previous_index])
        self._nearest_distance_m = nearest_distance
        self._accepted_progress_delta_m = progress_m - previous_progress_m
        progress_reward = self._progress_reward(previous_index)
        if self._index > previous_index:
            self._last_progress_step = self._step
        self._progress_history.append((self._step, progress_m))
        while self._progress_history and (
            self._step - self._progress_history[0][0] > self.slow_progress_window_steps
        ):
            self._progress_history.popleft()
        self._window_progress_m = (
            progress_m - self._progress_history[0][1] if self._progress_history else 0.0
        )
        near_finish = (
            self.progress_m / max(1.0, float(self._cumulative_distance[-1])) >= self.finish_progress
        )
        valid_finish = (
            finish_ui_active
            and near_finish
            and self._step >= self.minimum_finish_steps
            and nearest_distance <= self.crash_distance
        )
        if valid_finish and self._index < len(self.points) - 1:
            remaining_m = float(self._cumulative_distance[-1]) - progress_m
            progress_reward += (
                self.progress_reward_full_lap
                * remaining_m
                / max(1.0, float(self._cumulative_distance[-1]))
            )
            self._index = len(self.points) - 1
            progress_m = float(self._cumulative_distance[-1])
        potential = self._potential()
        progress_potential = potential
        projected_velocity_mps = self._projected_velocity_mps(velocity)
        projected_velocity_ratio = float(
            np.clip(projected_velocity_mps / self.max_projected_speed_mps, -1.0, 1.0)
        )
        projected_velocity_reward = (
            self.projected_velocity_scale * projected_velocity_mps * elapsed_s
        )
        projected_speed_reward = (
            self.projected_speed_bonus_scale * max(0.0, projected_velocity_ratio) ** 2 * elapsed_s
        )
        steering_delta_reward = self._steering_delta_reward(steering)
        if self._previous_potential is None:
            self._previous_potential = potential
        off_track = nearest_distance > self.crash_distance
        time_limit = (
            self.maximum_race_time_s is not None
            and race_time_s is not None
            and race_time_s >= self.maximum_race_time_s
        )
        no_progress = self._step - self._last_progress_step >= self.no_progress_steps
        slow_progress = (
            len(self._progress_history) >= 2
            and self._step >= self.slow_progress_window_steps
            and self._below_progress_threshold()
        )
        terminal = off_track or valid_finish or time_limit or no_progress or slow_progress
        pace_reward, reference_time_s, time_debt_s = self._pace_reward(
            race_time_ms,
            finished=valid_finish,
            terminal=terminal,
        )
        if off_track:
            return self._with_pace(
                self._apply_collision(
                    self._terminal(
                        "off_track",
                        time_reward,
                        progress_potential,
                        -abs(self.terminal_failure_penalty),
                        progress_reward,
                        projected_velocity_reward,
                        projected_speed_reward,
                        steering_delta_reward,
                        projected_velocity_mps,
                        projected_velocity_ratio,
                    ),
                    collision,
                    race_time_s,
                ),
                pace_reward,
                reference_time_s,
                time_debt_s,
            )
        if valid_finish:
            time_attack_reward = self._time_attack_terminal_reward(race_time_s)
            return self._with_pace(
                self._apply_collision(
                    self._terminal(
                        "finished",
                        time_reward,
                        progress_potential,
                        self.finish_reward,
                        progress_reward,
                        projected_velocity_reward,
                        projected_speed_reward,
                        steering_delta_reward,
                        projected_velocity_mps,
                        projected_velocity_ratio,
                        time_attack_terminal_reward=time_attack_reward,
                    ),
                    collision,
                    race_time_s,
                ),
                pace_reward,
                reference_time_s,
                time_debt_s,
            )
        if time_limit:
            return self._with_pace(
                self._apply_collision(
                    self._terminal(
                        "time_limit",
                        time_reward,
                        progress_potential,
                        -abs(self.terminal_failure_penalty),
                        progress_reward,
                        projected_velocity_reward,
                        projected_speed_reward,
                        steering_delta_reward,
                        projected_velocity_mps,
                        projected_velocity_ratio,
                    ),
                    collision,
                    race_time_s,
                ),
                pace_reward,
                reference_time_s,
                time_debt_s,
            )
        if no_progress:
            return self._with_pace(
                self._apply_collision(
                    self._terminal(
                        "no_progress",
                        time_reward,
                        progress_potential,
                        -abs(self.terminal_failure_penalty),
                        progress_reward,
                        projected_velocity_reward,
                        projected_speed_reward,
                        steering_delta_reward,
                        projected_velocity_mps,
                        projected_velocity_ratio,
                    ),
                    collision,
                    race_time_s,
                ),
                pace_reward,
                reference_time_s,
                time_debt_s,
            )
        if slow_progress:
            return self._with_pace(
                self._apply_collision(
                    self._terminal(
                        "slow_progress",
                        time_reward,
                        progress_potential,
                        -abs(self.terminal_failure_penalty),
                        progress_reward,
                        projected_velocity_reward,
                        projected_speed_reward,
                        steering_delta_reward,
                        projected_velocity_mps,
                        projected_velocity_ratio,
                    ),
                    collision,
                    race_time_s,
                ),
                pace_reward,
                reference_time_s,
                time_debt_s,
            )
        pbrs_reward = self.reward_gamma * potential - self._previous_potential
        self._previous_potential = potential
        return self._with_pace(
            self._apply_collision(
                RewardResult(
                    reward=(
                        time_reward
                        + pbrs_reward
                        + progress_reward
                        + projected_velocity_reward
                        + projected_speed_reward
                        + steering_delta_reward
                    ),
                    terminated=False,
                    reason=None,
                    time_reward=time_reward,
                    pbrs_reward=pbrs_reward,
                    progress_reward=progress_reward,
                    projected_velocity_reward=projected_velocity_reward,
                    projected_speed_reward=projected_speed_reward,
                    steering_delta_reward=steering_delta_reward,
                    potential_progress=progress_potential,
                    projected_velocity_mps=projected_velocity_mps,
                    projected_velocity_ratio=projected_velocity_ratio,
                ),
                collision,
                race_time_s,
            ),
            pace_reward,
            reference_time_s,
            time_debt_s,
        )

    def _apply_collision(
        self, result: RewardResult, collision: bool, race_time_s: float | None
    ) -> RewardResult:
        if not collision:
            return result
        if (
            race_time_s is not None
            and self._last_penalized_collision_s is not None
            and race_time_s - self._last_penalized_collision_s < self.collision_cooldown_s
        ):
            return replace(result, collision_detected=True)
        self._last_penalized_collision_s = race_time_s
        collision_reward = -self.collision_penalty
        return replace(
            result,
            reward=result.reward + collision_reward,
            collision_reward=collision_reward,
            collided=True,
            collision_detected=True,
        )

    def _nearest_point(self, point: np.ndarray) -> tuple[int, float]:
        window_start = max(0, self._index - self.nearest_backward_points)
        window_stop = min(len(self.points), self._index + self.nearest_forward_points + 1)
        distances = np.linalg.norm(self.points[window_start:window_stop] - point, axis=1)
        nearest = window_start + int(np.argmin(distances))
        return nearest, float(distances[nearest - window_start])

    def _bounded_advance(
        self,
        nearest: int,
        point: np.ndarray,
        elapsed_s: float,
        *,
        has_race_time: bool,
    ) -> int:
        previous = self._previous_position
        self._previous_position = point
        if previous is None:
            return self._index
        displacement_m = float(np.linalg.norm(point - previous))
        time_budget_s = elapsed_s if has_race_time else self.max_time_delta_s
        accepted_motion_m = min(displacement_m, self.max_projected_speed_mps * time_budget_s)
        self._reachable_progress_m += accepted_motion_m
        reachable = (
            int(
                np.searchsorted(self._cumulative_distance, self._reachable_progress_m, side="right")
            )
            - 1
        )
        return min(nearest, max(reachable, self._index))

    def _potential(self) -> float:
        return (
            self.potential_progress_weight
            * self.progress_m
            / max(1.0, float(self._cumulative_distance[-1]))
        )

    def _progress_reward(self, previous_index: int) -> float:
        progress_delta = (
            self._cumulative_distance[self._index] - self._cumulative_distance[previous_index]
        )
        return float(
            self.progress_reward_full_lap
            * progress_delta
            / max(1.0, float(self._cumulative_distance[-1]))
        )

    def _projected_velocity_mps(self, velocity: np.ndarray | None) -> float:
        if velocity is None:
            return 0.0
        tangent = self._path_tangent()
        velocity_mps = self._vector3("velocity", velocity) * self.velocity_to_mps_scale
        projected = float(np.dot(velocity_mps, tangent))
        return float(
            np.clip(projected, -self.max_projected_speed_mps, self.max_projected_speed_mps)
        )

    def _steering_delta_reward(self, steering: float | None) -> float:
        if steering is None:
            return 0.0
        current = float(np.clip(steering, -1.0, 1.0))
        delta = abs(current - self._previous_steering)
        self._previous_steering = current
        return -self.steering_delta_penalty * delta

    def _time_attack_terminal_reward(self, race_time_s: float | None) -> float:
        if self.time_attack_target_s is None or race_time_s is None:
            return 0.0
        improvement_s = self.time_attack_target_s - race_time_s
        raw_reward = (
            self.time_attack_bonus_scale * max(0.0, improvement_s) ** 2
            + self.time_attack_linear_scale * improvement_s
        )
        return max(-self.finish_reward, raw_reward)

    def _below_progress_threshold(self) -> bool:
        threshold = self.minimum_progress_per_window_m
        return self._window_progress_m < threshold and not np.isclose(
            self._window_progress_m,
            threshold,
            rtol=1.0e-3,
            atol=1.0e-3,
        )

    def _time_debt(
        self, race_time_ms: float | None, *, finished: bool = False
    ) -> tuple[float, float]:
        if self.pace_profile is None or race_time_ms is None:
            return 0.0, 0.0
        reference_time_s = (
            float(self.pace_profile.reference_times_s[-1])
            if finished
            else self.pace_profile.time_at_index(self._index)
        )
        race_time_s = self._race_time_s(race_time_ms)
        assert race_time_s is not None
        debt = float(
            np.clip(race_time_s - reference_time_s, -self.pace_debt_clip_s, self.pace_debt_clip_s)
        )
        return reference_time_s, debt

    def _pace_reward(
        self,
        race_time_ms: float | None,
        *,
        finished: bool = False,
        terminal: bool = False,
    ) -> tuple[float, float, float]:
        reference_time_s, time_debt_s = self._time_debt(race_time_ms, finished=finished)
        previous = self._previous_time_debt_s
        self._previous_time_debt_s = time_debt_s
        if self.pace_profile is None or previous is None:
            return 0.0, reference_time_s, time_debt_s
        previous_potential = -self.pace_reward_scale * previous
        current_potential = -self.pace_reward_scale * time_debt_s
        shaping = (
            -previous_potential
            if terminal
            else self.reward_gamma * current_potential - previous_potential
        )
        return shaping, reference_time_s, time_debt_s

    def _with_pace(
        self,
        result: RewardResult,
        pace_reward: float,
        reference_time_s: float,
        time_debt_s: float,
    ) -> RewardResult:
        return replace(
            result,
            reward=result.reward + pace_reward,
            pace_reward=pace_reward,
            reference_time_s=reference_time_s,
            time_debt_s=time_debt_s,
            nearest_distance_m=self._nearest_distance_m,
            accepted_progress_delta_m=self._accepted_progress_delta_m,
            window_progress_m=self._window_progress_m,
            steps_since_progress=self._step - self._last_progress_step,
        )

    def _path_tangent(self) -> np.ndarray:
        if self._index == 0:
            direction = self._segment_directions[0]
        elif self._index == len(self.points) - 1:
            direction = self._segment_directions[-1]
        else:
            direction = (
                self._segment_directions[self._index - 1] + self._segment_directions[self._index]
            )
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("trajectory must not contain a zero-length local tangent")
        return cast(np.ndarray, direction / norm)

    def _time_reward(self, race_time_ms: float | None) -> tuple[float, float]:
        race_time_s = self._race_time_s(race_time_ms)
        if race_time_s is None or self._previous_race_time_s is None:
            self._previous_race_time_s = race_time_s
            return 0.0, 0.0
        elapsed_s = race_time_s - self._previous_race_time_s
        if elapsed_s < 0.0:
            raise ValueError("race time must be monotonic within an episode")
        self._previous_race_time_s = race_time_s
        bounded_elapsed_s = min(elapsed_s, self.max_time_delta_s)
        return -self.time_penalty_per_second * bounded_elapsed_s, bounded_elapsed_s

    @staticmethod
    def _race_time_s(race_time_ms: float | None) -> float | None:
        if race_time_ms is None:
            return None
        value = float(race_time_ms)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("race time must be finite and non-negative")
        return value / 1_000.0

    @staticmethod
    def _vector3(name: str, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite vector with shape (3,)")
        return vector

    def _terminal(
        self,
        reason: str,
        time_reward: float,
        progress_potential: float,
        terminal_reward: float,
        progress_reward: float,
        projected_velocity_reward: float,
        projected_speed_reward: float,
        steering_delta_reward: float,
        projected_velocity_mps: float,
        projected_velocity_ratio: float,
        time_attack_terminal_reward: float = 0.0,
    ) -> RewardResult:
        pbrs_reward = -float(self._previous_potential or 0.0)
        self._previous_potential = 0.0
        return RewardResult(
            reward=(
                time_reward
                + pbrs_reward
                + progress_reward
                + projected_velocity_reward
                + projected_speed_reward
                + steering_delta_reward
                + terminal_reward
                + time_attack_terminal_reward
            ),
            terminated=True,
            reason=reason,
            time_reward=time_reward,
            pbrs_reward=pbrs_reward,
            progress_reward=progress_reward,
            projected_velocity_reward=projected_velocity_reward,
            projected_speed_reward=projected_speed_reward,
            steering_delta_reward=steering_delta_reward,
            time_attack_terminal_reward=time_attack_terminal_reward,
            terminal_reward=terminal_reward,
            potential_progress=progress_potential,
            projected_velocity_mps=projected_velocity_mps,
            projected_velocity_ratio=projected_velocity_ratio,
        )
