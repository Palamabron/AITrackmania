"""Geometry-based, deterministic TrackMania progress reward."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    terminated: bool
    reason: str | None
    time_reward: float = 0.0
    pbrs_reward: float = 0.0
    progress_reward: float = 0.0
    projected_velocity_reward: float = 0.0
    steering_delta_reward: float = 0.0
    terminal_reward: float = 0.0
    collision_reward: float = 0.0
    collided: bool = False
    potential_progress: float = 0.0
    projected_velocity_mps: float = 0.0
    projected_velocity_ratio: float = 0.0


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
        minimum_finish_steps: int = 50,
        nearest_forward_points: int = 500,
        nearest_backward_points: int = 10,
        time_penalty_per_second: float = 0.1,
        max_time_delta_s: float = 1.0,
        progress_reward_full_lap: float = 10.0,
        finish_reward: float = 30.0,
        potential_progress_weight: float = 2.0,
        max_projected_speed_mps: float = 100.0,
        velocity_to_mps_scale: float = 0.001,
        projected_velocity_scale: float = 0.0,
        steering_delta_penalty: float = 0.0,
        reward_gamma: float = 0.995,
    ) -> None:
        points = np.asarray(trajectory, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            raise ValueError("trajectory must have shape (points >= 2, coordinates >= 3)")
        self.points = points[:, :3]
        self.crash_distance = crash_distance
        self.finish_progress = finish_progress
        if no_progress_steps < 1 or slow_progress_window_steps < 2:
            raise ValueError("progress timeout windows must be positive")
        if (
            minimum_progress_per_window_m < 0.0
            or minimum_finish_steps < 1
            or collision_penalty < 0.0
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
            or steering_delta_penalty < 0.0
            or not 0.0 <= reward_gamma <= 1.0
        ):
            raise ValueError("reward limits must be non-negative")
        self.no_progress_steps = no_progress_steps
        self.slow_progress_window_steps = slow_progress_window_steps
        self.minimum_progress_per_window_m = minimum_progress_per_window_m
        self.terminal_failure_penalty = terminal_failure_penalty
        self.collision_penalty = collision_penalty
        self.minimum_finish_steps = minimum_finish_steps
        self.nearest_forward_points = nearest_forward_points
        self.nearest_backward_points = nearest_backward_points
        self.time_penalty_per_second = time_penalty_per_second
        self.max_time_delta_s = max_time_delta_s
        self.progress_reward_full_lap = progress_reward_full_lap
        self.finish_reward = finish_reward
        self.potential_progress_weight = potential_progress_weight
        self.max_projected_speed_mps = max_projected_speed_mps
        self.velocity_to_mps_scale = velocity_to_mps_scale
        self.projected_velocity_scale = projected_velocity_scale
        self.steering_delta_penalty = steering_delta_penalty
        self.reward_gamma = reward_gamma
        self._cumulative_distance = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))
        ]
        self._index = 0
        self._step = 0
        self._last_progress_step = 0
        self._progress_history: deque[tuple[int, float]] = deque()
        self._previous_potential: float | None = None
        self._previous_race_time_s: float | None = None
        self._previous_steering = 0.0

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
        if position is not None:
            self._index, _ = self._nearest_point(np.asarray(position, dtype=np.float32)[:3])
            self._previous_potential = self._potential()

    @property
    def progress_m(self) -> float:
        """Distance reached along the recorded centre line in metres."""

        return float(self._cumulative_distance[self._index])

    @property
    def progress_pct(self) -> float:
        """Monotonic centre-line completion percentage in the current episode."""

        return 100.0 * self._index / max(1, len(self.points) - 1)

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

        point = np.asarray(position, dtype=np.float32)[:3]
        nearest, nearest_distance = self._nearest_point(point)
        previous_index = self._index
        self._index = max(self._index, nearest)
        self._step += 1
        progress_m = float(self._cumulative_distance[self._index])
        progress_reward = self._progress_reward(previous_index)
        if self._index > previous_index:
            self._last_progress_step = self._step
        self._progress_history.append((self._step, progress_m))
        while self._progress_history and (
            self._step - self._progress_history[0][0] > self.slow_progress_window_steps
        ):
            self._progress_history.popleft()
        time_reward, elapsed_s = self._time_reward(race_time_ms)
        potential = self._potential()
        progress_potential = potential
        projected_velocity_mps = self._projected_velocity_mps(velocity)
        projected_velocity_ratio = float(
            np.clip(projected_velocity_mps / self.max_projected_speed_mps, -1.0, 1.0)
        )
        projected_velocity_reward = (
            self.projected_velocity_scale * projected_velocity_mps * elapsed_s
        )
        steering_delta_reward = self._steering_delta_reward(steering)
        if self._previous_potential is None:
            self._previous_potential = potential
        if nearest_distance > self.crash_distance:
            return self._apply_collision(
                self._terminal(
                    "off_track",
                    time_reward,
                    progress_potential,
                    -abs(self.terminal_failure_penalty),
                    progress_reward,
                    projected_velocity_reward,
                    steering_delta_reward,
                    projected_velocity_mps,
                    projected_velocity_ratio,
                ),
                collision,
            )
        near_finish = self._index / (len(self.points) - 1) >= self.finish_progress
        if finish_ui_active and near_finish and self._step >= self.minimum_finish_steps:
            return self._apply_collision(
                self._terminal(
                    "finished",
                    time_reward,
                    progress_potential,
                    self.finish_reward,
                    progress_reward,
                    projected_velocity_reward,
                    steering_delta_reward,
                    projected_velocity_mps,
                    projected_velocity_ratio,
                ),
                collision,
            )
        if self._step - self._last_progress_step >= self.no_progress_steps:
            return self._apply_collision(
                self._terminal(
                    "no_progress",
                    time_reward,
                    progress_potential,
                    -abs(self.terminal_failure_penalty),
                    progress_reward,
                    projected_velocity_reward,
                    steering_delta_reward,
                    projected_velocity_mps,
                    projected_velocity_ratio,
                ),
                collision,
            )
        if (
            len(self._progress_history) >= 2
            and self._step >= self.slow_progress_window_steps
            and progress_m - self._progress_history[0][1] < self.minimum_progress_per_window_m
        ):
            return self._apply_collision(
                self._terminal(
                    "slow_progress",
                    time_reward,
                    progress_potential,
                    -abs(self.terminal_failure_penalty),
                    progress_reward,
                    projected_velocity_reward,
                    steering_delta_reward,
                    projected_velocity_mps,
                    projected_velocity_ratio,
                ),
                collision,
            )
        pbrs_reward = self.reward_gamma * potential - self._previous_potential
        self._previous_potential = potential
        return self._apply_collision(
            RewardResult(
                reward=(
                    time_reward
                    + pbrs_reward
                    + progress_reward
                    + projected_velocity_reward
                    + steering_delta_reward
                ),
                terminated=False,
                reason=None,
                time_reward=time_reward,
                pbrs_reward=pbrs_reward,
                progress_reward=progress_reward,
                projected_velocity_reward=projected_velocity_reward,
                steering_delta_reward=steering_delta_reward,
                potential_progress=progress_potential,
                projected_velocity_mps=projected_velocity_mps,
                projected_velocity_ratio=projected_velocity_ratio,
            ),
            collision,
        )

    def _apply_collision(self, result: RewardResult, collision: bool) -> RewardResult:
        if not collision:
            return result
        collision_reward = -self.collision_penalty
        return replace(
            result,
            reward=result.reward + collision_reward,
            collision_reward=collision_reward,
            collided=True,
        )

    def _nearest_point(self, point: np.ndarray) -> tuple[int, float]:
        window_start = max(0, self._index - self.nearest_backward_points)
        window_stop = min(len(self.points), self._index + self.nearest_forward_points + 1)
        distances = np.linalg.norm(self.points[window_start:window_stop] - point, axis=1)
        nearest = window_start + int(np.argmin(distances))
        return nearest, float(distances[nearest - window_start])

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
        velocity_mps = np.asarray(velocity, dtype=np.float32)[:3] * self.velocity_to_mps_scale
        return float(np.dot(velocity_mps, tangent))

    def _steering_delta_reward(self, steering: float | None) -> float:
        if steering is None:
            return 0.0
        current = float(np.clip(steering, -1.0, 1.0))
        delta = abs(current - self._previous_steering)
        self._previous_steering = current
        return -self.steering_delta_penalty * delta

    def _path_tangent(self) -> np.ndarray:
        if self._index == 0:
            direction = self.points[1] - self.points[0]
        elif self._index == len(self.points) - 1:
            direction = self.points[-1] - self.points[-2]
        else:
            direction = self.points[self._index + 1] - self.points[self._index - 1]
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
        self._previous_race_time_s = race_time_s
        bounded_elapsed_s = min(max(0.0, elapsed_s), self.max_time_delta_s)
        return -self.time_penalty_per_second * bounded_elapsed_s, bounded_elapsed_s

    @staticmethod
    def _race_time_s(race_time_ms: float | None) -> float | None:
        if race_time_ms is None:
            return None
        if race_time_ms < 0.0:
            raise ValueError("race time must be non-negative")
        return float(race_time_ms) / 1_000.0

    def _terminal(
        self,
        reason: str,
        time_reward: float,
        progress_potential: float,
        terminal_reward: float,
        progress_reward: float,
        projected_velocity_reward: float,
        steering_delta_reward: float,
        projected_velocity_mps: float,
        projected_velocity_ratio: float,
    ) -> RewardResult:
        pbrs_reward = -float(self._previous_potential or 0.0)
        self._previous_potential = 0.0
        return RewardResult(
            reward=(
                time_reward
                + pbrs_reward
                + progress_reward
                + projected_velocity_reward
                + steering_delta_reward
                + terminal_reward
            ),
            terminated=True,
            reason=reason,
            time_reward=time_reward,
            pbrs_reward=pbrs_reward,
            progress_reward=progress_reward,
            projected_velocity_reward=projected_velocity_reward,
            steering_delta_reward=steering_delta_reward,
            terminal_reward=terminal_reward,
            potential_progress=progress_potential,
            projected_velocity_mps=projected_velocity_mps,
            projected_velocity_ratio=projected_velocity_ratio,
        )
