"""Geometry-based, deterministic TrackMania progress reward."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    terminated: bool
    reason: str | None


class TrajectoryReward:
    """Progress reward over a recorded X/Y/Z trajectory without global configuration."""

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
        minimum_finish_steps: int = 50,
        nearest_forward_points: int = 500,
        nearest_backward_points: int = 10,
        progress_reward_full_lap: float = 200.0,
        finish_reward: float = 10.0,
        speed_reward_weight: float = 0.25,
        max_speed_mps: float = 100.0,
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
            or nearest_forward_points < 1
            or nearest_backward_points < 0
            or progress_reward_full_lap < 0.0
            or finish_reward < 0.0
            or speed_reward_weight < 0.0
            or max_speed_mps <= 0.0
        ):
            raise ValueError("progress and finish limits must be non-negative")
        self.no_progress_steps = no_progress_steps
        self.slow_progress_window_steps = slow_progress_window_steps
        self.minimum_progress_per_window_m = minimum_progress_per_window_m
        self.terminal_failure_penalty = terminal_failure_penalty
        self.minimum_finish_steps = minimum_finish_steps
        self.nearest_forward_points = nearest_forward_points
        self.nearest_backward_points = nearest_backward_points
        self.progress_reward_full_lap = progress_reward_full_lap
        self.finish_reward = finish_reward
        self.speed_reward_weight = speed_reward_weight
        self.max_speed_mps = max_speed_mps
        self._cumulative_distance = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))
        ]
        self._index = 0
        self._step = 0
        self._last_progress_step = 0
        self._progress_history: deque[tuple[int, float]] = deque()

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> TrajectoryReward:
        source = Path(path)
        values = np.load(source) if source.suffix == ".npy" else np.loadtxt(source, delimiter=",")
        return cls(values, **kwargs)

    def reset(self) -> None:
        self._index = 0
        self._step = 0
        self._last_progress_step = 0
        self._progress_history.clear()

    @property
    def progress_m(self) -> float:
        """Distance reached along the recorded centre line in metres."""

        return float(self._cumulative_distance[self._index])

    @property
    def progress_pct(self) -> float:
        """Monotonic centre-line completion percentage in the current episode."""

        return 100.0 * self._index / max(1, len(self.points) - 1)

    def step(
        self, position: np.ndarray, *, finish_ui_active: bool, speed_mps: float = 0.0
    ) -> RewardResult:
        """Score one frame; a geometric near-finish is never a finish without UI confirmation."""

        point = np.asarray(position, dtype=np.float32)[:3]
        # A global nearest-point lookup can jump directly to a later segment when
        # two parts of a map cross or run close together.  Keep the association
        # local to the currently reached path position, matching the legacy
        # reward's bounded forward/backward search.
        window_start = max(0, self._index - self.nearest_backward_points)
        window_stop = min(len(self.points), self._index + self.nearest_forward_points + 1)
        window_distances = np.linalg.norm(self.points[window_start:window_stop] - point, axis=1)
        nearest = window_start + int(np.argmin(window_distances))
        previous_index = self._index
        progress = max(0, nearest - previous_index) / max(1, len(self.points) - 1)
        progress_reward = progress * self.progress_reward_full_lap
        speed_reward = (
            self.speed_reward_weight * min(max(float(speed_mps), 0.0) / self.max_speed_mps, 1.0)
            if progress > 0.0
            else 0.0
        )
        self._index = max(self._index, nearest)
        self._step += 1
        progress_m = float(self._cumulative_distance[self._index])
        if self._index > previous_index:
            self._last_progress_step = self._step
        self._progress_history.append((self._step, progress_m))
        while self._progress_history and (
            self._step - self._progress_history[0][0] > self.slow_progress_window_steps
        ):
            self._progress_history.popleft()
        if float(window_distances[nearest - window_start]) > self.crash_distance:
            return RewardResult(-1.0, True, "off_track")
        near_finish = self._index / (len(self.points) - 1) >= self.finish_progress
        if finish_ui_active and near_finish and self._step >= self.minimum_finish_steps:
            return RewardResult(
                self.finish_reward + progress_reward + speed_reward, True, "finished"
            )
        if near_finish:
            # A valid finish signal can arrive after trajectory progress reaches the end.
            return RewardResult(progress_reward + speed_reward, False, None)
        if self._step - self._last_progress_step >= self.no_progress_steps:
            return RewardResult(-abs(self.terminal_failure_penalty), True, "no_progress")
        if (
            len(self._progress_history) >= 2
            and self._step >= self.slow_progress_window_steps
            and progress_m - self._progress_history[0][1] < self.minimum_progress_per_window_m
        ):
            return RewardResult(-abs(self.terminal_failure_penalty), True, "slow_progress")
        return RewardResult(progress_reward + speed_reward, False, None)
