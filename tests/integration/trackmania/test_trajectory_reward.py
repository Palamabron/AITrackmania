"""Release contracts for trajectory reward."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import pairwise

import numpy as np
import pytest

from trackmaniarl.trackmania.reward import RewardResult, TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import TransitionInput

_ORIGIN = np.zeros(3, dtype=np.float32)


class _FinishState(Enum):
    RUNNING = auto()
    FINISHED = auto()


class _Outcome(Enum):
    FINISH = auto()
    FAILURE = auto()


@dataclass(frozen=True, slots=True)
class _RolloutSpec:
    progress: float
    duration_s: float
    outcome: _Outcome


def _trajectory(length: int) -> np.ndarray:
    return np.stack(
        (
            np.arange(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
        ),
        axis=1,
    )


def _stationary_transition(
    position: np.ndarray, race_time_ms: float, state: _FinishState = _FinishState.RUNNING
) -> TransitionInput:
    return TransitionInput(
        position, state is _FinishState.FINISHED, _ORIGIN, race_time_ms, False, None
    )


def _short_reward() -> TrajectoryReward:
    trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    reward = TrajectoryReward(trajectory, RewardConfig(minimum_finish_steps=1))
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    return reward


def _shaping_total(result: RewardResult) -> float:
    return sum(
        (
            result.time_reward,
            result.pbrs_reward,
            result.progress_reward,
            result.projected_velocity_reward,
            result.steering_delta_reward,
        )
    )


def test_trajectory_reward_reports_progress_components() -> None:
    reward = _short_reward()
    progress = reward.step(_stationary_transition(np.array([1, 0, 0]), 100.0))
    assert progress.reward > 0
    assert progress.reward == _shaping_total(progress)


def test_trajectory_reward_reports_finish_components() -> None:
    reward = _short_reward()
    reward.step(_stationary_transition(np.array([2, 0, 0]), 200.0))
    finish = reward.step(_stationary_transition(np.array([2, 0, 0]), 300.0, _FinishState.FINISHED))
    assert finish.reason == "finished"
    assert finish.reward == _shaping_total(finish) + finish.terminal_reward


def test_off_track_transition_cannot_collect_progress() -> None:
    trajectory = _trajectory(101)
    config = RewardConfig(
        crash_distance=10.0,
        terminal_failure_penalty=0.0,
        time_penalty_per_second=0.0,
        potential_progress_weight=0.0,
    )
    reward = TrajectoryReward(trajectory, config)
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    result = reward.step(_stationary_transition(np.array([50, 20, 0]), 50.0))
    assert result.reason == "off_track"
    assert result.progress_reward == 0.0
    assert result.potential_progress == 0.0
    assert reward.progress_m == 0.0


def _task_aligned_reward(trajectory: np.ndarray) -> TrajectoryReward:
    config = RewardConfig(
        crash_distance=15.0,
        no_progress_steps=2_000,
        slow_progress_window_steps=2_000,
        minimum_finish_steps=1,
        terminal_failure_penalty=0.0,
        time_penalty_per_second=0.25,
        progress_reward_full_lap=40.0,
        finish_reward=60.0,
        potential_progress_weight=0.0,
        projected_velocity_scale=0.0,
        projected_speed_bonus_scale=0.0,
        time_attack_target_s=55.0,
        time_attack_linear_scale=2.0,
        reward_gamma=0.9994,
    )
    return TrajectoryReward(trajectory, config)


def _finish_transition(trajectory: np.ndarray, step: int, steps: int) -> TransitionInput:
    index = round(step * (len(trajectory) - 1) / steps)
    state = _FinishState.FINISHED if step == steps else _FinishState.RUNNING
    return _stationary_transition(trajectory[index], step * 50.0, state)


def _finish_return(trajectory: np.ndarray, finish_time_s: float) -> tuple[float, float]:
    reward = _task_aligned_reward(trajectory)
    reward.reset(trajectory[0], velocity=_ORIGIN, race_time_ms=0.0)
    steps = round(finish_time_s * 20.0)
    total, discounted, discount = 0.0, 0.0, 1.0
    for step in range(1, steps + 1):
        result = reward.step(_finish_transition(trajectory, step, steps))
        assert result.terminated == (step == steps)
        total += result.reward
        discounted += discount * result.reward
        discount *= 0.9994
    return total, discounted


def test_task_aligned_reward_strictly_ranks_finish_times() -> None:
    trajectory = _trajectory(1_101)
    finish_times = (35.0, 36.0, 37.0, 40.0, 50.0, 55.0)
    returns = [_finish_return(trajectory, finish_time) for finish_time in finish_times]
    expected = [131.25, 129.0, 126.75, 120.0, 97.5, 86.25]
    assert [value[0] for value in returns] == pytest.approx(expected)
    assert all(left[0] > right[0] for left, right in pairwise(returns))
    assert all(left[1] > right[1] for left, right in pairwise(returns))


def _rollout_position(trajectory: np.ndarray, step: int, spec: _RolloutSpec) -> np.ndarray:
    steps = round(spec.duration_s * 20.0)
    index = round(step * spec.progress * (len(trajectory) - 1) / steps)
    if step == steps and spec.outcome is _Outcome.FAILURE:
        return trajectory[index] + np.asarray([0.0, 20.0, 0.0])
    return trajectory[index]


def _rollout(trajectory: np.ndarray, spec: _RolloutSpec) -> float:
    reward = _task_aligned_reward(trajectory)
    reward.reset(trajectory[0], race_time_ms=0.0)
    steps, total = round(spec.duration_s * 20.0), 0.0
    for step in range(1, steps + 1):
        position = _rollout_position(trajectory, step, spec)
        finished = spec.outcome is _Outcome.FINISH and step == steps
        state = _FinishState.FINISHED if finished else _FinishState.RUNNING
        result = reward.step(_stationary_transition(position, step * 50.0, state))
        total += result.reward
    return total


def test_slowest_finish_is_better_than_a_near_complete_failure() -> None:
    trajectory = _trajectory(1_001)
    finish = _RolloutSpec(1.0, 55.0, _Outcome.FINISH)
    failure = _RolloutSpec(0.999, 35.0, _Outcome.FAILURE)
    slowest_finish, near_complete_failure = (
        _rollout(trajectory, finish),
        _rollout(trajectory, failure),
    )
    assert slowest_finish == pytest.approx(86.25)
    assert near_complete_failure < 40.0
    assert slowest_finish > near_complete_failure
