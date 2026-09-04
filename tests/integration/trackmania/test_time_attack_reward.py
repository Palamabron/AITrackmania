"""Release contracts for time-attack reward."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.reward import RewardResult, TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import TransitionInput

_ORIGIN = np.zeros(3, dtype=np.float32)
_SHORT_TRAJECTORY = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)


class _FinishState(Enum):
    RUNNING = auto()
    FINISHED = auto()


def _transition(
    position: np.ndarray, race_time_ms: float, state: _FinishState = _FinishState.RUNNING
) -> TransitionInput:
    return TransitionInput(
        position, state is _FinishState.FINISHED, np.zeros(3), race_time_ms, False, None
    )


def _finish_at(reward: TrajectoryReward, race_time_ms: float) -> RewardResult:
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    return reward.step(_transition(_SHORT_TRAJECTORY[-1], race_time_ms, _FinishState.FINISHED))


def _time_attack_reward(config: RewardConfig) -> TrajectoryReward:
    return TrajectoryReward(_SHORT_TRAJECTORY, config)


def test_environment_config_forwards_the_finish_threshold() -> None:
    config = TrackmaniaEnvironmentConfig(geometry_path=Path("geometry.npz"), finish_progress=0.9)
    assert config.reward_config().finish_progress == 0.9


def test_time_attack_reward_is_bounded_and_ranks_finish_times() -> None:
    config = RewardConfig(
        minimum_finish_steps=1,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        finish_reward=1.0,
        potential_progress_weight=0.0,
        time_attack_target_s=65.0,
        time_attack_linear_scale=1.0,
    )
    reward = _time_attack_reward(config)
    slow, target, fast = (_finish_at(reward, time) for time in (59_400.0, 36_000.0, 35_000.0))
    assert fast.time_attack_terminal_reward > target.time_attack_terminal_reward
    assert target.time_attack_terminal_reward > slow.time_attack_terminal_reward
    assert (slow.reward, target.reward, fast.reward) == pytest.approx((6.6, 30.0, 31.0))
    assert slow.terminal_reward == target.terminal_reward == fast.terminal_reward == 1.0


def _velocity_result(reward: TrajectoryReward, velocity_x: float) -> RewardResult:
    reward.reset(_ORIGIN, race_time_ms=0.0)
    transition = TransitionInput(_ORIGIN, False, np.array([velocity_x, 0, 0]), 1_000.0, False, None)
    return reward.step(transition)


def test_projected_velocity_reward_clips_telemetry_outliers() -> None:
    config = RewardConfig(
        no_progress_steps=100,
        slow_progress_window_steps=100,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        potential_progress_weight=0.0,
        max_projected_speed_mps=100.0,
        velocity_to_mps_scale=1.0,
        projected_velocity_scale=1.0,
    )
    reward = _time_attack_reward(config)
    forward, reverse = _velocity_result(reward, 1_000.0), _velocity_result(reward, -1_000.0)
    assert forward.projected_velocity_mps == forward.projected_velocity_reward == 100.0
    assert reverse.projected_velocity_mps == reverse.projected_velocity_reward == -100.0


def _second_step(distance_m: float) -> RewardResult:
    trajectory = np.array([[0, 0, 0], [distance_m, 0, 0], [10, 0, 0]], dtype=np.float32)
    config = RewardConfig(
        no_progress_steps=100,
        slow_progress_window_steps=2,
        minimum_progress_per_window_m=5.0,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        potential_progress_weight=0.0,
    )
    reward = TrajectoryReward(trajectory, config)
    reward.reset(_ORIGIN, race_time_ms=0.0)
    reward.step(_transition(_ORIGIN, 500.0))
    return reward.step(_transition(trajectory[1], 1_000.0))


def test_slow_progress_threshold_tolerates_geometry_rounding() -> None:
    rounded, genuinely_slow = _second_step(4.996), _second_step(4.9)
    assert rounded.reason is None
    assert rounded.window_progress_m == pytest.approx(4.996)
    assert genuinely_slow.reason == "slow_progress"


def test_race_time_limit_is_a_penalized_terminal_transition() -> None:
    trajectory = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
    config = RewardConfig(
        no_progress_steps=100,
        slow_progress_window_steps=100,
        terminal_failure_penalty=7.0,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        finish_reward=0.0,
        potential_progress_weight=0.0,
        maximum_race_time_s=1.0,
    )
    reward = TrajectoryReward(trajectory, config)
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    result = reward.step(_transition(np.array([1, 0, 0]), 1_000.0))
    assert result.terminated
    assert result.reason == "time_limit"
    assert result.terminal_reward == -7.0


def _stall_reward() -> TrajectoryReward:
    trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    return TrajectoryReward(
        trajectory,
        RewardConfig(no_progress_steps=3, slow_progress_window_steps=10, minimum_finish_steps=1),
    )


def test_trajectory_reward_terminates_after_progress_stalls() -> None:
    reward = _stall_reward()
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    results = [reward.step(_transition(_ORIGIN, time)) for time in (100.0, 200.0, 300.0)]
    assert [result.reason for result in results] == [None, None, "no_progress"]


def _collision_results() -> tuple[RewardResult, RewardResult, RewardResult]:
    config = RewardConfig(
        collision_penalty=0.5,
        collision_cooldown_s=2.0,
        time_penalty_per_second=0.0,
        potential_progress_weight=0.0,
    )
    reward = TrajectoryReward(_SHORT_TRAJECTORY, config)
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    transitions = [
        TransitionInput(_ORIGIN, False, _ORIGIN, time, True, None)
        for time in (100.0, 1_100.0, 2_100.0)
    ]
    first, repeated, later = (reward.step(transition) for transition in transitions)
    return first, repeated, later


def test_trajectory_reward_debounces_collision_penalties_by_race_time() -> None:
    first, repeated, later = _collision_results()
    assert first.collision_detected
    assert first.collided
    assert first.collision_reward == pytest.approx(-0.5)
    assert repeated.collision_detected
    assert not repeated.collided
    assert repeated.collision_reward == 0.0
    assert later.collision_detected
    assert later.collided
    assert later.collision_reward == pytest.approx(-0.5)


def test_trajectory_reward_does_not_skip_ahead_at_a_track_crossover() -> None:
    # Point 4 passes close to point 1 after a distant part of the lap.
    trajectory = np.array(
        [[0, 0, 0], [10, 0, 0], [20, 0, 0], [20, 0, 10], [10, 0, 1], [0, 0, 1]],
        dtype=np.float32,
    )
    config = RewardConfig(nearest_forward_points=2, minimum_finish_steps=1)
    reward = TrajectoryReward(trajectory, config)
    reward.reset(_ORIGIN, velocity=_ORIGIN, race_time_ms=0.0)
    result = reward.step(_transition(np.array([10, 0, 0.9]), 100.0))
    assert result.pbrs_reward > 0.0
    assert reward._index == 1
