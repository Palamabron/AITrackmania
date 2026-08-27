from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.pace import (
    PaceFrameRequest,
    ReferencePaceProfile,
    demonstration_guidance_line,
)
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import TransitionInput


def _pace_reward(profile: ReferencePaceProfile) -> TrajectoryReward:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    config = RewardConfig(
        minimum_finish_steps=1,
        pace_profile=profile,
        pace_reward_scale=2.0,
        reward_gamma=0.9994,
    )
    reward = TrajectoryReward(trajectory, config)
    reward.reset(np.asarray([0, 0, 0]), race_time_ms=0.0)
    return reward


def _finished_reward(profile: ReferencePaceProfile) -> TrajectoryReward:
    config = RewardConfig(
        pace_profile=profile,
        pace_reward_scale=2.0,
        minimum_finish_steps=1,
        finish_progress=0.49,
        progress_reward_full_lap=0.0,
        finish_reward=0.0,
        potential_progress_weight=0.0,
    )
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    reward = TrajectoryReward(trajectory, config)
    reward.reset(np.asarray([0, 0, 0]), race_time_ms=0.0)
    return reward


def test_reference_pace_interpolates_monotonic_demo_progress() -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    frames = np.zeros((3, 33), dtype=np.float32)
    frames[:, 3] = [0.0, 1_000.0, 2_000.0]
    frames[:, 4] = [0.0, 1.0, 2.0]
    frames[-1, 2] = 1.0
    frames[:, 16] = [10_000.0, 20_000.0, 30_000.0]

    profile = ReferencePaceProfile.from_frames(PaceFrameRequest(frames, trajectory, 2.0))

    assert profile.reference_times_s == pytest.approx([0.0, 1.0, 2.0])
    assert profile.reference_speeds_mps == pytest.approx([10.0, 20.0, 30.0])
    assert profile.speed_at_index(10) == pytest.approx(30.0)


def test_demonstration_guidance_interpolates_lateral_expert_positions(tmp_path: Path) -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    frames = np.zeros((3, 33), dtype=np.float32)
    frames[:, 3] = [0.0, 1_000.0, 2_000.0]
    frames[:, 4:7] = [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
    path = tmp_path / "guidance.npz"
    np.savez_compressed(
        path,
        map_uid=np.asarray("map"),
        geometry_sha256=np.asarray("geometry"),
        frames=frames,
        finish_time_s=np.asarray(2.0),
    )
    geometry = type("Geometry", (), {"map_uid": "map", "sha256": "geometry"})()

    guidance = demonstration_guidance_line(path, geometry, trajectory)

    assert guidance[:, 0] == pytest.approx([0.0, 1.0, 2.0])
    assert guidance[:, 2] == pytest.approx([2.0, 2.0, 2.0])


def test_pace_reward_penalizes_time_debt_and_rewards_recovery() -> None:
    profile = ReferencePaceProfile(np.asarray([0.0, 1.0, 2.0]))
    reward = _pace_reward(profile)

    behind = reward.step(TransitionInput(np.asarray([1, 0, 0]), False, None, 1_250.0, False, None))
    recovered = reward.step(
        TransitionInput(np.asarray([2, 0, 0]), True, None, 2_100.0, False, None)
    )

    assert behind.time_debt_s == pytest.approx(0.25)
    assert behind.pace_reward == pytest.approx(-0.4997)
    assert recovered.time_debt_s == pytest.approx(0.1)
    assert recovered.pace_reward == pytest.approx(0.5)
    assert behind.pace_reward + 0.9994 * recovered.pace_reward == pytest.approx(0.0)


def test_finished_pace_uses_the_complete_reference_time() -> None:
    profile = ReferencePaceProfile(
        reference_times_s=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        reference_speeds_mps=np.ones(3, dtype=np.float32),
    )
    reward = _finished_reward(profile)

    result = reward.step(TransitionInput(np.asarray([1, 0, 0]), True, None, 2_500.0, False, None))

    assert result.terminated
    assert result.reference_time_s == 2.0
    assert result.time_debt_s == 0.5
    assert result.pace_reward == 0.0
