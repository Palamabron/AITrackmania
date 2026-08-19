from __future__ import annotations

import numpy as np
import pytest

from trackmaniarl.trackmania.pace import ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward


def test_reference_pace_interpolates_monotonic_demo_progress() -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    frames = np.zeros((3, 33), dtype=np.float32)
    frames[:, 3] = [0.0, 1_000.0, 2_000.0]
    frames[:, 4] = [0.0, 1.0, 2.0]
    frames[:, 16] = [10.0, 20.0, 30.0]

    profile = ReferencePaceProfile.from_frames(frames, trajectory, finish_time_s=2.0)

    assert profile.reference_times_s == pytest.approx([0.0, 1.0, 2.0])
    assert profile.reference_speeds_mps == pytest.approx([10.0, 20.0, 30.0])
    assert profile.speed_at_index(10) == pytest.approx(30.0)


def test_pace_reward_penalizes_time_debt_and_rewards_recovery() -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    profile = ReferencePaceProfile(np.asarray([0.0, 1.0, 2.0]))
    reward = TrajectoryReward(
        trajectory,
        minimum_finish_steps=1,
        pace_profile=profile,
        pace_reward_scale=2.0,
    )
    reward.reset(np.asarray([0, 0, 0]), race_time_ms=0.0)

    behind = reward.step(np.asarray([1, 0, 0]), finish_ui_active=False, race_time_ms=1_250.0)
    recovered = reward.step(np.asarray([2, 0, 0]), finish_ui_active=True, race_time_ms=2_100.0)

    assert behind.time_debt_s == pytest.approx(0.25)
    assert behind.pace_reward == pytest.approx(-0.5)
    assert recovered.time_debt_s == pytest.approx(0.1)
    assert recovered.pace_reward == pytest.approx(0.3)
    assert behind.pace_reward + recovered.pace_reward == pytest.approx(-0.2)


def test_pace_reward_requires_profile() -> None:
    with pytest.raises(ValueError, match="pace_reward_scale requires a pace_profile"):
        TrajectoryReward(
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            pace_reward_scale=1.0,
        )
