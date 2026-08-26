from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.pace import ReferencePaceProfile, demonstration_guidance_line
from trackmaniarl.trackmania.reward import TrajectoryReward


def test_reference_pace_interpolates_monotonic_demo_progress() -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    frames = np.zeros((3, 33), dtype=np.float32)
    frames[:, 3] = [0.0, 1_000.0, 2_000.0]
    frames[:, 4] = [0.0, 1.0, 2.0]
    frames[-1, 2] = 1.0
    frames[:, 16] = [10_000.0, 20_000.0, 30_000.0]

    profile = ReferencePaceProfile.from_frames(frames, trajectory, finish_time_s=2.0)

    assert profile.reference_times_s == pytest.approx([0.0, 1.0, 2.0])
    assert profile.reference_speeds_mps == pytest.approx([10.0, 20.0, 30.0])
    assert profile.speed_at_index(10) == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("race_times_ms", "finish_time_s", "finish_flags", "message"),
    [
        ([0.0, 1_000.0, 1_000.0], 1.0, [0.0, 0.0, 1.0], "race times"),
        ([0.0, 1_000.0, 2_000.0], 1.5, [0.0, 0.0, 1.0], "finish time"),
        ([0.0, 1_000.0, 2_000.0], 2.0, [0.0, 0.0, 0.0], "finish frame"),
    ],
)
def test_reference_pace_rejects_incomplete_timing_metadata(
    race_times_ms: list[float],
    finish_time_s: float,
    finish_flags: list[float],
    message: str,
) -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    frames = np.zeros((3, 33), dtype=np.float32)
    frames[:, 3] = race_times_ms
    frames[:, 4] = [0.0, 1.0, 2.0]
    frames[:, 2] = finish_flags

    with pytest.raises(ValueError, match=message):
        ReferencePaceProfile.from_frames(frames, trajectory, finish_time_s=finish_time_s)


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


@pytest.mark.parametrize(
    ("map_uid", "geometry_sha256", "message"),
    [("wrong", "geometry", "map UID"), ("map", "wrong", "geometry hash")],
)
def test_demonstration_guidance_validates_geometry_identity(
    tmp_path: Path,
    map_uid: str,
    geometry_sha256: str,
    message: str,
) -> None:
    frames = np.zeros((2, 33), dtype=np.float32)
    frames[:, 4] = [0.0, 1.0]
    path = tmp_path / "guidance.npz"
    np.savez_compressed(
        path,
        map_uid=np.asarray("map"),
        geometry_sha256=np.asarray("geometry"),
        frames=frames,
        finish_time_s=np.asarray(1.0),
    )
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    geometry = type("Geometry", (), {"map_uid": map_uid, "sha256": geometry_sha256})()

    with pytest.raises(ValueError, match=message):
        demonstration_guidance_line(path, geometry, trajectory)


def test_demonstration_guidance_rejects_non_finite_frames(tmp_path: Path) -> None:
    frames = np.zeros((2, 33), dtype=np.float32)
    frames[:, 4] = [0.0, np.nan]
    path = tmp_path / "guidance.npz"

    np.savez_compressed(
        path,
        map_uid=np.asarray("map"),
        geometry_sha256=np.asarray("geometry"),
        frames=frames,
        finish_time_s=np.asarray(1.0),
    )
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    geometry = type("Geometry", (), {"map_uid": "map", "sha256": "geometry"})()
    with pytest.raises(ValueError, match="finite"):
        demonstration_guidance_line(path, geometry, trajectory)


def test_pace_reward_penalizes_time_debt_and_rewards_recovery() -> None:
    trajectory = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    profile = ReferencePaceProfile(np.asarray([0.0, 1.0, 2.0]))
    reward = TrajectoryReward(
        trajectory,
        minimum_finish_steps=1,
        pace_profile=profile,
        pace_reward_scale=2.0,
        reward_gamma=0.9994,
    )
    reward.reset(np.asarray([0, 0, 0]), race_time_ms=0.0)

    behind = reward.step(np.asarray([1, 0, 0]), finish_ui_active=False, race_time_ms=1_250.0)
    recovered = reward.step(np.asarray([2, 0, 0]), finish_ui_active=True, race_time_ms=2_100.0)

    assert behind.time_debt_s == pytest.approx(0.25)
    assert behind.pace_reward == pytest.approx(-0.4997)
    assert recovered.time_debt_s == pytest.approx(0.1)
    assert recovered.pace_reward == pytest.approx(0.5)
    assert behind.pace_reward + 0.9994 * recovered.pace_reward == pytest.approx(0.0)


def test_pace_reward_requires_profile() -> None:
    with pytest.raises(ValueError, match="pace_reward_scale requires a pace_profile"):
        TrajectoryReward(
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            pace_reward_scale=1.0,
        )


def test_finished_pace_uses_the_complete_reference_time() -> None:
    profile = ReferencePaceProfile(
        reference_times_s=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        reference_speeds_mps=np.ones(3, dtype=np.float32),
    )
    reward = TrajectoryReward(
        np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        pace_profile=profile,
        pace_reward_scale=2.0,
        minimum_finish_steps=1,
        finish_progress=0.49,
        progress_reward_full_lap=0.0,
        finish_reward=0.0,
        potential_progress_weight=0.0,
    )
    reward.reset(np.asarray([0, 0, 0]), race_time_ms=0.0)

    result = reward.step(
        np.asarray([1, 0, 0]),
        finish_ui_active=True,
        race_time_ms=2_500.0,
    )

    assert result.terminated
    assert result.reference_time_s == 2.0
    assert result.time_debt_s == 0.5
    assert result.pace_reward == 0.0
