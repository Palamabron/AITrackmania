from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trackmaniarl.trackmania.actions import (
    continuous_control_to_discrete_index,
    select_brake_tap_actions,
)
from trackmaniarl.trackmania.behavior_cloning import (
    RECOVERY_DATASET_FORMAT,
    load_behavior_cloning_recovery,
)
from trackmaniarl.trackmania.demonstrations import Demonstration
from trackmaniarl.trackmania.guidance import TrajectoryTrackingDemonstrationPolicy
from trackmaniarl.trackmania.synthetic_recovery import (
    SyntheticRecoveryConfig,
    generate_synthetic_recovery,
)

ACTION_IDS = (0, 1, 3, 39, 72, 73, 75)


def _demonstration() -> Demonstration:
    frames = np.zeros((7, 33), dtype=np.float32)
    frames[:, 3] = np.arange(10.0, 80.0, 10.0)
    frames[:, 4] = np.arange(7, dtype=np.float32)
    frames[:, 7] = 20.0
    frames[:, 10] = 1.0
    frames[:, 14] = 1.0
    frames[:, 16] = 20.0
    frames[:, 18] = 2.0
    frames[:, 29] = 1.0
    frames[:, 31] = 1.0
    frames[-1, 2] = 1.0
    controls = np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (6, 1))
    actions = np.full(6, 39, dtype=np.int64)
    return Demonstration(
        map_uid="map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=actions,
        controls=controls,
        finish_time_s=0.07,
        decision_interval_ms=10.0,
    )


def test_synthetic_recovery_is_deterministic_and_keeps_monotonic_episodes() -> None:
    config = SyntheticRecoveryConfig(sample_stride=2)

    first = generate_synthetic_recovery(_demonstration(), ACTION_IDS, config)
    second = generate_synthetic_recovery(_demonstration(), ACTION_IDS, config)

    assert len(first.frames) == 33
    assert np.array_equal(first.frames, second.frames)
    assert np.array_equal(first.labels, second.labels)
    assert np.count_nonzero(first.episode_starts) == 11
    assert np.flatnonzero(first.episode_starts).tolist() == list(range(0, 33, 3))
    assert np.count_nonzero(first.interventions) == 30


def test_synthetic_recovery_keeps_raw_frame_geometry_and_speed_consistent() -> None:
    dataset = generate_synthetic_recovery(
        _demonstration(),
        ACTION_IDS,
        SyntheticRecoveryConfig(sample_stride=100),
    )
    episode_length = len(dataset.frames) // 11
    reference, right_offset, left_heading, left_velocity = dataset.frames[
        [0, episode_length, 2 * episode_length, 3 * episode_length]
    ]

    assert reference[4:17] == pytest.approx(_demonstration().frames[0, 4:17])
    assert right_offset[6] == pytest.approx(0.55)
    assert np.linalg.norm(left_heading[10:13]) == pytest.approx(1.0)
    assert np.dot(left_heading[10:13], left_heading[13:16]) == pytest.approx(0.0, abs=1e-6)
    assert left_velocity[9] == pytest.approx(3.0)
    assert left_velocity[16] == pytest.approx(np.linalg.norm(left_velocity[7:10]))


def test_synthetic_recovery_pd_labels_steer_toward_the_reference() -> None:
    dataset = generate_synthetic_recovery(
        _demonstration(),
        ACTION_IDS,
        SyntheticRecoveryConfig(sample_stride=100),
    )

    assert dataset.labels.tolist() == [3, 6, 6, 3, 6, 3, 2, 2, 3, 2, 3]
    assert dataset.sample_weights[0] == pytest.approx(0.5)
    assert dataset.sample_weights[4] == pytest.approx(3.0)
    assert dataset.state_errors[0] == 0.0
    assert bool(np.all(dataset.state_errors[1:] > 0.0))


def test_synthetic_recovery_labels_match_trajectory_tracker_pd() -> None:
    demonstration = _demonstration()
    dataset = generate_synthetic_recovery(
        demonstration,
        ACTION_IDS,
        SyntheticRecoveryConfig(sample_stride=100),
    )
    _, action_table = select_brake_tap_actions(ACTION_IDS)

    actual = []
    for frame in dataset.frames:
        tracker = TrajectoryTrackingDemonstrationPolicy(
            demonstration.frames[:-1],
            demonstration.controls,
            action_lead_ms=10.0,
        )
        control = tracker.act({"raw_telemetry": frame}, deterministic=True)
        actual.append(continuous_control_to_discrete_index(control, action_table))

    assert actual == dataset.labels.tolist()


class _Pipeline:
    def reset_episode(self) -> None:
        return None

    def transform_observation(self, observation: object) -> dict[str, torch.Tensor]:
        values = np.asarray(observation, dtype=np.float32)
        return {"telemetry": torch.from_numpy(values[:26].copy())}


def test_synthetic_recovery_explicit_save_uses_bc_recovery_v2(tmp_path: Path) -> None:
    dataset = generate_synthetic_recovery(
        _demonstration(),
        ACTION_IDS,
        SyntheticRecoveryConfig(sample_stride=100),
    )

    path = dataset.save(tmp_path / "synthetic")
    laps = load_behavior_cloning_recovery([path], _Pipeline(), ACTION_IDS)
    with np.load(path, allow_pickle=False) as data:
        format_name = str(data["format"].item())

    assert path.exists()
    assert format_name == RECOVERY_DATASET_FORMAT
    assert len(laps) == len(dataset.frames)
    assert all(len(lap.labels) == 1 for lap in laps)
