from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tmrl.core.data import Transition
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from tmrl.trackmania.demonstrations import (
    Demonstration,
    demonstration_transitions,
    load_demonstration,
    record_demonstration,
    save_demonstration,
)
from tmrl.trackmania.environment import TrackmaniaEnvironmentConfig
from tmrl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from tmrl.trackmania.telemetry import TelemetryFrame


class _IdentityPipeline:
    def reset_episode(self) -> None:
        return

    def transform_observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).copy()

    def collate(self, transitions: list[Transition]) -> list[Transition]:
        return transitions


class _TelemetryClient:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)

    def read(self) -> TelemetryFrame:
        return TelemetryFrame(next(self.frames))


def _geometry(tmp_path: Path) -> BoundaryGeometry:
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(101)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    map_path = tmp_path / "map.Map.Gbx"
    map_path.write_bytes(b"map")
    asset = build_geometry_asset(
        tmp_path / "geometry.npz",
        tmp_path / "left.npy",
        tmp_path / "right.npy",
        map_uid="test-map",
        map_path=map_path,
        spacing_m=1.0,
        lookahead_points=0,
    )
    return BoundaryGeometry(asset, expected_map_uid="test-map")


def _config(
    geometry: BoundaryGeometry, *, action_repeat_frames: int = 2
) -> TrackmaniaEnvironmentConfig:
    return TrackmaniaEnvironmentConfig(
        geometry_path=geometry.path,
        expected_map_uid=geometry.map_uid,
        action_repeat_frames=action_repeat_frames,
        velocity_to_mps_scale=1.0,
        minimum_finish_steps=50,
        no_progress_steps=100,
        slow_progress_window_steps=80,
        minimum_progress_per_window_m=1.0,
    )


def _frames(count: int = 61) -> np.ndarray:
    frames = np.zeros((count, 33), dtype=np.float32)
    frames[:, 3] = np.linspace(0.0, 36_000.0, count)
    frames[:, 4] = np.linspace(0.0, 100.0, count)
    frames[:, 7] = 40.0
    frames[:, 10] = 1.0
    frames[:, 31] = 1.0
    frames[-1, 2] = 1.0
    return frames


def _demonstration(geometry: BoundaryGeometry) -> Demonstration:
    frames = _frames()
    _, table = build_brake_tap_action_table()
    action = continuous_control_to_discrete_index(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32), table
    )
    return Demonstration(
        map_uid=geometry.map_uid,
        geometry_sha256=geometry.sha256,
        action_repeat_frames=2,
        frames=frames,
        actions=np.full(len(frames) - 1, action, dtype=np.int64),
        controls=np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (len(frames) - 1, 1)),
        finish_time_s=36.0,
    )


def test_demonstration_round_trip_and_transition_conversion(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap", _demonstration(geometry))

    loaded = load_demonstration(path)
    transitions = demonstration_transitions(path, _IdentityPipeline(), _config(geometry), geometry)

    assert loaded.finish_time_s == 36.0
    assert len(transitions) == 60
    assert transitions[-1].terminated
    assert transitions[-1].info["is_demo"] is True
    assert transitions[-1].info["sampling/projected_lap_time_s"] == 36.0


def test_demonstration_rejects_action_repeat_mismatch(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap.npz", _demonstration(geometry))

    with pytest.raises(ValueError, match="action repeat"):
        demonstration_transitions(
            path,
            _IdentityPipeline(),
            _config(geometry, action_repeat_frames=4),
            geometry,
        )


def test_recorder_waits_for_restart_and_quantizes_human_control(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    baseline = _frames(1)[0]
    baseline[2] = 0.0
    baseline[3] = 1_000.0
    reset = baseline.copy()
    reset[3] = 0.0
    start = baseline.copy()
    start[3] = 16.0
    start[31] = 1.0
    repeated = start.copy()
    repeated[3] = 32.0
    finish = repeated.copy()
    finish[2] = 1.0
    finish[3] = 48.0

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, start, repeated, finish]),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert demo.frames.shape == (2, 33)
    _, table = build_brake_tap_action_table()
    expected = continuous_control_to_discrete_index(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32), table
    )
    assert demo.actions.tolist() == [expected]
    assert demo.finish_time_s == pytest.approx(0.048)
