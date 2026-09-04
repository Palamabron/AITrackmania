from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest
from trackmaniarl.trackmania.telemetry import TelemetryFrame


class _IdentityPipeline:
    def reset_episode(self) -> None:
        return

    def transform_observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).copy()

    def collate(self, transitions: list[Transition]) -> list[Transition]:
        return transitions


class _TelemetryClient:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.cursor = 0

    def read(self) -> TelemetryFrame:
        frame = self.frames[min(self.cursor, len(self.frames) - 1)]
        self.cursor += 1
        return TelemetryFrame(frame)

    def read_next(self) -> TelemetryFrame:
        return self.read()


class _EventTelemetryClient:
    def __init__(self, events: list[np.ndarray | TimeoutError]) -> None:
        self.events = list(events)
        self.cursor = 0

    def read(self) -> TelemetryFrame:
        event = self.events[min(self.cursor, len(self.events) - 1)]
        self.cursor += 1
        if isinstance(event, TimeoutError):
            raise event
        return TelemetryFrame(event)

    def read_next(self) -> TelemetryFrame:
        return self.read()


def _geometry(tmp_path: Path) -> BoundaryGeometry:
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(101)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    map_path = tmp_path / "map.Map.Gbx"
    map_path.write_bytes(b"map")
    asset = build_geometry_asset(
        GeometryBuildRequest(
            tmp_path / "geometry.npz",
            tmp_path / "left.npy",
            tmp_path / "right.npy",
            "test-map",
            map_path,
            spacing_m=1.0,
            lookahead_points=0,
        )
    )
    return BoundaryGeometry(asset, expected_map_uid="test-map")


def _config(
    geometry: BoundaryGeometry,
    *,
    action_repeat_frames: int = 2,
) -> TrackmaniaEnvironmentConfig:
    return TrackmaniaEnvironmentConfig(
        geometry_path=geometry.path,
        expected_map_uid=geometry.map_uid,
        action_repeat_frames=action_repeat_frames,
        decision_interval_ms=None,
        start_timeout_s=15.0,
        start_poll_s=0.0,
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


def _lap_frames(*race_times_ms: float) -> list[np.ndarray]:
    """Frames for one full-throttle run: an idle frame, a reset, then the driven lap."""

    baseline = _frames(1)[0]
    baseline[2] = 0.0
    baseline[31] = 1.0
    idle = baseline.copy()
    idle[3] = 1_000.0
    reset = baseline.copy()
    reset[3] = 0.0
    driven = []
    for race_time_ms in race_times_ms:
        frame = baseline.copy()
        frame[3] = race_time_ms
        driven.append(frame)
    driven[-1][2] = 1.0
    return [idle, reset, *driven]
