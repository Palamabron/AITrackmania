from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.trajectory_stitching import (
    TrajectoryStitchingConfig,
    build_fastest_compatible_trajectory,
)


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


def _demonstration(
    geometry: BoundaryGeometry,
    race_times_ms: list[float],
    *,
    action_repeat_frames: int = 1,
    lateral_offset_m: float = 0.0,
    control_alignment: str = "frame_start",
) -> Demonstration:
    frames = np.zeros((len(race_times_ms), 33), dtype=np.float32)
    frames[:, 3] = race_times_ms
    frames[:, 4] = np.linspace(0.0, 100.0, len(frames))
    frames[:, 6] = lateral_offset_m
    frames[:, 7] = 20.0
    frames[:, 10] = 1.0
    frames[:, 31] = 1.0
    frames[-1, 2] = 1.0
    control = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    _, table = build_brake_tap_action_table()
    action = continuous_control_to_discrete_index(control, table)
    return Demonstration(
        map_uid=geometry.map_uid,
        geometry_sha256=geometry.sha256,
        action_repeat_frames=action_repeat_frames,
        frames=frames,
        actions=np.full(len(frames) - 1, action, dtype=np.int64),
        controls=np.tile(control, (len(frames) - 1, 1)),
        finish_time_s=race_times_ms[-1] / 1_000.0,
        control_alignment=control_alignment,
    )


def _paths(tmp_path: Path, geometry: BoundaryGeometry) -> tuple[Path, Path]:
    fast_start = _demonstration(
        geometry,
        [10, 410, 810, 1_210, 1_610, 2_010, 2_610, 3_210, 3_810, 4_410, 5_010],
    )
    fast_finish = _demonstration(
        geometry,
        [10, 610, 1_210, 1_810, 2_410, 3_010, 3_310, 3_610, 3_910, 4_210, 4_510],
    )
    first = save_demonstration(tmp_path / "fast-start.npz", fast_start)
    second = save_demonstration(tmp_path / "fast-finish.npz", fast_finish)
    return first, second


def test_stitcher_builds_a_faster_state_compatible_lap(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    first, second = _paths(tmp_path, geometry)

    result = build_fastest_compatible_trajectory(
        [first, second],
        geometry,
        TrajectoryStitchingConfig(segment_length_m=10.0),
    )

    assert result.source_paths == (first.resolve(), second.resolve())
    assert result.demonstration.finish_time_s == pytest.approx(3.51)
    assert result.estimated_gain_s == pytest.approx(1.0)
    assert len(result.joins) == 1
    assert result.joins[0].progress_fraction == pytest.approx(0.5)
    assert np.all(np.diff(result.demonstration.frames[:, 3]) > 0.0)
    assert len(result.demonstration.actions) == len(result.demonstration.frames) - 1


def test_stitcher_never_crosses_timing_contracts(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    first, second = _paths(tmp_path, geometry)
    incompatible = _demonstration(
        geometry,
        [10, 610, 1_210, 1_810, 2_410, 3_010, 3_310, 3_610, 3_910, 4_210, 4_510],
        action_repeat_frames=2,
    )
    save_demonstration(second, incompatible)

    result = build_fastest_compatible_trajectory([first, second], geometry)

    assert result.joins == ()
    assert result.demonstration.finish_time_s == pytest.approx(4.51)


def test_stitcher_never_crosses_control_alignment_contracts(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    first, second = _paths(tmp_path, geometry)
    incompatible = _demonstration(
        geometry,
        [10, 610, 1_210, 1_810, 2_410, 3_010, 3_310, 3_610, 3_910, 4_210, 4_510],
        control_alignment="transition_end",
    )
    save_demonstration(second, incompatible)

    result = build_fastest_compatible_trajectory([first, second], geometry)

    assert result.joins == ()
    assert result.demonstration.control_alignment == "transition_end"


def test_stitcher_rejects_a_spatially_discontinuous_join(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    first, _ = _paths(tmp_path, geometry)
    displaced = _demonstration(
        geometry,
        [10, 610, 1_210, 1_810, 2_410, 3_010, 3_310, 3_610, 3_910, 4_210, 4_510],
        lateral_offset_m=2.0,
    )
    second = save_demonstration(tmp_path / "displaced.npz", displaced)

    result = build_fastest_compatible_trajectory([first, second], geometry)

    assert result.joins == ()
    assert result.demonstration.finish_time_s == pytest.approx(4.51)
