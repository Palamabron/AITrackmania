"""Release contracts for the offline boundary lidar geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.trackmania._lidar_fixtures import _asset
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest
from trackmaniarl.trackmania.lidar_feature_setup import LidarFeatureConfig


def _geometry_from_boundaries(
    tmp_path: Path, left: np.ndarray, right: np.ndarray
) -> BoundaryGeometry:
    left_path, right_path = tmp_path / "left-test.npy", tmp_path / "right-test.npy"
    np.save(left_path, left)
    np.save(right_path, right)
    map_path = tmp_path / "trackmaniarl-test.Map.Gbx"
    map_path.write_bytes(b"trackmaniarl-test-map")
    request = GeometryBuildRequest(
        tmp_path / "test.npz",
        left_path,
        right_path,
        "trackmaniarl-test",
        map_path,
        spacing_m=1.0,
        lookahead_points=0,
    )
    return BoundaryGeometry(build_geometry_asset(request))


def _raw_observation() -> np.ndarray:
    observation = np.zeros(33, dtype=np.float32)
    observation[4:7] = [0, 0, 0]
    observation[10:13] = [1, 0, 0]
    observation[7], observation[16], observation[17] = 20_000.0, 40_000.0, 5_000.0
    observation[30] = -0.5
    return observation


def _pace_reference(tmp_path: Path, geometry: BoundaryGeometry) -> tuple[np.ndarray, Path]:
    frames = np.zeros((geometry.recorded_count, 33), dtype=np.float32)
    frames[:, 3] = np.linspace(0.0, 5_000.0, geometry.recorded_count)
    frames[:, 4:7] = geometry.racing_line
    frames[:, 16] = np.linspace(20_000.0, 40_000.0, geometry.recorded_count)
    frames[-1, 2] = 1.0
    path = tmp_path / "pace.npz"
    np.savez_compressed(
        path,
        map_uid=np.asarray(geometry.map_uid),
        geometry_sha256=np.asarray(geometry.sha256),
        frames=frames,
        finish_time_s=np.asarray(5.0),
    )
    return frames, path


def _pace_pipeline(asset: Path, pace: Path) -> LidarFeaturePipeline:
    return LidarFeaturePipeline(
        LidarFeatureConfig(
            asset,
            expected_map_uid="trackmaniarl-test",
            include_track_relative=True,
            use_racing_line=True,
            pace_reference_path=pace,
            include_racing_line_channels=True,
            include_finish_channels=True,
            include_dynamics=True,
            include_goal_features=True,
        )
    )


def test_geometry_asset_binds_uid_and_rejects_mismatch(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    geometry = BoundaryGeometry(asset, expected_map_uid="trackmaniarl-test")
    assert geometry.sha256
    with pytest.raises(ValueError, match="UID"):
        BoundaryGeometry(asset, expected_map_uid="other")
    with np.load(asset, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    for field in ("map_sha256", "recorded_count"):
        incomplete = tmp_path / f"missing-{field}.npz"
        np.savez_compressed(
            incomplete, **{key: value for key, value in values.items() if key != field}
        )
        with pytest.raises(ValueError, match="missing keys"):
            BoundaryGeometry(incomplete)


def test_geometry_pairing_stays_on_track_across_parallel_sections(tmp_path: Path) -> None:
    left = np.asarray([[float(x), 0.0, 0.0] for x in range(60)], dtype=np.float32)
    true_right = np.asarray([[float(x), 0.0, 10.0] for x in range(60)], dtype=np.float32)
    # Closer decoy for the middle of the track, appended far later in the file.
    decoy = np.asarray([[float(x), 0.0, 9.5] for x in range(20, 50)], dtype=np.float32)
    filler = np.asarray([[1000.0, 0.0, 1000.0 + float(i)] for i in range(2000)], dtype=np.float32)
    right = np.concatenate([true_right, filler, decoy], axis=0)
    geometry = _geometry_from_boundaries(tmp_path, left, right)
    assert np.allclose(geometry.right[:, 2], 10.0, atol=0.05)
    steps = np.linalg.norm(np.diff(geometry.center, axis=0), axis=1)
    assert float(steps.max()) < 3.0


def test_racing_line_stays_inside_boundaries_and_cuts_a_corner(tmp_path: Path) -> None:
    left = np.asarray(
        [[float(x), 0.0, 0.0] for x in range(21)] + [[20.0, 0.0, float(z)] for z in range(1, 21)],
        dtype=np.float32,
    )
    right = np.asarray(
        [[float(x), 0.0, 10.0] for x in range(21)] + [[10.0, 0.0, float(z)] for z in range(1, 21)],
        dtype=np.float32,
    )
    geometry = _geometry_from_boundaries(tmp_path, left, right)
    corridor = geometry.right - geometry.left
    fractions = np.sum((geometry.racing_line - geometry.left) * corridor, axis=1) / np.sum(
        np.square(corridor), axis=1
    )

    assert np.all((fractions >= 0.1) & (fractions <= 0.9))
    assert not np.allclose(geometry.racing_line, geometry.reward_center)


def test_open_track_gets_virtual_finish_lookahead(tmp_path: Path) -> None:
    geometry = BoundaryGeometry(_asset(tmp_path, lookahead_points=60))
    assert geometry.recorded_count < len(geometry.center)
    assert len(geometry.center) == geometry.recorded_count + 60
    assert len(geometry.reward_center) == geometry.recorded_count
    steps = np.linalg.norm(np.diff(geometry.center[geometry.recorded_count - 1 :], axis=0), axis=1)
    assert np.allclose(steps, geometry.spacing_m, atol=1e-3)


def test_lidar_pipeline_validates_schema_and_builds_masked_local_observation(
    tmp_path: Path,
) -> None:
    pipeline = LidarFeaturePipeline(
        LidarFeatureConfig(_asset(tmp_path), expected_map_uid="trackmaniarl-test")
    )
    observation = _raw_observation()
    output = pipeline.transform_observation(observation)
    assert output["lidar"].shape == (4, 60)
    assert output["lidar_mask"].shape == (60,)
    assert output["telemetry"].shape == (20,)
    assert torch.allclose(output["telemetry"][[4, 7, 8, 17]], torch.tensor([0.25, 0.5, 0.5, -0.5]))
    assert bool(output["lidar_mask"].all())
    assert pipeline.transform_observation(output)["lidar"].shape == (4, 60)
    with pytest.raises(ValueError, match="33 fields"):
        pipeline.transform_observation(np.zeros(32, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.transform_observation(np.full(33, np.nan, dtype=np.float32))


def test_lidar_exposes_racing_line_pace_dynamics_and_finish_gate(tmp_path: Path) -> None:
    asset = _asset(tmp_path, lookahead_points=60)
    geometry = BoundaryGeometry(asset)
    frames, pace = _pace_reference(tmp_path, geometry)
    pipeline = _pace_pipeline(asset, pace)
    observation = frames[-1].copy()
    observation[10] = 1.0
    prepared = pipeline.transform_observation(observation)

    assert prepared["lidar"].shape == (8, 60)
    assert prepared["telemetry"].shape == (49,)
    assert prepared["lidar"][6].max() > 0.8
    assert prepared["lidar"][7].max() == pytest.approx(1.0)
    assert prepared["telemetry"][27:31].tolist() == pytest.approx([0.5] * 4, abs=0.03)
    assert prepared["telemetry"][-1] == pytest.approx(1.0)
