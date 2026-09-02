from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.experiments.graph_iqn_v2 import (
    PHYSICS_V2_LAYOUT,
    BoundaryGraphFeaturePipelineV2,
    TrackGnnSimbaEncoderV2,
)
from trackmaniarl.trackmania.geometry import GEOMETRY_ASSET_VERSION


def _dense_geometry(path: Path) -> Path:
    positions = np.zeros((300, 3), dtype=np.float32)
    positions[:, 0] = np.arange(300, dtype=np.float32) * 0.5
    np.savez_compressed(
        path,
        version=np.array(GEOMETRY_ASSET_VERSION),
        map_uid=np.array("test-map"),
        map_sha256=np.array("test-map-hash"),
        left=positions - np.array([0.0, 0.0, 1.0], dtype=np.float32),
        center=positions,
        right=positions + np.array([0.0, 0.0, 1.0], dtype=np.float32),
        spacing_m=np.array(0.5),
        recorded_count=np.array(len(positions)),
    )
    return path


def _frame(spec: dict[str, Any]) -> np.ndarray:
    """Telemetry frame from a small spec: t (ms), x, speed, heading, steer, gas, brake, skid."""

    heading = spec.get("heading", (1.0, 0.0))
    speed = float(spec["speed"])
    values = np.zeros(33, dtype=np.float32)
    values[3], values[4] = spec["t"], spec["x"]
    values[7], values[9] = speed * heading[0], speed * heading[1]
    values[10], values[12] = heading
    values[14], values[16], values[18] = 1.0, speed, 3.0
    values[27] = spec.get("skid", 0.0)
    values[30] = spec.get("steer", 0.0)
    values[31] = spec.get("gas", 1.0)
    values[32] = spec.get("brake", 0.0)
    return values


def _named(physics: torch.Tensor) -> dict[str, float]:
    return dict(zip(PHYSICS_V2_LAYOUT, physics.tolist(), strict=True))


@pytest.fixture
def pipeline(tmp_path: Path) -> BoundaryGraphFeaturePipelineV2:
    result = BoundaryGraphFeaturePipelineV2(_dense_geometry(tmp_path / "g.npz"), "test-map")
    result.reset_episode()
    return result


def test_v2_shapes_and_first_frame(pipeline: BoundaryGraphFeaturePipelineV2) -> None:
    first = pipeline.transform_observation(_frame({"t": 0.0, "x": 10.0, "speed": 50.0}))

    assert first["physics"].shape == (60,)
    assert first["track"].shape == (3, 88)
    assert _named(first["physics"])["acceleration"] == 0.0
    assert _named(first["physics"])["yaw_rate"] == 0.0
    assert torch.isfinite(first["track"]).all()


_BRAKING_SPEC = {
    "t": 50.0,
    "x": 12.5,
    "speed": 52.0,
    "steer": 1.0,
    "gas": 0.0,
    "brake": 1.0,
    "skid": 2.0,
}


def _second_frame(pipeline: BoundaryGraphFeaturePipelineV2) -> dict[str, float]:
    pipeline.transform_observation(_frame({"t": 0.0, "x": 10.0, "speed": 50.0}))
    return _named(pipeline.transform_observation(_frame(_BRAKING_SPEC))["physics"])


def test_v2_straight_line_kinematics(pipeline: BoundaryGraphFeaturePipelineV2) -> None:
    physics = _second_frame(pipeline)

    assert physics["speed"] == pytest.approx(0.52)
    assert physics["forward_velocity"] == pytest.approx(0.52)
    assert physics["lateral_velocity"] == pytest.approx(0.0, abs=1e-6)
    assert physics["heading_cos"] == pytest.approx(1.0)
    assert physics["heading_sin"] == pytest.approx(0.0, abs=1e-6)
    assert physics["yaw_rate"] == 0.0
    assert physics["acceleration"] == pytest.approx((2.0 / 0.05) / 50.0)


def test_v2_control_echo_and_skidding(pipeline: BoundaryGraphFeaturePipelineV2) -> None:
    physics = _second_frame(pipeline)

    assert (physics["input_gas"], physics["input_brake"], physics["input_steer"]) == (0.0, 1.0, 1.0)
    assert physics["skidding_wheels"] == 0.5
    assert physics["gear"] == pytest.approx(0.6)


def test_v2_yaw_rate_and_lateral_velocity_follow_heading(
    pipeline: BoundaryGraphFeaturePipelineV2,
) -> None:
    pipeline.transform_observation(_frame({"t": 0.0, "x": 10.0, "speed": 40.0}))
    angle = 0.1
    rotated = (float(np.cos(angle)), float(np.sin(angle)))
    frame = _frame({"t": 50.0, "x": 12.0, "speed": 40.0, "heading": rotated})
    frame[7], frame[9] = 40.0, 0.0  # velocity still along +x: the car is sliding
    physics = _named(pipeline.transform_observation(frame)["physics"])

    expected_yaw_rate = (np.arctan2(rotated[0], rotated[1]) - np.arctan2(1.0, 0.0)) / 0.05
    assert physics["yaw_rate"] == pytest.approx(expected_yaw_rate / 3.0, abs=1e-4)
    assert physics["forward_velocity"] == pytest.approx(40.0 * np.cos(angle) / 100.0, abs=1e-5)
    assert abs(physics["lateral_velocity"]) == pytest.approx(40.0 * np.sin(angle) / 20.0, abs=1e-5)
    assert physics["heading_cos"] == pytest.approx(np.cos(angle), abs=1e-5)
    assert abs(physics["heading_sin"]) == pytest.approx(np.sin(angle), abs=1e-5)


def test_v2_prepared_observation_keeps_control_echo(
    pipeline: BoundaryGraphFeaturePipelineV2,
) -> None:
    prepared = pipeline.transform_observation(
        {"physics": np.ones(60, dtype=np.float32), "track": np.zeros((3, 88), dtype=np.float32)}
    )
    assert torch.equal(prepared["physics"], torch.ones(60))


def test_v2_encoder_shape_and_sensitivity_to_control_echo() -> None:
    encoder = TrackGnnSimbaEncoderV2()
    track = torch.randn(2, 3, 88)
    physics = torch.zeros(2, 60)
    echoed = physics.clone()
    echoed[:, PHYSICS_V2_LAYOUT.index("input_gas")] = 1.0
    echoed[:, PHYSICS_V2_LAYOUT.index("input_steer")] = -1.0

    baseline = encoder({"track": track, "physics": physics})
    with_echo = encoder({"track": track, "physics": echoed})

    assert baseline.shape == (2, 192)
    assert not torch.allclose(baseline, with_echo)
