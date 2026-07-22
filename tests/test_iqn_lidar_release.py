"""Release contracts for the IQN plus offline boundary lidar baseline."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from tmrl.algorithms import ImplicitQuantileQLearning
from tmrl.core.data import TrainingBatch
from tmrl.models.encoders import TrackGeometryEncoder
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    build_brake_tap_exploration_weights,
)
from tmrl.trackmania.features import LidarFeaturePipeline
from tmrl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from tmrl.trackmania.iqn import LidarIqnModel
from tmrl.trackmania.session import PLUGIN_PROTOCOL_VERSION, OpenPlanetSessionClient


def _asset(tmp_path: Path, *, lookahead_points: int = 60) -> Path:
    # Dense enough that opposite-boundary nearest neighbours stay on-station.
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(0, 11)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    return build_geometry_asset(
        tmp_path / "test-3.npz",
        tmp_path / "left.npy",
        tmp_path / "right.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        lookahead_points=lookahead_points,
    )


def test_geometry_asset_binds_uid_and_rejects_mismatch(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    geometry = BoundaryGeometry(asset, expected_map_uid="test-3")
    assert geometry.sha256
    with pytest.raises(ValueError, match="UID"):
        BoundaryGeometry(asset, expected_map_uid="other")


def test_geometry_asset_rejects_a_degenerate_centerline(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    with np.load(asset, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    payload["center"] = np.zeros_like(payload["center"])
    broken = tmp_path / "broken.npz"
    np.savez_compressed(broken, **payload)

    with pytest.raises(ValueError, match="degenerate"):
        BoundaryGeometry(broken)


def test_geometry_pairs_boundaries_by_location_not_recording_progress(tmp_path: Path) -> None:
    left = np.asarray([[0, 0, -5], [5, 0, -5], [10, 0, -5]], dtype=np.float32)
    # The right recording begins later but covers the same road edge.
    right = np.asarray([[5, 0, 5], [10, 0, 5], [0, 0, 5]], dtype=np.float32)
    np.save(tmp_path / "left-offset.npy", left)
    np.save(tmp_path / "right-offset.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    asset = build_geometry_asset(
        tmp_path / "offset.npz",
        tmp_path / "left-offset.npy",
        tmp_path / "right-offset.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        spacing_m=5.0,
        lookahead_points=0,
    )
    geometry = BoundaryGeometry(asset)
    assert np.allclose(np.linalg.norm(geometry.left - geometry.right, axis=1), 10.0)


def test_geometry_pairing_stays_on_track_across_parallel_sections(tmp_path: Path) -> None:
    left = np.asarray([[float(x), 0.0, 0.0] for x in range(60)], dtype=np.float32)
    true_right = np.asarray([[float(x), 0.0, 10.0] for x in range(60)], dtype=np.float32)
    # Closer decoy for the middle of the track, appended far later in the file.
    decoy = np.asarray([[float(x), 0.0, 9.5] for x in range(20, 50)], dtype=np.float32)
    filler = np.asarray([[1000.0, 0.0, 1000.0 + float(i)] for i in range(2000)], dtype=np.float32)
    right = np.concatenate([true_right, filler, decoy], axis=0)
    np.save(tmp_path / "left-parallel.npy", left)
    np.save(tmp_path / "right-parallel.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    asset = build_geometry_asset(
        tmp_path / "parallel.npz",
        tmp_path / "left-parallel.npy",
        tmp_path / "right-parallel.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        spacing_m=1.0,
        lookahead_points=0,
    )
    geometry = BoundaryGeometry(asset)
    assert np.allclose(geometry.right[:, 2], 10.0, atol=0.05)
    steps = np.linalg.norm(np.diff(geometry.center, axis=0), axis=1)
    assert float(steps.max()) < 3.0


def test_geometry_centerline_spacing_is_uniform_on_bends(tmp_path: Path) -> None:
    left = np.asarray(
        [[float(x), 0.0, 0.0] for x in range(0, 41)]
        + [[40.0, 0.0, float(z)] for z in range(1, 21)],
        dtype=np.float32,
    )
    right = np.asarray(
        [[float(x), 0.0, 10.0] for x in range(0, 41)]
        + [[30.0, 0.0, float(z)] for z in range(1, 21)],
        dtype=np.float32,
    )
    np.save(tmp_path / "left-bend.npy", left)
    np.save(tmp_path / "right-bend.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    asset = build_geometry_asset(
        tmp_path / "bend.npz",
        tmp_path / "left-bend.npy",
        tmp_path / "right-bend.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        spacing_m=2.0,
        lookahead_points=0,
    )
    steps = np.linalg.norm(np.diff(BoundaryGeometry(asset).center, axis=0), axis=1)
    assert float(steps.std()) < 0.05
    assert float(steps.max() - steps.min()) < 0.15
    assert abs(float(steps.mean()) - 2.0) < 0.15


def test_geometry_smoothing_reduces_boundary_jitter(tmp_path: Path) -> None:
    left = np.asarray([[float(x), 0.0, 0.05 * ((-1) ** x)] for x in range(40)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left-jitter.npy", left)
    np.save(tmp_path / "right-jitter.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    raw = build_geometry_asset(
        tmp_path / "raw.npz",
        tmp_path / "left-jitter.npy",
        tmp_path / "right-jitter.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        spacing_m=1.0,
        smooth_window=1,
        lookahead_points=0,
    )
    soft = build_geometry_asset(
        tmp_path / "soft.npz",
        tmp_path / "left-jitter.npy",
        tmp_path / "right-jitter.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        spacing_m=1.0,
        smooth_window=5,
        lookahead_points=0,
    )

    def jitter_energy(path: Path) -> float:
        return float(np.var(BoundaryGeometry(path).left[:, 2]))

    assert jitter_energy(soft) < jitter_energy(raw)


def test_open_track_gets_virtual_finish_lookahead(tmp_path: Path) -> None:
    geometry = BoundaryGeometry(_asset(tmp_path, lookahead_points=60))
    assert geometry.recorded_count < len(geometry.center)
    assert len(geometry.center) == geometry.recorded_count + 60
    assert len(geometry.reward_center) == geometry.recorded_count
    steps = np.linalg.norm(np.diff(geometry.center[geometry.recorded_count - 1 :], axis=0), axis=1)
    assert np.allclose(steps, geometry.spacing_m, atol=1e-3)


def test_closed_lap_does_not_extend_finish(tmp_path: Path) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    left = np.stack(
        [20.0 * np.cos(angles), np.zeros_like(angles), 20.0 * np.sin(angles)], axis=1
    ).astype(np.float32)
    right = np.stack(
        [15.0 * np.cos(angles), np.zeros_like(angles), 15.0 * np.sin(angles)], axis=1
    ).astype(np.float32)
    np.save(tmp_path / "left-loop.npy", left)
    np.save(tmp_path / "right-loop.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    asset = build_geometry_asset(
        tmp_path / "loop.npz",
        tmp_path / "left-loop.npy",
        tmp_path / "right-loop.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
        spacing_m=2.0,
        lookahead_points=60,
    )
    geometry = BoundaryGeometry(asset)
    assert geometry.recorded_count == len(geometry.center)
    gap = float(np.linalg.norm(geometry.center[0] - geometry.center[-1]))
    assert gap < 10.0


def test_lidar_pipeline_validates_schema_and_builds_masked_local_observation(
    tmp_path: Path,
) -> None:
    pipeline = LidarFeaturePipeline(_asset(tmp_path), expected_map_uid="test-3")
    observation = np.zeros(33, dtype=np.float32)
    observation[4:7] = [0, 0, 0]
    observation[10:13] = [1, 0, 0]
    observation[7] = 250.0
    observation[16] = 500.0
    observation[17] = 5_000.0
    observation[30] = -0.5
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


def test_lidar_near_finish_keeps_fresh_lookahead_on_open_track(tmp_path: Path) -> None:
    asset = _asset(tmp_path, lookahead_points=60)
    geometry = BoundaryGeometry(asset)
    pipeline = LidarFeaturePipeline(asset, expected_map_uid="test-3")
    observation = np.zeros(33, dtype=np.float32)
    observation[4:7] = geometry.reward_center[-1]
    observation[10:13] = [1, 0, 0]
    output = pipeline.transform_observation(observation)
    assert bool(output["lidar_mask"].all())
    assert not torch.allclose(output["lidar"][:, 0], output["lidar"][:, 14])


def test_lidar_pipeline_preserves_legacy_right_then_forward_car_frame(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path, lookahead_points=0), expected_map_uid="test-3", max_distance_m=10.0
    )
    observation = np.zeros(33, dtype=np.float32)
    observation[10] = 1.0  # Car points along +X.

    output = pipeline.transform_observation(observation)

    # The 2 m-resampled next left-boundary point is (2, 0, -5): it is 5 m right
    # and 2 m ahead in the established OpenPlanet local-frame convention.
    assert output["lidar"][:2, 0].tolist() == pytest.approx([0.5, 0.2])


def test_iqn_lidar_updates_and_handles_single_structured_observation(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(_asset(tmp_path), expected_map_uid="test-3")
    raw = np.zeros(33, dtype=np.float32)
    raw[10] = 1.0
    single = pipeline.transform_observation(raw)
    observations = {
        key: value.unsqueeze(0).repeat(2, *([1] * value.ndim)) for key, value in single.items()
    }
    batch = TrainingBatch(
        data=observations,
        observations=observations,
        actions=torch.tensor([0, 77]),
        rewards=torch.tensor([1.0, 0.0]),
        next_observations=observations,
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        bootstrap_discounts=torch.full((2,), 0.995),
        transition_ids=[1, 2],
    )
    learner = ImplicitQuantileQLearning(
        LidarIqnModel(cosine_count=8),
        train_quantile_count=8,
        target_quantile_count=8,
        evaluation_quantile_count=8,
        learning_rate=3e-5,
        gradient_clip_norm=1.0,
        exploration_epsilon=1.0,
        exploration_epsilon_final=0.05,
        exploration_epsilon_decay_updates=10,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    metrics, _ = learner.update(batch)
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert isinstance(learner.policy().act(single, deterministic=True), int)
    assert learner._current_epsilon() < 1.0


def test_track_geometry_attention_masks_bfloat16_logits() -> None:
    encoder = TrackGeometryEncoder(channels=4, telemetry_dim=0)
    track = torch.randn(2, 4, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    mask[:, -10:] = False

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(track, mask=mask)

    assert torch.isfinite(output).all()


def test_iqn_action_table_has_all_78_indices_and_brake_taps() -> None:
    count, table = build_brake_tap_action_table()
    assert count == 78
    assert len(table) == 78
    assert sum(float(action[1]) == -1.0 for action in table) == 26


def test_iqn_exploration_weights_favor_throttle_and_straight_actions() -> None:
    _, table = build_brake_tap_action_table()
    weights = build_brake_tap_exploration_weights()

    throttled = sum(
        weight for weight, action in zip(weights, table, strict=True) if action[0] == 1.0
    )
    braking = sum(weight for weight, action in zip(weights, table, strict=True) if action[1] != 0.0)
    center = next(
        weight
        for weight, action in zip(weights, table, strict=True)
        if action.tolist() == [1.0, 0.0, 0.0]
    )
    extreme = next(
        weight
        for weight, action in zip(weights, table, strict=True)
        if action.tolist() == [1.0, 0.0, 1.0]
    )

    assert throttled > braking
    assert center > extreme


def test_session_protocol_verifies_preloaded_map_and_ready_state() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    host, port = server.getsockname()

    commands: list[str] = []

    def serve() -> None:
        for _ in range(2):
            connection, _ = server.accept()
            with connection:
                request = json.loads(connection.recv(4096).decode("utf-8"))
                commands.append(request["command"])
                response = {
                    "status": "ok",
                    "protocol_version": PLUGIN_PROTOCOL_VERSION,
                    "map_uid": "test-3",
                    "ready": "true",
                }
                connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = OpenPlanetSessionClient(host, port, timeout_s=1)
    assert client.verify_loaded_map("test-3").map_uid == "test-3"
    assert client.confirm_ready("test-3").map_uid == "test-3"
    thread.join(timeout=1)
    assert commands == ["verify_loaded_map", "confirm_ready"]
