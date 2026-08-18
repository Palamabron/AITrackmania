"""Release contracts for the IQN plus offline boundary lidar baseline."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trackmaniarl.algorithms import ImplicitQuantileQLearning
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.encoders import TrackGeometryEncoder
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    build_brake_tap_exploration_weights,
)
from trackmaniarl.trackmania.behavior_cloning import LidarBehaviorCloningModel
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.iqn import LidarIqnModel, LidarIqnModelFactory
from trackmaniarl.trackmania.session import PLUGIN_PROTOCOL_VERSION, OpenPlanetSessionClient


def _asset(tmp_path: Path, *, lookahead_points: int = 60) -> Path:
    # Dense enough that opposite-boundary nearest neighbours stay on-station.
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(0, 11)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    return build_geometry_asset(
        tmp_path / "trackmaniarl-test.npz",
        tmp_path / "left.npy",
        tmp_path / "right.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        lookahead_points=lookahead_points,
    )


def test_geometry_asset_binds_uid_and_rejects_mismatch(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    geometry = BoundaryGeometry(asset, expected_map_uid="trackmaniarl-test")
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
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    asset = build_geometry_asset(
        tmp_path / "offset.npz",
        tmp_path / "left-offset.npy",
        tmp_path / "right-offset.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
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
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    asset = build_geometry_asset(
        tmp_path / "parallel.npz",
        tmp_path / "left-parallel.npy",
        tmp_path / "right-parallel.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
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
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    asset = build_geometry_asset(
        tmp_path / "bend.npz",
        tmp_path / "left-bend.npy",
        tmp_path / "right-bend.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        spacing_m=2.0,
        lookahead_points=0,
    )
    steps = np.linalg.norm(np.diff(BoundaryGeometry(asset).center, axis=0), axis=1)
    assert float(steps.std()) < 0.05
    assert float(steps.max() - steps.min()) < 0.15
    assert abs(float(steps.mean()) - 2.0) < 0.15


def test_racing_line_stays_inside_boundaries_and_cuts_a_corner(tmp_path: Path) -> None:
    left = np.asarray(
        [[float(x), 0.0, 0.0] for x in range(21)] + [[20.0, 0.0, float(z)] for z in range(1, 21)],
        dtype=np.float32,
    )
    right = np.asarray(
        [[float(x), 0.0, 10.0] for x in range(21)] + [[10.0, 0.0, float(z)] for z in range(1, 21)],
        dtype=np.float32,
    )
    np.save(tmp_path / "left-racing.npy", left)
    np.save(tmp_path / "right-racing.npy", right)
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    asset = build_geometry_asset(
        tmp_path / "racing.npz",
        tmp_path / "left-racing.npy",
        tmp_path / "right-racing.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        spacing_m=1.0,
        lookahead_points=0,
    )
    geometry = BoundaryGeometry(asset)
    corridor = geometry.right - geometry.left
    fractions = np.sum((geometry.racing_line - geometry.left) * corridor, axis=1) / np.sum(
        np.square(corridor), axis=1
    )

    assert np.all((fractions >= 0.1) & (fractions <= 0.9))
    assert not np.allclose(geometry.racing_line, geometry.reward_center)


def test_geometry_smoothing_reduces_boundary_jitter(tmp_path: Path) -> None:
    left = np.asarray([[float(x), 0.0, 0.05 * ((-1) ** x)] for x in range(40)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left-jitter.npy", left)
    np.save(tmp_path / "right-jitter.npy", right)
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    raw = build_geometry_asset(
        tmp_path / "raw.npz",
        tmp_path / "left-jitter.npy",
        tmp_path / "right-jitter.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
        spacing_m=1.0,
        smooth_window=1,
        lookahead_points=0,
    )
    soft = build_geometry_asset(
        tmp_path / "soft.npz",
        tmp_path / "left-jitter.npy",
        tmp_path / "right-jitter.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
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
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    asset = build_geometry_asset(
        tmp_path / "loop.npz",
        tmp_path / "left-loop.npy",
        tmp_path / "right-loop.npy",
        map_uid="trackmaniarl-test",
        map_path=tmp_path / "trackmaniarl-test.Map.Gbx",
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
    pipeline = LidarFeaturePipeline(_asset(tmp_path), expected_map_uid="trackmaniarl-test")
    observation = np.zeros(33, dtype=np.float32)
    observation[4:7] = [0, 0, 0]
    observation[10:13] = [1, 0, 0]
    observation[7] = 20_000.0
    observation[16] = 40_000.0
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


def test_lidar_pipeline_can_exclude_control_inputs_for_behavior_cloning(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path),
        expected_map_uid="trackmaniarl-test",
        include_track_relative=True,
        include_control_inputs=False,
    )
    observation = np.zeros(33, dtype=np.float32)
    observation[10] = 1.0
    observation[30:33] = [-1.0, 1.0, 1.0]

    output = pipeline.transform_observation(observation)

    assert output["telemetry"].shape == (23,)
    assert not torch.any(output["telemetry"] == -1.0)


def test_lidar_pipeline_can_encode_velocity_in_the_local_car_frame(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path), expected_map_uid="trackmaniarl-test", local_velocity_features=True
    )
    observation = np.zeros(33, dtype=np.float32)
    observation[7:10] = [10_000.0, 2_000.0, 5_000.0]
    observation[10] = 1.0

    prepared = pipeline.transform_observation(observation)

    assert prepared["telemetry"][4:7].tolist() == pytest.approx([0.125, 0.025, -0.0625])


def test_lidar_near_finish_keeps_fresh_lookahead_on_open_track(tmp_path: Path) -> None:
    asset = _asset(tmp_path, lookahead_points=60)
    geometry = BoundaryGeometry(asset)
    pipeline = LidarFeaturePipeline(asset, expected_map_uid="trackmaniarl-test")
    observation = np.zeros(33, dtype=np.float32)
    observation[4:7] = geometry.reward_center[-1]
    observation[10:13] = [1, 0, 0]
    output = pipeline.transform_observation(observation)
    assert bool(output["lidar_mask"].all())
    assert not torch.allclose(output["lidar"][:, 0], output["lidar"][:, 14])


def test_lidar_exposes_racing_line_pace_dynamics_and_finish_gate(tmp_path: Path) -> None:
    asset = _asset(tmp_path, lookahead_points=60)
    geometry = BoundaryGeometry(asset)
    frames = np.zeros((geometry.recorded_count, 33), dtype=np.float32)
    frames[:, 3] = np.linspace(0.0, 5_000.0, geometry.recorded_count)
    frames[:, 4:7] = geometry.racing_line
    frames[:, 16] = np.linspace(20.0, 40.0, geometry.recorded_count)
    pace = tmp_path / "pace.npz"
    np.savez_compressed(
        pace,
        map_uid=np.asarray(geometry.map_uid),
        geometry_sha256=np.asarray(geometry.sha256),
        frames=frames,
        finish_time_s=np.asarray(5.0),
    )
    pipeline = LidarFeaturePipeline(
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
    observation = frames[-1].copy()
    observation[10] = 1.0
    prepared = pipeline.transform_observation(observation)

    assert prepared["lidar"].shape == (8, 60)
    assert prepared["telemetry"].shape == (49,)
    assert prepared["lidar"][6].max() > 0.8
    assert prepared["lidar"][7].max() == pytest.approx(1.0)
    assert prepared["telemetry"][27:31].tolist() == pytest.approx([0.5] * 4, abs=0.03)
    assert prepared["telemetry"][-1] == pytest.approx(1.0)

    model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=49,
        lidar_channels=8,
        telemetry_group_dims=(26, 5, 4, 14),
        spatial_bins=2,
    )
    batched = {key: value.unsqueeze(0) for key, value in prepared.items()}
    assert model.q_values(batched, quantile_count=8).shape == (1, 78)


def test_lidar_keeps_pace_profile_when_evaluating_its_configured_geometry(tmp_path: Path) -> None:
    asset = _asset(tmp_path, lookahead_points=60)
    geometry = BoundaryGeometry(asset)
    frames = np.zeros((geometry.recorded_count, 33), dtype=np.float32)
    frames[:, 3] = np.linspace(0.0, 5_000.0, geometry.recorded_count)
    frames[:, 4:7] = geometry.racing_line
    pace = tmp_path / "pace.npz"
    np.savez_compressed(
        pace,
        map_uid=np.asarray(geometry.map_uid),
        geometry_sha256=np.asarray(geometry.sha256),
        frames=frames,
        finish_time_s=np.asarray(5.0),
    )
    pipeline = LidarFeaturePipeline(
        asset,
        expected_map_uid="trackmaniarl-test",
        pace_reference_path=pace,
    )

    pipeline.set_evaluation_map(
        SimpleNamespace(geometry_path=asset, expected_map_uid="trackmaniarl-test")
    )

    assert pipeline.pace_profile is not None


def test_lidar_progress_is_bounded_by_reachable_arc_length(tmp_path: Path) -> None:
    asset = _asset(tmp_path, lookahead_points=0)
    pipeline = LidarFeaturePipeline(
        asset,
        expected_map_uid="trackmaniarl-test",
        max_speed_mps=1.0,
        max_time_delta_s=1.0,
        limit_progress_by_kinematics=True,
    )
    geometry = BoundaryGeometry(asset)
    first = np.zeros(33, dtype=np.float32)
    first[4:7] = geometry.center[0]
    first[10] = 1.0
    pipeline.transform_observation(first)
    jumped = first.copy()
    jumped[3] = 100.0
    jumped[4:7] = geometry.center[-1]

    pipeline.transform_observation(jumped)

    assert pipeline._progress_index == 0


def test_lidar_pipeline_preserves_legacy_right_then_forward_car_frame(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path, lookahead_points=0),
        expected_map_uid="trackmaniarl-test",
        max_distance_m=10.0,
    )
    observation = np.zeros(33, dtype=np.float32)
    observation[10] = 1.0  # Car points along +X.

    output = pipeline.transform_observation(observation)

    # The 2 m-resampled next left-boundary point is (2, 0, -5): it is 5 m right
    # and 2 m ahead in the established OpenPlanet local-frame convention.
    assert output["lidar"][:2, 0].tolist() == pytest.approx([0.5, 0.2])


def test_lidar_pipeline_stacks_track_relative_history(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path),
        expected_map_uid="trackmaniarl-test",
        history_length=2,
        include_track_relative=True,
        max_speed_mps=10.0,
        velocity_to_mps_scale=1.0,
    )
    first = np.zeros(33, dtype=np.float32)
    first[10] = 1.0
    first[7] = 5.0
    initial = pipeline.transform_observation(first)
    second = first.copy()
    second[3] = 1_000.0
    second[4] = 4.0
    stacked = pipeline.transform_observation(second)

    assert initial["lidar"].shape == (2, 4, 60)
    assert initial["telemetry"].shape == (2, 26)
    assert torch.equal(initial["telemetry"][0], initial["telemetry"][1])
    assert stacked["telemetry"][-1, 20] > stacked["telemetry"][0, 20]
    assert stacked["telemetry"][-1, 23] == pytest.approx(1.0)
    assert stacked["telemetry"][-1, 24] == pytest.approx(0.5)

    pipeline.reset_episode()
    reset = pipeline.transform_observation(second)
    assert torch.equal(reset["telemetry"][0], reset["telemetry"][1])


def test_track_relative_velocity_uses_the_same_native_unit_scale_as_telemetry(
    tmp_path: Path,
) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path),
        expected_map_uid="trackmaniarl-test",
        include_track_relative=True,
        max_speed_mps=20.0,
    )
    observation = np.zeros(33, dtype=np.float32)
    observation[10] = 1.0
    observation[7] = 10_000.0

    prepared = pipeline.transform_observation(observation)

    assert prepared["telemetry"][4] == pytest.approx(0.5)
    assert prepared["telemetry"][-2] == pytest.approx(0.5)


def test_iqn_lidar_updates_and_handles_single_structured_observation(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(_asset(tmp_path), expected_map_uid="trackmaniarl-test")
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


def test_temporal_iqn_handles_explicit_history(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path),
        expected_map_uid="trackmaniarl-test",
        history_length=1,
        include_track_relative=True,
    )
    raw = np.zeros(33, dtype=np.float32)
    raw[10] = 1.0
    single = pipeline.transform_observation(raw)
    learner = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        train_quantile_count=8,
        target_quantile_count=8,
        evaluation_quantile_count=8,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    policy = learner.policy()

    assert isinstance(policy.act(single, deterministic=True), int)
    assert isinstance(policy.act(single, deterministic=True), int)
    policy.reset_episode()

    observations = {
        key: value.view(1, 1, *value.shape).repeat(2, 2, *([1] * value.ndim))
        for key, value in single.items()
    }
    batch = TrainingBatch(
        data=observations,
        observations=observations,
        actions=torch.tensor([[0, 1], [2, 3]]),
        rewards=torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        next_observations=observations,
        terminated=torch.zeros((2, 2), dtype=torch.bool),
        truncated=torch.zeros((2, 2), dtype=torch.bool),
        bootstrap_discounts=torch.full((2, 2), 0.99),
        transition_ids=[1, 2, 3, 4],
        importance_weights=torch.ones(2),
        metadata={"priority_transition_ids": (2, 4)},
    )
    metrics, priorities = learner.update(batch)

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert priorities.transition_ids == [2, 4]


def test_iqn_warm_starts_compatible_tensors_into_expanded_observation_model(
    tmp_path: Path,
) -> None:
    source = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        execution={"device": "cpu", "precision": "float32"},
    )
    source.setup({"seed": 0})
    checkpoint = tmp_path / "source.pt"
    torch.save({"learner": source.state_dict()}, checkpoint)
    target = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=49,
            lidar_channels=8,
            telemetry_group_dims=(26, 5, 4, 14),
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        model_initialization_checkpoint=checkpoint.name,
        base_dir=tmp_path,
        execution={"device": "cpu", "precision": "float32"},
    )
    target.setup({"seed": 1})
    assert target.model is not None
    source_state = source.model.state_dict()
    target_state = target.model.state_dict()
    conv_name = "encoder.encoder.frame.track.0.weight"

    assert target.initialized_exact_tensors > 10
    assert target.initialized_expanded_tensors == 2
    assert torch.equal(target_state[conv_name][:, :4], source_state[conv_name])
    assert torch.count_nonzero(target_state[conv_name][:, 4:]) == 0
    assert torch.equal(target_state["head.weight"], source_state["head.weight"])


def test_iqn_goal_residual_preserves_legacy_policy_and_anchor(tmp_path: Path) -> None:
    source = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        execution={"device": "cpu", "precision": "float32"},
    )
    source.setup({"seed": 0})
    checkpoint = tmp_path / "legacy-policy.pt"
    torch.save({"learner": source.state_dict()}, checkpoint)
    target = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=49,
            base_telemetry_dim=26,
            auxiliary_remaining_distance_index=47,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        model_initialization_checkpoint=checkpoint.name,
        policy_anchor_checkpoint=checkpoint.name,
        policy_anchor_weight=1.0,
        base_dir=tmp_path,
        execution={"device": "cpu", "precision": "float32"},
    )
    target.setup({"seed": 1})
    assert source.model is not None
    assert target.model is not None
    assert target.policy_anchor_model is not None
    lidar = torch.randn(2, 2, 4, 90)
    mask = torch.ones(2, 2, 90)
    telemetry = torch.randn(2, 2, 26)
    auxiliary = torch.randn(2, 2, 23)
    observations = {"lidar": lidar, "lidar_mask": mask, "telemetry": telemetry}
    expanded = {
        "lidar": lidar,
        "lidar_mask": mask,
        "telemetry": torch.cat((telemetry, auxiliary), dim=-1),
    }
    quantiles = torch.linspace(0.1, 0.9, 8).repeat(2, 1)

    expected = source.model(observations, quantiles)

    assert torch.equal(target.model(expanded, quantiles), expected)
    assert torch.equal(target.policy_anchor_model(expanded, quantiles), expected)


def test_iqn_goal_residual_is_distance_gated_and_only_branch_trains_offline() -> None:
    model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=49,
        base_telemetry_dim=26,
        auxiliary_remaining_distance_index=47,
        history_length=2,
        burn_in=1,
        spatial_bins=2,
    )
    assert model.encoder.auxiliary is not None
    with torch.no_grad():
        model.encoder.auxiliary[-1].weight.zero_()
        model.encoder.auxiliary[-1].bias.fill_(1.0)
    lidar = torch.randn(2, 2, 4, 90)
    mask = torch.ones(2, 2, 90)
    telemetry = torch.randn(2, 2, 49)
    telemetry[:, :, 47] = 1.0
    observations = {"lidar": lidar, "lidar_mask": mask, "telemetry": telemetry}
    baseline = model.encoder.encoder(lidar, telemetry[..., :26], mask)

    assert torch.equal(model.encoder(observations), baseline)
    telemetry[:, :, 47] = 0.0
    assert torch.equal(model.encoder(observations), baseline + 1.0)

    model.set_offline_pretraining(True)
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all(name.startswith("encoder.auxiliary.") for name in trainable)
    model.set_offline_pretraining(False)
    assert all(value.requires_grad for value in model.parameters())

    legacy_model = LidarIqnModel(cosine_count=8)
    legacy_model.set_offline_pretraining(True)
    assert all(value.requires_grad for value in legacy_model.parameters())


def test_iqn_auxiliary_residual_can_start_at_a_measured_progress_segment() -> None:
    model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=27,
        base_telemetry_dim=26,
        auxiliary_progress_index=20,
        auxiliary_start_progress=0.65,
        auxiliary_residual_scale=0.25,
        history_length=2,
        burn_in=1,
        spatial_bins=2,
    )
    assert model.encoder.auxiliary is not None
    with torch.no_grad():
        model.encoder.auxiliary[-1].weight.zero_()
        model.encoder.auxiliary[-1].bias.fill_(1.0)
    lidar = torch.randn(2, 2, 4, 90)
    mask = torch.ones(2, 2, 90)
    telemetry = torch.zeros(2, 2, 27)
    observations = {"lidar": lidar, "lidar_mask": mask, "telemetry": telemetry}

    telemetry[:, :, 20] = 0.65
    baseline = model.encoder.encoder(lidar, telemetry[..., :26], mask)
    assert torch.equal(model.encoder(observations), baseline)
    telemetry[:, :, 20] = 1.0
    baseline = model.encoder.encoder(lidar, telemetry[..., :26], mask)
    assert torch.equal(model.encoder(observations), baseline + torch.tanh(torch.ones(1)) * 0.25)


def test_iqn_auxiliary_only_training_remains_frozen_after_offline_phase() -> None:
    model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=49,
        base_telemetry_dim=26,
        auxiliary_remaining_distance_index=47,
        train_auxiliary_only=True,
        history_length=2,
        burn_in=1,
        spatial_bins=2,
    )

    model.set_offline_pretraining(True)
    model.set_offline_pretraining(False)

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all(name.startswith("encoder.auxiliary.") for name in trainable)


def test_iqn_demonstration_loss_is_weighted_toward_the_finish() -> None:
    model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=49,
        base_telemetry_dim=26,
        auxiliary_remaining_distance_index=47,
        history_length=4,
        burn_in=1,
        spatial_bins=2,
    )
    telemetry = torch.zeros(2, 4, 49)
    telemetry[:, :, 47] = torch.tensor([1.0, 0.75, 0.25, 0.0])
    observation = {
        "lidar": torch.zeros(2, 4, 4, 90),
        "lidar_mask": torch.ones(2, 4, 90),
        "telemetry": telemetry,
    }

    weights = model.demonstration_loss_weights(observation, [1, 3])

    assert weights is not None
    assert torch.equal(weights, torch.tensor([[0.25, 1.0], [0.25, 1.0]]))


def test_iqn_auxiliary_only_update_changes_no_legacy_parameters() -> None:
    model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=49,
        base_telemetry_dim=26,
        auxiliary_remaining_distance_index=47,
        train_auxiliary_only=True,
        history_length=3,
        burn_in=1,
        spatial_bins=2,
    )
    learner = ImplicitQuantileQLearning(
        model,
        train_quantile_count=8,
        target_quantile_count=8,
        evaluation_quantile_count=8,
        demonstration_cross_entropy_weight=1.0,
        demonstration_td_weight=0.0,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    telemetry = torch.randn(2, 3, 49)
    telemetry[:, :, 47] = 0.0
    observations = {
        "lidar": torch.randn(2, 3, 4, 90),
        "lidar_mask": torch.ones(2, 3, 90),
        "telemetry": telemetry,
    }
    batch = TrainingBatch(
        data=observations,
        observations=observations,
        actions=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        rewards=torch.zeros(2, 3),
        next_observations=observations,
        terminated=torch.zeros(2, 3, dtype=torch.bool),
        truncated=torch.zeros(2, 3, dtype=torch.bool),
        bootstrap_discounts=torch.full((2, 3), 0.999),
        transition_ids=list(range(6)),
        importance_weights=torch.ones(2),
        masks=torch.ones(2, 3, dtype=torch.bool),
        metadata={
            "gamma": 0.999,
            "n_step": 1,
            "priority_transition_ids": (2, 5),
            "demo_flags": (True, True),
        },
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}

    metrics, _ = learner.update(batch)

    changed = {
        name for name, value in model.state_dict().items() if not torch.equal(value, before[name])
    }
    assert metrics["loss/demonstration_cross_entropy"] > 0.0
    assert metrics["debug/gradient_norm"] > 0.0
    assert changed
    assert all(name.startswith("encoder.auxiliary.") for name in changed)


def test_iqn_policy_anchor_migrates_the_legacy_telemetry_encoder(tmp_path: Path) -> None:
    source = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        execution={"device": "cpu", "precision": "float32"},
    )
    source.setup({"seed": 0})
    assert source.model is not None
    source_state = dict(source.model.state_dict())
    for current, legacy in (
        (
            "encoder.encoder.frame.telemetry.0.0.weight",
            "encoder.encoder.frame.telemetry.0.weight",
        ),
        (
            "encoder.encoder.frame.telemetry.0.0.bias",
            "encoder.encoder.frame.telemetry.0.bias",
        ),
        (
            "encoder.encoder.frame.telemetry.0.1.weight",
            "encoder.encoder.frame.telemetry.1.weight",
        ),
        (
            "encoder.encoder.frame.telemetry.0.1.bias",
            "encoder.encoder.frame.telemetry.1.bias",
        ),
    ):
        source_state[legacy] = source_state.pop(current)
    checkpoint = tmp_path / "legacy-anchor.pt"
    torch.save({"learner": {"model": source_state}}, checkpoint)
    anchored = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        policy_anchor_weight=1.0,
        policy_anchor_checkpoint=checkpoint.name,
        base_dir=tmp_path,
        execution={"device": "cpu", "precision": "float32"},
    )

    anchored.setup({"seed": 1})

    assert anchored.policy_anchor_model is not None
    for name, value in source.model.state_dict().items():
        assert torch.equal(anchored.policy_anchor_model.state_dict()[name], value)


def test_iqn_policy_anchor_expands_the_legacy_telemetry_input(tmp_path: Path) -> None:
    source = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
            legacy_telemetry_layout=True,
        ),
        execution={"device": "cpu", "precision": "float32"},
    )
    source.setup({"seed": 0})
    checkpoint = tmp_path / "legacy-policy.pt"
    torch.save({"learner": source.state_dict()}, checkpoint)
    target = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=27,
            history_length=2,
            burn_in=1,
            spatial_bins=2,
            legacy_telemetry_layout=True,
        ),
        model_initialization_checkpoint=checkpoint.name,
        policy_anchor_checkpoint=checkpoint.name,
        policy_anchor_weight=1.0,
        base_dir=tmp_path,
        execution={"device": "cpu", "precision": "float32"},
    )

    target.setup({"seed": 1})

    assert target.model is not None
    assert target.policy_anchor_model is not None
    name = "encoder.encoder.frame.telemetry.0.weight"
    source_weight = source.model.state_dict()[name]
    model_weight = target.model.state_dict()[name]
    anchor_weight = target.policy_anchor_model.state_dict()[name]
    assert torch.equal(model_weight[:, :26], source_weight)
    assert torch.count_nonzero(model_weight[:, 26:]) == 0
    assert torch.equal(anchor_weight, model_weight)


def test_iqn_warm_starts_the_frame_encoder_from_behavior_cloning(tmp_path: Path) -> None:
    source = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=49,
        lidar_channels=8,
        telemetry_group_dims=(26, 5, 4, 14),
        spatial_bins=2,
    )
    checkpoint = tmp_path / "behavior-cloning.pt"
    torch.save({"learner": {"model": source.state_dict()}}, checkpoint)
    target = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=49,
            lidar_channels=8,
            telemetry_group_dims=(26, 5, 4, 14),
            history_length=2,
            burn_in=1,
            spatial_bins=2,
        ),
        model_initialization_checkpoint=checkpoint.name,
        base_dir=tmp_path,
        execution={"device": "cpu", "precision": "float32"},
    )

    target.setup({"seed": 0})

    assert target.model is not None
    source_state = source.state_dict()
    target_state = target.model.state_dict()
    assert target.initialized_exact_tensors > 10
    assert torch.equal(
        target_state["encoder.encoder.frame.track.0.weight"],
        source_state["encoder.encoder.track.0.weight"],
    )
    assert torch.equal(
        target_state["encoder.encoder.frame.projection.0.weight"],
        source_state["encoder.encoder.projection.0.weight"],
    )


def test_iqn_warm_start_preserves_behavior_cloning_greedy_policy(tmp_path: Path) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    source = LidarBehaviorCloningModel(
        action_ids=action_ids,
        telemetry_dim=26,
        spatial_bins=2,
        masked_telemetry_indices=(3, 23),
    ).eval()
    checkpoint = tmp_path / "behavior-cloning-policy.pt"
    torch.save(
        {"learner": {"model": source.state_dict(), "policy_action_ids": action_ids}},
        checkpoint,
    )
    learner = ImplicitQuantileQLearning(
        LidarIqnModel(
            cosine_count=8,
            telemetry_dim=26,
            spatial_bins=2,
            masked_telemetry_indices=(3, 23),
        ),
        policy_action_ids=action_ids,
        model_initialization_checkpoint=checkpoint.name,
        base_dir=tmp_path,
        evaluation_quantile_count=8,
        execution={"device": "cpu", "precision": "float32"},
    )

    learner.setup({"seed": 0})

    assert learner.model is not None
    observations = {
        "lidar": torch.randn(16, 4, 90),
        "lidar_mask": torch.ones(16, 90),
        "telemetry": torch.randn(16, 26),
    }
    with torch.inference_mode():
        compact_actions = source(observations).argmax(dim=-1)
        q_values = learner.model.q_values(observations, quantile_count=8)
    expected_actions = torch.tensor(action_ids)[compact_actions]
    actual_actions = q_values.masked_fill(
        ~torch.tensor([index in action_ids for index in range(78)]), -torch.inf
    ).argmax(dim=-1)
    state = learner.model.state_dict()
    disallowed = torch.tensor([index not in action_ids for index in range(78)])

    assert torch.equal(actual_actions, expected_actions)
    assert torch.equal(state["head.weight"][list(action_ids)], source.state_dict()["head.weight"])
    assert torch.equal(state["head.bias"][list(action_ids)], source.state_dict()["head.bias"])
    assert torch.count_nonzero(state["head.weight"][disallowed]) == 0
    assert torch.count_nonzero(state["head.bias"][disallowed]) == 0
    assert torch.count_nonzero(state["quantile_embedding.0.weight"]) == 0
    assert torch.equal(
        learner.model.quantile_embedding(torch.randn(4, 8)),
        torch.ones(4, learner.model.feature_dim),
    )
    assert torch.count_nonzero(state["value.weight"]) == 0
    assert torch.count_nonzero(state["value.bias"]) == 0


def test_iqn_rejects_behavior_cloning_action_contract_mismatch(tmp_path: Path) -> None:
    source_action_ids = (0, 1, 3, 39, 72, 73, 75)
    source = LidarBehaviorCloningModel(action_ids=source_action_ids, spatial_bins=2)
    checkpoint = tmp_path / "behavior-cloning-policy.pt"
    torch.save(
        {
            "learner": {
                "model": source.state_dict(),
                "policy_action_ids": source_action_ids,
            }
        },
        checkpoint,
    )
    learner = ImplicitQuantileQLearning(
        LidarIqnModel(cosine_count=8, spatial_bins=2),
        policy_action_ids=(0, 1, 3, 39, 72, 73, 74),
        model_initialization_checkpoint=checkpoint.name,
        base_dir=tmp_path,
        execution={"device": "cpu", "precision": "float32"},
    )

    with pytest.raises(ValueError, match="action contract"):
        learner.setup({"seed": 0})


def test_iqn_masks_telemetry_for_frame_sequence_and_policy() -> None:
    frame_model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=26,
        spatial_bins=2,
        masked_telemetry_indices=(3, 23),
    )
    factory_model = LidarIqnModelFactory(
        cosine_count=8,
        telemetry_dim=26,
        spatial_bins=2,
        masked_telemetry_indices=(3, 23),
    ).build()
    base = {
        "lidar": torch.randn(2, 4, 90),
        "lidar_mask": torch.ones(2, 90),
        "telemetry": torch.randn(2, 26),
    }
    changed = {key: value.clone() for key, value in base.items()}
    changed["telemetry"][:, 3] += 100.0
    changed["telemetry"][:, 23] -= 100.0
    quantiles = torch.linspace(0.1, 0.9, 8).expand(2, -1)
    sequence_model = LidarIqnModel(
        cosine_count=8,
        telemetry_dim=26,
        history_length=2,
        burn_in=1,
        spatial_bins=2,
        masked_telemetry_indices=(3, 23),
    )
    sequence = {
        key: value.unsqueeze(1).expand(-1, 2, *value.shape[1:]).clone()
        for key, value in base.items()
    }
    changed_sequence = {key: value.clone() for key, value in sequence.items()}
    changed_sequence["telemetry"][:, :, 3] += 100.0
    changed_sequence["telemetry"][:, :, 23] -= 100.0

    assert frame_model.masked_telemetry_indices.tolist() == [3, 23]
    assert factory_model.masked_telemetry_indices.tolist() == [3, 23]
    assert torch.equal(frame_model(base, quantiles), frame_model(changed, quantiles))
    assert torch.equal(
        sequence_model.encode_sequence(sequence),
        sequence_model.encode_sequence(changed_sequence),
    )
    learner = ImplicitQuantileQLearning(
        frame_model,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    policy = learner.policy()
    single = {key: value[0] for key, value in base.items()}
    changed_single = {key: value[0] for key, value in changed.items()}
    assert policy.act(single, deterministic=True) == policy.act(changed_single, deterministic=True)


@pytest.mark.parametrize("indices", [(3, 3), (-1,), (26,)])
def test_iqn_rejects_invalid_masked_telemetry_indices(indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="masked telemetry"):
        LidarIqnModel(telemetry_dim=26, masked_telemetry_indices=indices)


def test_iqn_resume_uses_configured_learning_rate(tmp_path: Path) -> None:
    model = LidarIqnModel(cosine_count=8)
    first = ImplicitQuantileQLearning(
        model,
        learning_rate=3e-5,
        execution={"device": "cpu", "precision": "float32"},
    )
    first.setup({"seed": 0})
    state = first.state_dict()
    resumed = ImplicitQuantileQLearning(
        LidarIqnModel(cosine_count=8),
        learning_rate=1e-4,
        execution={"device": "cpu", "precision": "float32"},
    )
    resumed.setup({"seed": 0})
    resumed.load_state_dict(state)

    assert {group["lr"] for group in resumed.optimizer.param_groups} == {1e-4}


def test_iqn_best_evaluation_checkpoint_uses_the_exact_policy_and_clean_optimizer() -> None:
    learner = ImplicitQuantileQLearning(
        LidarIqnModel(cosine_count=8),
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    policy_state = {
        name: torch.zeros_like(value) for name, value in learner.policy().export_state().items()
    }

    checkpoint = learner.state_dict_for_policy(policy_state)

    assert all(torch.count_nonzero(value) == 0 for value in checkpoint["model"].values())
    assert all(
        torch.equal(checkpoint["model"][name], checkpoint["target_model"][name])
        for name in checkpoint["model"]
    )
    assert checkpoint["optimizer"]["state"] == {}


def test_track_geometry_attention_masks_bfloat16_logits() -> None:
    encoder = TrackGeometryEncoder(channels=4, telemetry_dim=0)
    track = torch.randn(2, 4, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    mask[:, -10:] = False

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(track, mask=mask)

    assert torch.isfinite(output).all()


def test_track_geometry_encoder_can_preserve_ordered_spatial_bins() -> None:
    encoder = TrackGeometryEncoder(4, 6, output_dim=32, spatial_bins=6)
    track = torch.randn(3, 4, 90)
    telemetry = torch.randn(3, 6)
    mask = torch.ones(3, 90)

    assert encoder(track, telemetry, mask).shape == (3, 32)


def test_iqn_can_build_the_legacy_telemetry_checkpoint_layout() -> None:
    model = LidarIqnModel(
        telemetry_dim=26,
        history_length=64,
        burn_in=32,
        spatial_bins=12,
        legacy_telemetry_layout=True,
    )

    state = model.state_dict()

    assert "encoder.encoder.frame.telemetry.0.weight" in state
    assert "encoder.encoder.frame.telemetry.1.weight" in state
    assert "encoder.encoder.frame.telemetry.0.0.weight" not in state


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
                    "map_uid": "trackmaniarl-test",
                    "ready": "true",
                }
                connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = OpenPlanetSessionClient(host, port, timeout_s=1)
    assert client.verify_loaded_map("trackmaniarl-test").map_uid == "trackmaniarl-test"
    assert client.confirm_ready("trackmaniarl-test").map_uid == "trackmaniarl-test"
    thread.join(timeout=1)
    assert commands == ["verify_loaded_map", "confirm_ready"]
