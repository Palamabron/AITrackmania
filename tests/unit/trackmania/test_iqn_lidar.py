"""Release contracts for the IQN plus offline boundary lidar baseline."""

from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from trackmaniarl.algorithms import ImplicitQuantileQLearning
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.encoders import (
    TemporalMambaTrackGeometryEncoder,
    require_mamba_layer,
)
from trackmaniarl.trackmania.behavior_cloning import LidarBehaviorCloningModel
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.iqn import LidarIqnModel
from trackmaniarl.trackmania.mamba import (
    LidarMambaModel,
    LidarMambaModelFactory,
)
from trackmaniarl.trackmania.session import PLUGIN_PROTOCOL_VERSION, OpenPlanetSessionClient


class _FakeMamba(torch.nn.Module):
    def __init__(self, d_model: int, **_: object) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor, **_: object) -> torch.Tensor:
        return cast(torch.Tensor, self.projection(values))


def test_mamba_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)

    with pytest.raises(RuntimeError, match="uv sync --extra mamba"):
        require_mamba_layer()


def test_mamba_encoder_validates_window_and_returns_loss_steps() -> None:
    with pytest.raises(ValueError, match=r"burn_in must be in \[0, history_length\)"):
        TemporalMambaTrackGeometryEncoder(
            4,
            6,
            history_length=3,
            burn_in=3,
            mamba_cls=_FakeMamba,
        )
    encoder = TemporalMambaTrackGeometryEncoder(
        4,
        6,
        history_length=4,
        burn_in=1,
        spatial_bins=2,
        mamba_cls=_FakeMamba,
    )

    features = encoder.encode_steps(
        torch.randn(2, 4, 4, 16),
        torch.randn(2, 4, 6),
        torch.ones(2, 4, 16),
    )

    assert features.shape == (2, 3, 256)
    assert torch.isfinite(features).all()


def test_mamba_burn_in_does_not_backpropagate_through_context() -> None:
    encoder = TemporalMambaTrackGeometryEncoder(
        4,
        6,
        history_length=4,
        burn_in=2,
        mamba_cls=_FakeMamba,
    )
    track = torch.randn(2, 4, 4, 16, requires_grad=True)

    encoder.encode_steps(
        track, torch.randn(2, 4, 6), torch.ones(2, 4, 16)
    ).square().sum().backward()

    assert track.grad is not None
    assert torch.count_nonzero(track.grad[:, :2]) == 0
    assert torch.count_nonzero(track.grad[:, 2:]) > 0


def test_mamba_factory_preserves_sequence_and_action_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trackmaniarl.models.encoders.track_geometry.require_mamba_layer",
        lambda: _FakeMamba,
    )
    model = LidarMambaModelFactory(
        cosine_count=8,
        telemetry_dim=6,
        history_length=3,
        burn_in=1,
        spatial_bins=2,
        action_ids=(0, 1, 3),
    ).build()
    assert isinstance(model, LidarMambaModel)
    observations = {
        "lidar": torch.randn(2, 3, 4, 16),
        "lidar_mask": torch.ones(2, 3, 16),
        "telemetry": torch.randn(2, 3, 6),
    }

    assert model.encode_sequence(observations).shape == (2, 2, 256)
    assert model.q_values(observations, quantile_count=8).shape == (2, 3)


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


def test_lidar_pipeline_can_encode_velocity_in_the_local_car_frame(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _asset(tmp_path), expected_map_uid="trackmaniarl-test", local_velocity_features=True
    )
    observation = np.zeros(33, dtype=np.float32)
    observation[7:10] = [10_000.0, 2_000.0, 5_000.0]
    observation[10] = 1.0

    prepared = pipeline.transform_observation(observation)

    assert prepared["telemetry"][4:7].tolist() == pytest.approx([0.125, 0.025, -0.0625])


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
