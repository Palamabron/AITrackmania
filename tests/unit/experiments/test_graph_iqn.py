from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.experiments.graph_iqn import (
    BoundaryGraphFeaturePipeline,
    DuelingImplicitQuantileHead,
    TrackArcLengthGnnSimbaEncoder,
    TrackDirectionalGnnSimbaEncoder,
    TrackGnnSimbaEncoder,
    TrackGtnSimbaEncoder,
)
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.contracts import ValueSupport
from trackmaniarl.models.strategies import RandomQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore
from trackmaniarl.models.track_graphs import (
    ArcLengthTrackNeighborGraph,
    TrackGraphTransformer,
    TrackNeighborGraph,
)
from trackmaniarl.trackmania.geometry import GEOMETRY_ASSET_VERSION


@pytest.mark.parametrize(
    "encoder_type",
    [TrackGnnSimbaEncoder, TrackArcLengthGnnSimbaEncoder, TrackDirectionalGnnSimbaEncoder],
)
def test_graph_iqn_model_contract(encoder_type: type[torch.nn.Module]) -> None:
    encoder = encoder_type()
    head = DuelingImplicitQuantileHead()
    observation = {
        "track": torch.zeros(2, 3, 88),
        "physics": torch.zeros(2, 60),
    }
    support = ValueSupport(torch.full((2, 32), 0.5), torch.full((2, 32), 1.0 / 32.0))

    features = encoder(observation)
    values = head.evaluate_all(features, support)

    assert features.shape == (2, 192)
    assert values.shape == (2, 32, 78)


def test_graph_iqn_head_selects_actions_across_sequence_dimensions() -> None:
    head = DuelingImplicitQuantileHead()
    features = torch.randn(2, 3, 192)
    points = torch.rand(2, 3, 5)
    support = ValueSupport(points, torch.full_like(points, 0.2))
    actions = torch.tensor([[0, 12, 77], [39, 4, 61]])

    all_actions = head.evaluate_all(features, support)
    selected = head.evaluate_actions(features, support, actions)
    expected = all_actions.gather(
        -1,
        actions[..., None, None].expand(*actions.shape, points.shape[-1], 1),
    ).squeeze(-1)

    assert selected.shape == (2, 3, 5)
    torch.testing.assert_close(selected, expected)


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


def test_graph_pipeline_uses_versioned_dense_geometry(tmp_path: Path) -> None:
    geometry = _dense_geometry(tmp_path / "graph-geometry.npz")
    pipeline = BoundaryGraphFeaturePipeline(geometry, "test-map")
    telemetry = np.zeros(33, dtype=np.float32)
    telemetry[12] = 1.0

    observation = pipeline.transform_observation(telemetry)

    assert observation["track"].shape == (3, 88)
    assert observation["physics"].shape == (60,)
    torch.testing.assert_close(observation["track"][1, :2], torch.tensor([2.5, 0.0]))
    assert observation["track"][1, -2] > 100.0
    synthetic = pipeline.transform_observation(pipeline.synthetic_observation())
    assert pipeline.observation_space.contains(
        {key: value.numpy() for key, value in synthetic.items()}
    )


def test_graph_pipeline_masks_current_control_labels(tmp_path: Path) -> None:
    pipeline = BoundaryGraphFeaturePipeline(_dense_geometry(tmp_path / "geometry.npz"), "test-map")
    telemetry = np.zeros(33, dtype=np.float32)
    telemetry[12] = 1.0
    baseline = pipeline.transform_observation(telemetry)
    pipeline.reset_episode()
    telemetry[30:33] = [1.0, 1.0, -1.0]

    controlled = pipeline.transform_observation(telemetry)

    torch.testing.assert_close(controlled["physics"], baseline["physics"])


def test_graph_pipeline_masks_prepared_control_label_features(tmp_path: Path) -> None:
    pipeline = BoundaryGraphFeaturePipeline(_dense_geometry(tmp_path / "geometry.npz"), "test-map")
    physics = torch.zeros(60)
    physics[4:7] = 1.0
    physics[10:12] = 1.0
    physics[-1] = 77.0
    prepared = pipeline.transform_observation({"track": torch.zeros(3, 88), "physics": physics})

    assert tuple(prepared) == tuple(pipeline.observation_space.spaces)
    torch.testing.assert_close(prepared["physics"][4:7], torch.zeros(3))
    torch.testing.assert_close(prepared["physics"][10:12], torch.zeros(2))
    assert prepared["physics"][-1] == 0.0


def test_graph_pipeline_rejects_non_finite_prepared_observations(tmp_path: Path) -> None:
    pipeline = BoundaryGraphFeaturePipeline(_dense_geometry(tmp_path / "geometry.npz"), "test-map")
    physics = torch.zeros(60)
    physics[0] = torch.nan

    with pytest.raises(ValueError, match="non-finite"):
        pipeline.transform_observation({"physics": physics, "track": torch.zeros(3, 88)})


def test_track_graph_and_transformer_are_differentiable() -> None:
    track = torch.randn(3, 3, 88, requires_grad=True)
    gnn = TrackNeighborGraph()
    gtn = TrackGraphTransformer()

    loss = gnn(track).square().mean() + gtn(track).square().mean()
    loss.backward()

    assert gnn(track.detach()).shape == (3, 128)
    assert gtn(track.detach()).shape == (3, 128)
    assert track.grad is not None
    assert torch.isfinite(track.grad).all()


def test_neighbor_graph_pairs_boundary_coordinates_per_point() -> None:
    graph = TrackNeighborGraph()
    track = torch.arange(264, dtype=torch.float32).reshape(1, 3, 88)
    captured: list[torch.Tensor] = []
    hook = graph.node_in.register_forward_pre_hook(lambda _, args: captured.append(args[0]))

    graph(track)
    hook.remove()

    expected = track.reshape(1, 3, 44, 2).permute(0, 2, 1, 3).flatten(2)
    torch.testing.assert_close(captured[0], expected)


def test_arc_length_graph_preserves_lookahead_order() -> None:
    torch.manual_seed(0)
    track = torch.randn(2, 3, 88)
    reversed_track = track.reshape(2, 3, 44, 2).flip(2).reshape(2, 3, 88)
    baseline = TrackNeighborGraph().eval()
    ordered = ArcLengthTrackNeighborGraph().eval()

    with torch.inference_mode():
        baseline_difference = (baseline(track) - baseline(reversed_track)).abs().max()
        ordered_difference = (ordered(track) - ordered(reversed_track)).abs().max()

    assert baseline_difference < 1.0e-6
    assert ordered_difference > 1.0e-4


def test_arc_length_graph_is_deterministic_and_differentiable() -> None:
    torch.manual_seed(1)
    graph = ArcLengthTrackNeighborGraph()
    track = torch.randn(3, 3, 88, requires_grad=True)

    first = graph(track)
    second = graph(track)
    first.square().mean().backward()

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert track.grad is not None
    assert torch.isfinite(track.grad).all()


def test_arc_length_graph_uses_fixed_normalized_lookahead_distance() -> None:
    graph = ArcLengthTrackNeighborGraph(point_count=4)

    torch.testing.assert_close(
        graph.normalized_arc_length,
        torch.tensor([0.25, 0.5, 0.75, 1.0]),
    )
    assert not graph.normalized_arc_length.requires_grad


def test_arc_length_graph_rejects_invalid_point_count() -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        ArcLengthTrackNeighborGraph(point_count=1)


def _graph_value_model(encoder: torch.nn.Module) -> CompositeValueModel:
    return CompositeValueModel(
        CompositeModules(
            encoder,
            IdentityTemporalCore(192),
            DuelingImplicitQuantileHead(),
            RandomQuantileStrategy(64, 64, 32),
        )
    )


def _graph_observations(generator: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        "track": torch.randn(2, 3, 88, generator=generator),
        "physics": torch.randn(2, 60, generator=generator),
    }


def _graph_batch() -> TrainingBatch:
    generator = torch.Generator().manual_seed(31)
    return TrainingBatch(
        data={},
        observations=_graph_observations(generator),
        actions=torch.tensor([7, 70]),
        rewards=torch.tensor([0.5, -0.25]),
        next_observations=_graph_observations(generator),
        terminated=torch.tensor([False, True]),
        truncated=torch.zeros(2, dtype=torch.bool),
        bootstrap_discounts=torch.tensor([0.99, 0.0]),
        transition_ids=[0, 1],
    )


@pytest.mark.parametrize(
    "encoder_type", [TrackArcLengthGnnSimbaEncoder, TrackDirectionalGnnSimbaEncoder]
)
def test_order_aware_model_checkpoint_and_policy_export_round_trip(
    encoder_type: type[torch.nn.Module],
) -> None:
    baseline = _graph_value_model(TrackGnnSimbaEncoder())
    learner = DiscreteValueLearner(_graph_value_model(encoder_type()))
    learner.setup({"seed": 17})
    checkpoint = learner.state_dict()
    exported = learner.policy().export_state()
    restored = DiscreteValueLearner(_graph_value_model(encoder_type()))
    restored.setup({"seed": 23})

    restored.load_state_dict(checkpoint)

    assert baseline.architecture_fingerprint() != learner.model.architecture_fingerprint()
    restored_export = restored.policy().export_state()
    assert restored_export.keys() == exported.keys()
    for name, value in exported.items():
        torch.testing.assert_close(restored_export[name], value, rtol=0.0, atol=0.0)


def test_graph_encoder_architecture_fingerprints_are_distinct() -> None:
    encoder_types = (
        TrackGnnSimbaEncoder,
        TrackArcLengthGnnSimbaEncoder,
        TrackDirectionalGnnSimbaEncoder,
        TrackGtnSimbaEncoder,
    )
    fingerprints = {
        _graph_value_model(encoder_type()).architecture_fingerprint()
        for encoder_type in encoder_types
    }

    assert len(fingerprints) == len(encoder_types)


def _cpu_graph_learner(seed: int, encoder_type: type[torch.nn.Module]) -> DiscreteValueLearner:
    learner = DiscreteValueLearner(_graph_value_model(encoder_type()), execution={"device": "cpu"})
    learner.setup({"seed": seed})
    return learner


@pytest.mark.parametrize(
    "encoder_type", [TrackArcLengthGnnSimbaEncoder, TrackDirectionalGnnSimbaEncoder]
)
def test_order_aware_model_cpu_update_resumes_exactly(
    encoder_type: type[torch.nn.Module],
) -> None:
    batch = _graph_batch()
    learner = _cpu_graph_learner(17, encoder_type)
    metrics, priorities = learner.update(batch)
    checkpoint = deepcopy(learner.state_dict())
    learner.update(batch)
    expected = deepcopy(learner.model.state_dict())
    restored = _cpu_graph_learner(23, encoder_type)
    restored.load_state_dict(checkpoint)
    restored.update(batch)
    assert metrics["loss/total"] > 0.0
    assert all(torch.isfinite(torch.tensor(priorities.priorities)))
    assert restored.update_count == learner.update_count == 2
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)


def test_gtn_simba_encoder_matches_value_model_contract() -> None:
    observation = {"track": torch.zeros(2, 3, 88), "physics": torch.zeros(2, 60)}

    features = TrackGtnSimbaEncoder()(observation)

    assert features.shape == (2, 192)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    "encoder_type",
    [
        TrackGnnSimbaEncoder,
        TrackArcLengthGnnSimbaEncoder,
        TrackDirectionalGnnSimbaEncoder,
        TrackGtnSimbaEncoder,
    ],
)
def test_graph_encoders_ignore_masked_control_label_features(
    encoder_type: type[torch.nn.Module],
) -> None:
    encoder = encoder_type()
    baseline = {"track": torch.zeros(2, 3, 88), "physics": torch.zeros(2, 60)}
    leaked = {key: value.clone() for key, value in baseline.items()}
    leaked["physics"][:, 4:7] = 1.0
    leaked["physics"][:, 10:12] = 1.0
    leaked["physics"][:, -1] = 77.0
    encoder.eval()

    with torch.inference_mode():
        expected = encoder(baseline)
        actual = encoder(leaked)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
