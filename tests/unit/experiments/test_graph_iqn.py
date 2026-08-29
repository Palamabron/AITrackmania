from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trackmaniarl.experiments.graph_iqn import (
    BoundaryGraphFeaturePipeline,
    DuelingImplicitQuantileHead,
    TrackGnnSimbaEncoder,
    TrackGtnSimbaEncoder,
)
from trackmaniarl.models.contracts import ValueSupport
from trackmaniarl.models.track_graphs import TrackGraphTransformer, TrackNeighborGraph
from trackmaniarl.trackmania.geometry import GEOMETRY_ASSET_VERSION


def test_graph_iqn_model_contract() -> None:
    encoder = TrackGnnSimbaEncoder()
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


def test_gtn_simba_encoder_matches_value_model_contract() -> None:
    observation = {"track": torch.zeros(2, 3, 88), "physics": torch.zeros(2, 60)}

    features = TrackGtnSimbaEncoder()(observation)

    assert features.shape == (2, 192)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("encoder_type", [TrackGnnSimbaEncoder, TrackGtnSimbaEncoder])
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
