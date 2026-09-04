from __future__ import annotations

import torch

from trackmaniarl.models.track_graphs import (
    DirectionalTrackNeighborGraph,
    TrackNeighborGraph,
)


def test_directional_graph_routes_predecessor_and_successor_separately() -> None:
    graph = DirectionalTrackNeighborGraph(point_count=4, hidden_dim=8, layer_count=1)
    hidden = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    predecessor, successor = graph._directional_neighbors(hidden)
    zero = torch.zeros_like(hidden[:, :1])

    torch.testing.assert_close(predecessor, torch.cat((zero, hidden[:, :-1]), dim=1))
    torch.testing.assert_close(successor, torch.cat((hidden[:, 1:], zero), dim=1))
    assert graph.predecessor_layers[0] is not graph.successor_layers[0]


def _linear(module: torch.nn.Module) -> torch.nn.Linear:
    if not isinstance(module, torch.nn.Linear):
        raise AssertionError("directional graph layers must be linear")
    return module


def _bias(linear: torch.nn.Linear) -> torch.Tensor:
    if linear.bias is None:
        raise AssertionError("graph self transforms require a bias")
    return linear.bias


def _tie_directional_layer(
    baseline: TrackNeighborGraph, directional: DirectionalTrackNeighborGraph, index: int
) -> None:
    source = _linear(baseline.layers[index])
    self_layer = _linear(directional.self_layers[index])
    predecessor = _linear(directional.predecessor_layers[index])
    successor = _linear(directional.successor_layers[index])
    split = directional.hidden_dim
    with torch.no_grad():
        self_layer.weight.copy_(source.weight[:, :split])
        _bias(self_layer).copy_(_bias(source))
        predecessor.weight.copy_(source.weight[:, split:])
        successor.weight.copy_(source.weight[:, split:])
    directional.norms[index].load_state_dict(baseline.norms[index].state_dict())


def _tie_directional_layers(
    baseline: TrackNeighborGraph, directional: DirectionalTrackNeighborGraph
) -> None:
    directional.node_in.load_state_dict(baseline.node_in.state_dict())
    directional.readout.load_state_dict(baseline.readout.state_dict())
    for index in range(len(baseline.layers)):
        _tie_directional_layer(baseline, directional, index)


def test_directional_graph_generalizes_tied_neighbor_aggregation() -> None:
    torch.manual_seed(2)
    baseline = TrackNeighborGraph(point_count=4, hidden_dim=8, layer_count=2).eval()
    directional = DirectionalTrackNeighborGraph(point_count=4, hidden_dim=8, layer_count=2).eval()
    _tie_directional_layers(baseline, directional)
    track = torch.randn(3, 3, 8)

    with torch.inference_mode():
        expected = baseline(track)
        actual = directional(track)

    torch.testing.assert_close(actual, expected)


def test_directional_graph_adds_only_untied_edge_parameters() -> None:
    baseline_count = sum(parameter.numel() for parameter in TrackNeighborGraph().parameters())
    directional_count = sum(
        parameter.numel() for parameter in DirectionalTrackNeighborGraph().parameters()
    )

    assert directional_count - baseline_count == 2 * 128**2


def test_directional_graph_matches_baseline_parameter_scale() -> None:
    graph = DirectionalTrackNeighborGraph(hidden_dim=128)
    baseline_bound = (2 * graph.hidden_dim) ** -0.5

    for layers in (graph.self_layers, graph.predecessor_layers, graph.successor_layers):
        for module in layers:
            linear = _linear(module)
            assert linear.weight.abs().max() <= baseline_bound
            if linear.bias is not None:
                assert linear.bias.abs().max() <= baseline_bound


def test_directional_graph_is_deterministic_and_direction_sensitive() -> None:
    torch.manual_seed(3)
    graph = DirectionalTrackNeighborGraph().eval()
    track = torch.randn(2, 3, 88)
    reversed_track = track.reshape(2, 3, 44, 2).flip(2).reshape(2, 3, 88)

    with torch.inference_mode():
        first = graph(track)
        second = graph(track)
        reversed_output = graph(reversed_track)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert (first - reversed_output).abs().max() > 1.0e-4


def test_directional_graph_shape_and_gradients() -> None:
    torch.manual_seed(4)
    graph = DirectionalTrackNeighborGraph()
    track = torch.randn(3, 3, 88, requires_grad=True)

    output = graph(track)
    output.square().mean().backward()

    assert output.shape == (3, 128)
    assert track.grad is not None
    assert torch.isfinite(track.grad).all()
    for layers in (graph.predecessor_layers, graph.successor_layers):
        for layer in layers:
            assert all(parameter.grad is not None for parameter in layer.parameters())
            assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
