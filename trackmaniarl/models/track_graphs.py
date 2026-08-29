"""Graph encoders for ordered TrackMania boundary lookahead."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


class TrackNeighborGraph(nn.Module):
    """Bidirectional neighbor-aggregation graph over ordered track features."""

    def __init__(self, point_count: int = 44, hidden_dim: int = 128, layer_count: int = 2) -> None:
        super().__init__()
        if point_count < 2 or hidden_dim < 1 or layer_count < 1:
            raise ValueError("track graph dimensions must be positive")
        self.point_count = point_count
        self.hidden_dim = hidden_dim
        edge_src = torch.cat((torch.arange(point_count - 1), torch.arange(1, point_count)))
        edge_dst = torch.cat((torch.arange(1, point_count), torch.arange(point_count - 1)))
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        degree = torch.zeros(point_count)
        degree.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=degree.dtype))
        self.register_buffer("degree", degree.clamp_min(1), persistent=False)
        self.node_in = nn.Sequential(nn.Linear(6, hidden_dim), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layer_count)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layer_count))
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (3, self.point_count * 2):
            raise ValueError("track graph expects paired XZ coordinates for each point")
        paired = value.reshape(value.shape[0], 3, self.point_count, 2)
        hidden = self.node_in(paired.permute(0, 2, 1, 3).flatten(2))
        for linear, norm in zip(self.layers, self.norms, strict=True):
            hidden = self._message_step(hidden, cast(nn.Linear, linear), cast(nn.LayerNorm, norm))
        return cast(torch.Tensor, self.readout(hidden).mean(dim=1))

    def _message_step(
        self, hidden: torch.Tensor, linear: nn.Linear, norm: nn.LayerNorm
    ) -> torch.Tensor:
        edge_src = cast(torch.Tensor, self.edge_src)
        edge_dst = cast(torch.Tensor, self.edge_dst)
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(1, edge_dst, hidden[:, edge_src])
        degree = cast(torch.Tensor, self.degree).to(dtype=hidden.dtype)
        aggregate = aggregate / degree.view(1, -1, 1)
        updated = norm(linear(torch.cat((hidden, aggregate), dim=-1)).relu())
        return cast(torch.Tensor, updated)


def _graph_attention_mask(point_count: int) -> torch.Tensor:
    mask = torch.ones(point_count + 1, point_count + 1, dtype=torch.bool)
    mask[0, :] = False
    mask[:, 0] = False
    indices = torch.arange(1, point_count + 1)
    mask[indices, indices] = False
    mask[indices[:-1], indices[1:]] = False
    mask[indices[1:], indices[:-1]] = False
    return mask


@dataclass(frozen=True, slots=True)
class TrackGraphTransformerConfig:
    point_count: int = 44
    hidden_dim: int = 128
    layer_count: int = 2
    head_count: int = 4

    @classmethod
    def from_value(
        cls, value: TrackGraphTransformerConfig | Mapping[str, int] | None
    ) -> TrackGraphTransformerConfig:
        if value is None:
            return cls()
        if isinstance(value, TrackGraphTransformerConfig):
            return value
        return cls(**dict(value))

    @property
    def valid(self) -> bool:
        return (
            self.point_count >= 2
            and self.hidden_dim >= 1
            and self.layer_count >= 1
            and self.head_count >= 1
            and self.hidden_dim % self.head_count == 0
        )


class TrackGraphTransformer(nn.Module):
    """Local-attention graph transformer for 44 ordered boundary points."""

    def __init__(
        self, config: TrackGraphTransformerConfig | Mapping[str, int] | None = None
    ) -> None:
        super().__init__()
        options = TrackGraphTransformerConfig.from_value(config)
        if not options.valid:
            raise ValueError("graph transformer dimensions are invalid")
        self.point_count = options.point_count
        self.node_projection = nn.Linear(6, options.hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, options.point_count + 1, options.hidden_dim))
        self.summary = nn.Parameter(torch.zeros(1, 1, options.hidden_dim))
        self.encoder = _transformer_encoder(options)
        self.output_norm = nn.LayerNorm(options.hidden_dim)
        mask = _graph_attention_mask(options.point_count)
        self.register_buffer("attention_mask", mask, persistent=False)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (3, self.point_count * 2):
            raise ValueError("graph transformer expects paired XZ-boundary coordinates")
        paired = value.reshape(value.shape[0], 3, self.point_count, 2)
        nodes = self.node_projection(paired.permute(0, 2, 1, 3).flatten(2))
        summary = self.summary.expand(value.shape[0], -1, -1)
        sequence = torch.cat((summary, nodes), dim=1) + self.position
        encoded = self.encoder(sequence, mask=cast(torch.Tensor, self.attention_mask))
        return cast(torch.Tensor, self.output_norm(encoded[:, 0]))


def _transformer_encoder(options: TrackGraphTransformerConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        options.hidden_dim,
        options.head_count,
        options.hidden_dim * 4,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, options.layer_count, enable_nested_tensor=False)
