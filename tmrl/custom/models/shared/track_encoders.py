"""Shared track-point encoder modules (GTN transformer, Conv1d, spline-MLP).

Single source of truth for ``TrackGTN``, ``build_track_gtn_branch``, and
``is_gtn_encoder``.  Imported by ``sophy.py``,
``gnn_effnet_sophy.py``, and ``iqn_discrete_q_network.py``.
"""

from typing import cast

import torch
import torch.nn as nn

TRACK_CHANNELS_DEFAULT = 4
TRACK_CHANNELS_GTN = 7


def is_gtn_encoder(track_encoder: str) -> bool:
    """True when the encoder string selects the GTN (graph-transformer) path."""
    return str(track_encoder).lower() == "gtn"


class TrackGTN(nn.Module):
    """Graph Transformer encoder over ordered track point sequences."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = max(1, int(num_heads))
        if hidden_dim % self.num_heads != 0:
            self.num_heads = 1
        self.node_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_nodes, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, n = x.shape
        x = x.permute(0, 2, 1)
        h = self.node_in(x) + self.positional_embedding[:, :n, :]
        h = self.encoder(h)
        out = self.readout(h)
        return cast(torch.Tensor, out.mean(dim=1))


def build_track_gtn_branch(
    dim_track: int, hidden_dim: int, gnn_hidden: int = 64, gnn_layers: int = 3
) -> nn.Module:
    """Build a GTN-based track encoding branch."""
    assert dim_track >= TRACK_CHANNELS_GTN, "track dim must be at least 7"
    assert dim_track % TRACK_CHANNELS_GTN == 0, "track dim must be 7*N (7 channels)"
    num_nodes = dim_track // TRACK_CHANNELS_GTN
    gtn = TrackGTN(
        num_nodes=num_nodes,
        in_dim=TRACK_CHANNELS_GTN,
        hidden_dim=gnn_hidden,
        num_layers=gnn_layers,
    )
    return nn.Sequential(
        gtn,
        nn.Linear(gnn_hidden, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
    )
