"""First-party 78-action IQN model over lidar and telemetry observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from tmrl.models.critics import DiscreteQuantileNetwork
from tmrl.models.encoders.track_geometry import TrackGeometryEncoder
from tmrl.trackmania.actions import build_brake_tap_action_table
from tmrl.trackmania.features import LidarFeaturePipelineV1


class _LidarObservationEncoder(nn.Module):
    output_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TrackGeometryEncoder(
            2, LidarFeaturePipelineV1.telemetry_dim, output_dim=self.output_dim
        )

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = {"lidar", "lidar_mask", "telemetry"}
        if set(observation) != required:
            raise ValueError(f"lidar IQN observation keys must be {sorted(required)}")
        return cast(
            torch.Tensor,
            self.encoder(observation["lidar"], observation["telemetry"], observation["lidar_mask"]),
        )


class LidarIqnModel(DiscreteQuantileNetwork):
    """Dueling IQN network for the fixed 13 x 2 x 3 TrackMania action table."""

    def __init__(self, *, cosine_count: int = 64) -> None:
        action_count, _ = build_brake_tap_action_table()
        encoder = _LidarObservationEncoder()
        super().__init__(encoder, encoder.output_dim, action_count, cosine_count, dueling=True)


class LidarIqnModelFactory:
    def __init__(self, cosine_count: int = 64) -> None:
        self.cosine_count = cosine_count

    def build(self) -> LidarIqnModel:
        return LidarIqnModel(cosine_count=self.cosine_count)
