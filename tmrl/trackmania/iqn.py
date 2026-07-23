"""First-party 78-action IQN model over lidar and telemetry observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from tmrl.models.critics import DiscreteQuantileNetwork
from tmrl.models.encoders.track_geometry import (
    TemporalTrackGeometryEncoder,
    TrackGeometryEncoder,
)
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    build_brake_tap_exploration_weights,
)
from tmrl.trackmania.features import LidarFeaturePipeline


class _LidarObservationEncoder(nn.Module):
    output_dim = 256

    def __init__(self, *, telemetry_dim: int, history_length: int) -> None:
        super().__init__()
        encoder: nn.Module
        if history_length == 1:
            encoder = TrackGeometryEncoder(4, telemetry_dim, output_dim=self.output_dim)
        else:
            encoder = TemporalTrackGeometryEncoder(
                4,
                telemetry_dim,
                history_length=history_length,
                output_dim=self.output_dim,
            )
        self.encoder = encoder

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

    def __init__(
        self,
        *,
        cosine_count: int = 64,
        telemetry_dim: int = LidarFeaturePipeline.telemetry_dim,
        history_length: int = 1,
    ) -> None:
        action_count, _ = build_brake_tap_action_table()
        if telemetry_dim < 1 or history_length < 1:
            raise ValueError("telemetry_dim and history_length must be positive")
        self.history_length = history_length
        encoder = _LidarObservationEncoder(
            telemetry_dim=telemetry_dim,
            history_length=history_length,
        )
        super().__init__(encoder, encoder.output_dim, action_count, cosine_count, dueling=True)
        self.register_buffer(
            "exploration_action_weights",
            torch.from_numpy(build_brake_tap_exploration_weights()),
            persistent=False,
        )

    def observation_is_single(self, observation: Mapping[str, torch.Tensor]) -> bool:
        expected = 2 if self.history_length == 1 else 3
        return observation["lidar"].ndim == expected


class LidarIqnModelFactory:
    def __init__(
        self,
        cosine_count: int = 64,
        telemetry_dim: int = LidarFeaturePipeline.telemetry_dim,
        history_length: int = 1,
    ) -> None:
        self.cosine_count = cosine_count
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length

    def build(self) -> LidarIqnModel:
        return LidarIqnModel(
            cosine_count=self.cosine_count,
            telemetry_dim=self.telemetry_dim,
            history_length=self.history_length,
        )
