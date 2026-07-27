"""First-party 78-action IQN model over lidar and telemetry observations."""

from __future__ import annotations

from collections import deque
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

    def __init__(
        self,
        *,
        telemetry_dim: int,
        history_length: int,
        spatial_bins: int,
        burn_in: int,
    ) -> None:
        super().__init__()
        encoder: nn.Module
        if history_length == 1:
            encoder = TrackGeometryEncoder(
                4,
                telemetry_dim,
                output_dim=self.output_dim,
                spatial_bins=spatial_bins,
            )
        else:
            encoder = TemporalTrackGeometryEncoder(
                4,
                telemetry_dim,
                history_length=history_length,
                output_dim=self.output_dim,
                spatial_bins=spatial_bins,
                burn_in=burn_in,
            )
        self.encoder = encoder

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        lidar, telemetry, mask = self._unpack(observation)
        return cast(torch.Tensor, self.encoder(lidar, telemetry, mask))

    def encode_steps(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        encode_steps = getattr(self.encoder, "encode_steps", None)
        if not callable(encode_steps):
            raise TypeError("single-frame lidar encoder has no per-step sequence features")
        lidar, telemetry, mask = self._unpack(observation)
        return cast(torch.Tensor, encode_steps(lidar, telemetry, mask))

    @staticmethod
    def _unpack(
        observation: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {"lidar", "lidar_mask", "telemetry"}
        if set(observation) != required:
            raise ValueError(f"lidar IQN observation keys must be {sorted(required)}")
        return observation["lidar"], observation["telemetry"], observation["lidar_mask"]


class LidarIqnModel(DiscreteQuantileNetwork):
    """Dueling IQN network for the fixed 13 x 2 x 3 TrackMania action table."""

    def __init__(
        self,
        *,
        cosine_count: int = 64,
        telemetry_dim: int = LidarFeaturePipeline.telemetry_dim,
        history_length: int = 1,
        spatial_bins: int = 0,
        burn_in: int = 0,
    ) -> None:
        action_count, _ = build_brake_tap_action_table()
        if telemetry_dim < 1 or history_length < 1:
            raise ValueError("telemetry_dim and history_length must be positive")
        self.history_length = history_length
        self.sequence_burn_in = burn_in if history_length > 1 else 0
        encoder = _LidarObservationEncoder(
            telemetry_dim=telemetry_dim,
            history_length=history_length,
            spatial_bins=spatial_bins,
            burn_in=burn_in,
        )
        super().__init__(encoder, encoder.output_dim, action_count, cosine_count, dueling=True)
        self.register_buffer(
            "exploration_action_weights",
            torch.from_numpy(build_brake_tap_exploration_weights()),
            persistent=False,
        )
        self._policy_history: deque[Mapping[str, torch.Tensor]] = deque(maxlen=history_length)

    def observation_is_single(self, observation: Mapping[str, torch.Tensor]) -> bool:
        expected = 2 if self.history_length == 1 else 3
        return observation["lidar"].ndim == expected

    def prepare_policy_observation(
        self, observation: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        if self.history_length == 1 or observation["lidar"].ndim != 2:
            return observation
        self._policy_history.append(observation)
        frames = [self._policy_history[0]] * (
            self.history_length - len(self._policy_history)
        ) + list(self._policy_history)
        return {
            key: torch.stack([frame[key] for frame in frames])
            for key in ("lidar", "lidar_mask", "telemetry")
        }

    def reset_policy_state(self) -> None:
        self._policy_history.clear()


class LidarIqnModelFactory:
    def __init__(
        self,
        cosine_count: int = 64,
        telemetry_dim: int = LidarFeaturePipeline.telemetry_dim,
        history_length: int = 1,
        spatial_bins: int = 0,
        burn_in: int = 0,
    ) -> None:
        self.cosine_count = cosine_count
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length
        self.spatial_bins = spatial_bins
        self.burn_in = burn_in

    def build(self) -> LidarIqnModel:
        return LidarIqnModel(
            cosine_count=self.cosine_count,
            telemetry_dim=self.telemetry_dim,
            history_length=self.history_length,
            spatial_bins=self.spatial_bins,
            burn_in=self.burn_in,
        )
