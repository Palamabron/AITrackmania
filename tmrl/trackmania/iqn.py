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
    select_brake_tap_actions,
    select_brake_tap_exploration_weights,
)
from tmrl.trackmania.features import LidarFeaturePipeline


class _LidarObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        telemetry_dim: int,
        history_length: int,
        spatial_bins: int,
        burn_in: int,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        hidden_dim: int = 192,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        encoder: nn.Module
        if history_length == 1:
            encoder = TrackGeometryEncoder(
                lidar_channels,
                telemetry_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                spatial_bins=spatial_bins,
                telemetry_group_dims=telemetry_group_dims,
            )
        else:
            encoder = TemporalTrackGeometryEncoder(
                lidar_channels,
                telemetry_dim,
                history_length=history_length,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                spatial_bins=spatial_bins,
                burn_in=burn_in,
                telemetry_group_dims=telemetry_group_dims,
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
        action_ids: tuple[int, ...] | None = None,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
    ) -> None:
        action_count, _ = select_brake_tap_actions(action_ids)
        if (
            telemetry_dim < 1
            or history_length < 1
            or lidar_channels < 1
            or encoder_hidden_dim < 1
            or encoder_output_dim < 1
        ):
            raise ValueError("model dimensions and history_length must be positive")
        self.history_length = history_length
        self.sequence_burn_in = burn_in if history_length > 1 else 0
        encoder = _LidarObservationEncoder(
            telemetry_dim=telemetry_dim,
            history_length=history_length,
            spatial_bins=spatial_bins,
            burn_in=burn_in,
            lidar_channels=lidar_channels,
            telemetry_group_dims=telemetry_group_dims,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
        )
        super().__init__(encoder, encoder.output_dim, action_count, cosine_count, dueling=True)
        self.register_buffer(
            "exploration_action_weights",
            torch.from_numpy(select_brake_tap_exploration_weights(action_ids)),
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
        action_ids: tuple[int, ...] | None = None,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
    ) -> None:
        self.cosine_count = cosine_count
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length
        self.spatial_bins = spatial_bins
        self.burn_in = burn_in
        self.action_ids = action_ids
        self.lidar_channels = lidar_channels
        self.telemetry_group_dims = telemetry_group_dims
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim

    def build(self) -> LidarIqnModel:
        return LidarIqnModel(
            cosine_count=self.cosine_count,
            telemetry_dim=self.telemetry_dim,
            history_length=self.history_length,
            spatial_bins=self.spatial_bins,
            burn_in=self.burn_in,
            action_ids=self.action_ids,
            lidar_channels=self.lidar_channels,
            telemetry_group_dims=self.telemetry_group_dims,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
        )
