"""Shared lidar encoders and discrete quantile models."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.models.critics import DiscreteQuantileNetwork
from trackmaniarl.models.encoders.track_geometry import (
    TemporalTrackGeometryEncoder,
    TrackGeometryEncoder,
)
from trackmaniarl.trackmania.actions import (
    select_brake_tap_actions,
    select_brake_tap_exploration_weights,
)
from trackmaniarl.trackmania.features import LidarFeaturePipeline


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
        telemetry_layer_norm: bool = True,
        legacy_telemetry_layout: bool = False,
        base_telemetry_dim: int | None = None,
        auxiliary_remaining_distance_index: int | None = None,
        auxiliary_progress_index: int | None = None,
        auxiliary_start_progress: float = 0.0,
        auxiliary_residual_scale: float | None = None,
        hidden_dim: int = 192,
        output_dim: int = 256,
        temporal_encoder_cls: type[nn.Module] = TemporalTrackGeometryEncoder,
        temporal_encoder_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.telemetry_dim = telemetry_dim
        self.base_telemetry_dim = (
            telemetry_dim if base_telemetry_dim is None else base_telemetry_dim
        )
        if not 0 < self.base_telemetry_dim <= telemetry_dim:
            raise ValueError("base telemetry dimension must be inside the observation")
        if auxiliary_remaining_distance_index is not None and not (
            self.base_telemetry_dim <= auxiliary_remaining_distance_index < telemetry_dim
        ):
            raise ValueError("auxiliary remaining-distance index must be in auxiliary telemetry")
        if auxiliary_progress_index is not None and not (
            0 <= auxiliary_progress_index < self.base_telemetry_dim
        ):
            raise ValueError("auxiliary progress index must be in base telemetry")
        if not 0.0 <= auxiliary_start_progress < 1.0:
            raise ValueError("auxiliary start progress must be in [0, 1)")
        if auxiliary_residual_scale is not None and auxiliary_residual_scale <= 0.0:
            raise ValueError("auxiliary residual scale must be positive")
        self.auxiliary_remaining_distance_index = auxiliary_remaining_distance_index
        self.auxiliary_progress_index = auxiliary_progress_index
        self.auxiliary_start_progress = auxiliary_start_progress
        self.auxiliary_residual_scale = auxiliary_residual_scale
        encoder: nn.Module
        if history_length == 1:
            encoder = TrackGeometryEncoder(
                lidar_channels,
                self.base_telemetry_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                spatial_bins=spatial_bins,
                telemetry_group_dims=telemetry_group_dims,
                telemetry_layer_norm=telemetry_layer_norm,
                legacy_telemetry_layout=legacy_telemetry_layout,
            )
        else:
            encoder = temporal_encoder_cls(
                lidar_channels,
                self.base_telemetry_dim,
                history_length=history_length,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                spatial_bins=spatial_bins,
                burn_in=burn_in,
                telemetry_group_dims=telemetry_group_dims,
                telemetry_layer_norm=telemetry_layer_norm,
                legacy_telemetry_layout=legacy_telemetry_layout,
                **dict(temporal_encoder_kwargs or {}),
            )
        self.encoder = encoder
        auxiliary_dim = telemetry_dim - self.base_telemetry_dim
        self.auxiliary: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(auxiliary_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            if auxiliary_dim
            else None
        )
        if self.auxiliary is not None:
            output = self.auxiliary[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        lidar, telemetry, mask = self._unpack(observation)
        encoded = cast(
            torch.Tensor,
            self.encoder(lidar, telemetry[..., : self.base_telemetry_dim], mask),
        )
        if self.auxiliary is None:
            return encoded
        auxiliary = telemetry[..., self.base_telemetry_dim :]
        base_telemetry = telemetry[..., : self.base_telemetry_dim]
        if auxiliary.ndim == 3:
            auxiliary = auxiliary[:, -1]
            base_telemetry = base_telemetry[:, -1]
        return encoded + self._auxiliary_residual(auxiliary, base_telemetry)

    def encode_steps(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        encode_steps = getattr(self.encoder, "encode_steps", None)
        if not callable(encode_steps):
            raise TypeError("single-frame lidar encoder has no per-step sequence features")
        lidar, telemetry, mask = self._unpack(observation)
        encoded = cast(
            torch.Tensor,
            encode_steps(lidar, telemetry[..., : self.base_telemetry_dim], mask),
        )
        if self.auxiliary is None:
            return encoded
        burn_in = int(getattr(self.encoder, "burn_in", 0))
        auxiliary = telemetry[:, burn_in:, self.base_telemetry_dim :]
        base_telemetry = telemetry[:, burn_in:, : self.base_telemetry_dim]
        return encoded + self._auxiliary_residual(auxiliary, base_telemetry)

    def _auxiliary_residual(
        self, auxiliary: torch.Tensor, base_telemetry: torch.Tensor
    ) -> torch.Tensor:
        assert self.auxiliary is not None
        residual = cast(torch.Tensor, self.auxiliary(auxiliary))
        if self.auxiliary_residual_scale is not None:
            residual = torch.tanh(residual) * self.auxiliary_residual_scale
        if self.auxiliary_remaining_distance_index is not None:
            local_index = self.auxiliary_remaining_distance_index - self.base_telemetry_dim
            remaining = auxiliary[..., local_index].clamp(0.0, 1.0)
            residual = residual * (1.0 - remaining).unsqueeze(-1)
        if self.auxiliary_progress_index is None:
            return residual
        progress = base_telemetry[..., self.auxiliary_progress_index].clamp(0.0, 1.0)
        activation = (
            (progress - self.auxiliary_start_progress) / (1.0 - self.auxiliary_start_progress)
        ).clamp(0.0, 1.0)
        return residual * activation.unsqueeze(-1)

    def set_offline_pretraining(self, enabled: bool) -> None:
        if self.auxiliary is None:
            return
        for parameter in self.parameters():
            parameter.requires_grad_(not enabled)
        for parameter in self.auxiliary.parameters():
            parameter.requires_grad_(True)

    @staticmethod
    def _unpack(
        observation: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {"lidar", "lidar_mask", "telemetry"}
        if set(observation) != required:
            raise ValueError(f"lidar IQN observation keys must be {sorted(required)}")
        return observation["lidar"], observation["telemetry"], observation["lidar_mask"]


class LidarDiscreteQuantileModel(DiscreteQuantileNetwork):
    """Dueling quantile network for the fixed 13 x 2 x 3 TrackMania action table."""

    masked_telemetry_indices: torch.Tensor

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
        telemetry_layer_norm: bool = True,
        legacy_telemetry_layout: bool = False,
        base_telemetry_dim: int | None = None,
        auxiliary_remaining_distance_index: int | None = None,
        auxiliary_progress_index: int | None = None,
        auxiliary_start_progress: float = 0.0,
        auxiliary_residual_scale: float | None = None,
        train_auxiliary_only: bool = False,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
        masked_telemetry_indices: tuple[int, ...] = (),
        _temporal_encoder_cls: type[nn.Module] = TemporalTrackGeometryEncoder,
        _temporal_encoder_kwargs: Mapping[str, object] | None = None,
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
        if len(set(masked_telemetry_indices)) != len(masked_telemetry_indices) or any(
            index < 0 or index >= telemetry_dim for index in masked_telemetry_indices
        ):
            raise ValueError("masked telemetry indices must be unique and inside telemetry")
        self.history_length = history_length
        self.sequence_burn_in = burn_in if history_length > 1 else 0
        encoder = _LidarObservationEncoder(
            telemetry_dim=telemetry_dim,
            history_length=history_length,
            spatial_bins=spatial_bins,
            burn_in=burn_in,
            lidar_channels=lidar_channels,
            telemetry_group_dims=telemetry_group_dims,
            telemetry_layer_norm=telemetry_layer_norm,
            legacy_telemetry_layout=legacy_telemetry_layout,
            base_telemetry_dim=base_telemetry_dim,
            auxiliary_remaining_distance_index=auxiliary_remaining_distance_index,
            auxiliary_progress_index=auxiliary_progress_index,
            auxiliary_start_progress=auxiliary_start_progress,
            auxiliary_residual_scale=auxiliary_residual_scale,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            temporal_encoder_cls=_temporal_encoder_cls,
            temporal_encoder_kwargs=_temporal_encoder_kwargs,
        )
        super().__init__(encoder, encoder.output_dim, action_count, cosine_count, dueling=True)
        self.register_buffer(
            "masked_telemetry_indices",
            torch.tensor(masked_telemetry_indices, dtype=torch.long),
            persistent=False,
        )
        if train_auxiliary_only and encoder.auxiliary is None:
            raise ValueError("auxiliary-only training requires auxiliary telemetry")
        self.train_auxiliary_only = train_auxiliary_only
        self.register_buffer(
            "exploration_action_weights",
            torch.from_numpy(select_brake_tap_exploration_weights(action_ids)),
            persistent=False,
        )
        self._policy_history: deque[Mapping[str, torch.Tensor]] = deque(maxlen=history_length)
        self._configure_trainable_parameters(train_auxiliary_only)

    def forward(
        self, observation: Mapping[str, torch.Tensor], quantiles: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(self._masked_observation(observation), quantiles)

    def encode_sequence(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return super().encode_sequence(self._masked_observation(observation))

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

    def _masked_observation(
        self, observation: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        if not self.masked_telemetry_indices.numel():
            return observation
        return {
            "lidar": observation["lidar"],
            "lidar_mask": observation["lidar_mask"],
            "telemetry": observation["telemetry"].index_fill(
                -1, self.masked_telemetry_indices, 0.0
            ),
        }

    def set_offline_pretraining(self, enabled: bool) -> None:
        encoder = cast(_LidarObservationEncoder, self.encoder)
        if encoder.auxiliary is None:
            return
        self._configure_trainable_parameters(enabled or self.train_auxiliary_only)

    def demonstration_loss_weights(
        self,
        observation: Mapping[str, torch.Tensor],
        positions: list[int] | None,
    ) -> torch.Tensor | None:
        encoder = cast(_LidarObservationEncoder, self.encoder)
        telemetry = observation["telemetry"]
        weights: torch.Tensor | None = None
        if encoder.auxiliary_remaining_distance_index is not None:
            remaining = telemetry[..., encoder.auxiliary_remaining_distance_index].clamp(0.0, 1.0)
            weights = 1.0 - remaining
        if encoder.auxiliary_progress_index is not None:
            progress = telemetry[..., encoder.auxiliary_progress_index].clamp(0.0, 1.0)
            activation = (
                (progress - encoder.auxiliary_start_progress)
                / (1.0 - encoder.auxiliary_start_progress)
            ).clamp(0.0, 1.0)
            weights = activation if weights is None else weights * activation
        if weights is None:
            return None
        if positions is not None:
            return weights[:, positions]
        return weights[:, -1] if weights.ndim == 2 else weights

    def _configure_trainable_parameters(self, auxiliary_only: bool) -> None:
        encoder = cast(_LidarObservationEncoder, self.encoder)
        for parameter in self.parameters():
            parameter.requires_grad_(not auxiliary_only)
        if auxiliary_only and encoder.auxiliary is not None:
            for parameter in encoder.auxiliary.parameters():
                parameter.requires_grad_(True)


class LidarIqnModel(LidarDiscreteQuantileModel):
    """Lidar quantile network used by the IQN learner."""


class LidarIqnModelFactory:
    model_contract = ModelContract.DISCRETE_QUANTILE

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
        telemetry_layer_norm: bool = True,
        legacy_telemetry_layout: bool = False,
        base_telemetry_dim: int | None = None,
        auxiliary_remaining_distance_index: int | None = None,
        auxiliary_progress_index: int | None = None,
        auxiliary_start_progress: float = 0.0,
        auxiliary_residual_scale: float | None = None,
        train_auxiliary_only: bool = False,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
        masked_telemetry_indices: tuple[int, ...] = (),
    ) -> None:
        self.cosine_count = cosine_count
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length
        self.spatial_bins = spatial_bins
        self.burn_in = burn_in
        self.action_ids = action_ids
        self.lidar_channels = lidar_channels
        self.telemetry_group_dims = telemetry_group_dims
        self.telemetry_layer_norm = telemetry_layer_norm
        self.legacy_telemetry_layout = legacy_telemetry_layout
        self.base_telemetry_dim = base_telemetry_dim
        self.auxiliary_remaining_distance_index = auxiliary_remaining_distance_index
        self.auxiliary_progress_index = auxiliary_progress_index
        self.auxiliary_start_progress = auxiliary_start_progress
        self.auxiliary_residual_scale = auxiliary_residual_scale
        self.train_auxiliary_only = train_auxiliary_only
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.masked_telemetry_indices = masked_telemetry_indices

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
            telemetry_layer_norm=self.telemetry_layer_norm,
            legacy_telemetry_layout=self.legacy_telemetry_layout,
            base_telemetry_dim=self.base_telemetry_dim,
            auxiliary_remaining_distance_index=self.auxiliary_remaining_distance_index,
            auxiliary_progress_index=self.auxiliary_progress_index,
            auxiliary_start_progress=self.auxiliary_start_progress,
            auxiliary_residual_scale=self.auxiliary_residual_scale,
            train_auxiliary_only=self.train_auxiliary_only,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
            masked_telemetry_indices=self.masked_telemetry_indices,
        )
