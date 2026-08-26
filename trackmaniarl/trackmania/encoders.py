"""TrackMania sensor encoders without temporal behavior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from trackmaniarl.models.backbones import SimbaV2Backbone
from trackmaniarl.models.encoders.track_geometry import TrackGeometryEncoder
from trackmaniarl.trackmania.features import LidarFeaturePipeline


class LidarSensorEncoder(nn.Module):
    """Vectorized encoder for independent lidar and telemetry frames."""

    masked_telemetry_indices: torch.Tensor

    def __init__(
        self,
        *,
        telemetry_dim: int = LidarFeaturePipeline.telemetry_dim,
        spatial_bins: int = 0,
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
        masked_telemetry_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.telemetry_dim = telemetry_dim
        self.base_telemetry_dim = (
            telemetry_dim if base_telemetry_dim is None else base_telemetry_dim
        )
        self._validate_dimensions(masked_telemetry_indices)
        self.register_buffer(
            "masked_telemetry_indices",
            torch.tensor(masked_telemetry_indices, dtype=torch.long),
        )
        self.frame = TrackGeometryEncoder(
            lidar_channels,
            self.base_telemetry_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            spatial_bins=spatial_bins,
            telemetry_group_dims=telemetry_group_dims,
            telemetry_layer_norm=telemetry_layer_norm,
            legacy_telemetry_layout=legacy_telemetry_layout,
        )
        auxiliary_dim = telemetry_dim - self.base_telemetry_dim
        self.auxiliary = self._auxiliary(auxiliary_dim, hidden_dim, output_dim)
        self.auxiliary_remaining_distance_index = auxiliary_remaining_distance_index
        self.auxiliary_progress_index = auxiliary_progress_index
        self.auxiliary_start_progress = auxiliary_start_progress
        self.auxiliary_residual_scale = auxiliary_residual_scale
        self._validate_auxiliary()

    def forward(self, frames: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(frames) != {"lidar", "lidar_mask", "telemetry"}:
            raise ValueError("lidar frames require lidar, lidar_mask, and telemetry tensors")
        lidar = frames["lidar"]
        mask = frames["lidar_mask"]
        telemetry = frames["telemetry"].clone()
        if lidar.ndim != 3 or mask.ndim != 2 or telemetry.ndim != 2:
            raise ValueError("LidarSensorEncoder accepts independent frame batches [N, ...]")
        if self.masked_telemetry_indices.numel():
            telemetry[:, self.masked_telemetry_indices] = 0.0
        base = telemetry[:, : self.base_telemetry_dim]
        encoded = cast(torch.Tensor, self.frame(lidar, base, mask))
        if self.auxiliary is None:
            return encoded
        return encoded + self._auxiliary_residual(telemetry[:, self.base_telemetry_dim :], base)

    def set_offline_pretraining(self, enabled: bool) -> None:
        if self.auxiliary is None:
            return
        for parameter in self.parameters():
            parameter.requires_grad_(not enabled)
        for parameter in self.auxiliary.parameters():
            parameter.requires_grad_(True)

    def _auxiliary_residual(self, auxiliary: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        assert self.auxiliary is not None
        residual = cast(torch.Tensor, self.auxiliary(auxiliary))
        if self.auxiliary_residual_scale is not None:
            residual = torch.tanh(residual) * self.auxiliary_residual_scale
        if self.auxiliary_remaining_distance_index is not None:
            index = self.auxiliary_remaining_distance_index - self.base_telemetry_dim
            residual = residual * (1.0 - auxiliary[:, index].clamp(0.0, 1.0)).unsqueeze(-1)
        if self.auxiliary_progress_index is not None:
            progress = base[:, self.auxiliary_progress_index].clamp(0.0, 1.0)
            activation = (
                (progress - self.auxiliary_start_progress) / (1.0 - self.auxiliary_start_progress)
            ).clamp(0.0, 1.0)
            residual = residual * activation.unsqueeze(-1)
        return residual

    def _validate_dimensions(self, masked: tuple[int, ...]) -> None:
        if not 0 < self.base_telemetry_dim <= self.telemetry_dim:
            raise ValueError("base telemetry dimension must be inside the observation")
        if len(set(masked)) != len(masked) or any(
            index < 0 or index >= self.telemetry_dim for index in masked
        ):
            raise ValueError("masked telemetry indices must be unique and valid")

    def _validate_auxiliary(self) -> None:
        remaining = self.auxiliary_remaining_distance_index
        if remaining is not None and not self.base_telemetry_dim <= remaining < self.telemetry_dim:
            raise ValueError("remaining-distance index must select auxiliary telemetry")
        progress = self.auxiliary_progress_index
        if progress is not None and not 0 <= progress < self.base_telemetry_dim:
            raise ValueError("progress index must select base telemetry")
        if not 0.0 <= self.auxiliary_start_progress < 1.0:
            raise ValueError("auxiliary start progress must be in [0, 1)")
        if self.auxiliary_residual_scale is not None and self.auxiliary_residual_scale <= 0.0:
            raise ValueError("auxiliary residual scale must be positive")

    @staticmethod
    def _auxiliary(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential | None:
        if not input_dim:
            return None
        module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        output = cast(nn.Linear, module[-1])
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        return module


class LidarSimbaSensorEncoder(nn.Module):
    """A fixed lidar history followed by a feed-forward SimbaV2 backbone."""

    def __init__(
        self,
        sensor: Mapping[str, Any],
        backbone: Mapping[str, Any],
        history_length: int = 1,
    ) -> None:
        super().__init__()
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.sensor = LidarSensorEncoder(**sensor)
        configured = dict(backbone)
        hidden_dim = int(configured.pop("hidden_dim"))
        self.backbone = SimbaV2Backbone(
            input_dim=self.sensor.output_dim * history_length,
            hidden_dim=hidden_dim,
            **configured,
        )
        self.output_dim = hidden_dim

    def forward(self, frames: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.history_length == 1:
            return cast(torch.Tensor, self.backbone(self.sensor(frames)))
        lidar = frames.get("lidar")
        mask = frames.get("lidar_mask")
        telemetry = frames.get("telemetry")
        if lidar is None or mask is None or telemetry is None:
            raise ValueError("lidar history requires lidar, lidar_mask, and telemetry tensors")
        if (
            lidar.ndim != 4
            or mask.ndim != 3
            or telemetry.ndim != 3
            or lidar.shape[:2] != (lidar.shape[0], self.history_length)
            or mask.shape[:2] != lidar.shape[:2]
            or telemetry.shape[:2] != lidar.shape[:2]
        ):
            raise ValueError(
                "lidar history must have shapes [batch, history, channels, points], "
                "[batch, history, points], and [batch, history, telemetry]"
            )
        batch = lidar.shape[0]
        flattened = {
            "lidar": lidar.reshape(batch * self.history_length, *lidar.shape[2:]),
            "lidar_mask": mask.reshape(batch * self.history_length, *mask.shape[2:]),
            "telemetry": telemetry.reshape(batch * self.history_length, telemetry.shape[-1]),
        }
        encoded_frames = self.sensor(flattened).reshape(batch, self.history_length, -1)
        encoded = encoded_frames.flip(1).reshape(batch, -1)
        return cast(torch.Tensor, self.backbone(encoded))
