"""Track-geometry encoders for lidar and local-frame boundary observations."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class ObservationEncoder(Protocol):
    """A replaceable module that maps a batched observation to feature vectors."""

    output_dim: int

    def __call__(self, observation: torch.Tensor) -> torch.Tensor: ...


class TrackGeometryEncoder(nn.Module):
    """Conv1D encoder for equally spaced, car-local track-boundary samples."""

    def __init__(
        self,
        channels: int,
        telemetry_dim: int,
        *,
        hidden_dim: int = 192,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        if channels < 1 or telemetry_dim < 0:
            raise ValueError("channels must be positive and telemetry_dim non-negative")
        self.channels = channels
        self.telemetry_dim = telemetry_dim
        self.output_dim = output_dim
        self.track = nn.Sequential(
            nn.Conv1d(channels, hidden_dim // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.telemetry = (
            nn.Sequential(nn.Linear(telemetry_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
            if telemetry_dim
            else None
        )
        joined = hidden_dim * (2 if telemetry_dim else 1)
        self.projection = nn.Sequential(
            nn.Linear(joined, output_dim), nn.LayerNorm(output_dim), nn.SiLU()
        )

    def forward(
        self,
        track: torch.Tensor,
        telemetry: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``(batch, channels, points)`` geometry and optional telemetry."""

        if track.ndim != 3 or track.shape[1] != self.channels:
            raise ValueError(
                f"track must have shape (batch, {self.channels}, points), got {tuple(track.shape)}"
            )
        track = torch.nan_to_num(track.float())
        encoded_track = self.track(track)
        if mask is not None:
            if mask.shape != (track.shape[0], track.shape[2]):
                raise ValueError("track mask must have shape (batch, points)")
            valid = mask.to(encoded_track.dtype).unsqueeze(1)
            track_features = (encoded_track * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        else:
            track_features = encoded_track.mean(dim=2)
        features = [track_features]
        if self.telemetry is not None:
            if telemetry is None or telemetry.shape != (track.shape[0], self.telemetry_dim):
                raise ValueError(
                    f"telemetry must have shape (batch, {self.telemetry_dim}) when configured"
                )
            features.append(self.telemetry(torch.nan_to_num(telemetry.float())))
        return cast(torch.Tensor, self.projection(torch.cat(features, dim=-1)))
