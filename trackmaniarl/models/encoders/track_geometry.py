"""Track-geometry encoders for lidar and local-frame boundary observations."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn


def _telemetry_encoder(input_dim: int, hidden_dim: int, use_layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.SiLU())
    return nn.Sequential(*layers)


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
        spatial_bins: int = 0,
        telemetry_group_dims: tuple[int, ...] | None = None,
        telemetry_layer_norm: bool = True,
        legacy_telemetry_layout: bool = False,
    ) -> None:
        super().__init__()
        if channels < 1 or telemetry_dim < 0 or spatial_bins < 0:
            raise ValueError("channels must be positive and telemetry_dim non-negative")
        groups = telemetry_group_dims or ((telemetry_dim,) if telemetry_dim else ())
        if any(group < 1 for group in groups) or sum(groups) != telemetry_dim:
            raise ValueError("telemetry groups must be positive and sum to telemetry_dim")
        if legacy_telemetry_layout and len(groups) != 1:
            raise ValueError("legacy telemetry layout requires one telemetry group")
        self.channels = channels
        self.telemetry_dim = telemetry_dim
        self.telemetry_group_dims = groups
        self.legacy_telemetry_layout = legacy_telemetry_layout
        self.output_dim = output_dim
        self.spatial_bins = spatial_bins
        self.track = nn.Sequential(
            nn.Conv1d(channels, hidden_dim // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.track_attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.telemetry: nn.Module
        if legacy_telemetry_layout:
            self.telemetry = _telemetry_encoder(telemetry_dim, hidden_dim, telemetry_layer_norm)
        else:
            self.telemetry = nn.ModuleList(
                _telemetry_encoder(group, hidden_dim, telemetry_layer_norm) for group in groups
            )
        track_feature_count = 1 + spatial_bins
        joined = hidden_dim * (track_feature_count + len(groups))
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

        if track.device.type == "cpu" and torch.is_autocast_enabled("cpu"):
            with torch.autocast(device_type="cpu", enabled=False):
                return self._encode(track, telemetry, mask)
        return self._encode(track, telemetry, mask)

    def _encode(
        self,
        track: torch.Tensor,
        telemetry: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode geometry outside unsupported CPU reduced-precision kernels."""

        if track.ndim != 3 or track.shape[1] != self.channels:
            raise ValueError(
                f"track must have shape (batch, {self.channels}, points), got {tuple(track.shape)}"
            )
        track = torch.nan_to_num(track.float())
        encoded_track = self.track(track)
        attention_logits = self.track_attention(encoded_track).squeeze(1)
        if mask is not None:
            if mask.shape != (track.shape[0], track.shape[2]):
                raise ValueError("track mask must have shape (batch, points)")
            valid = mask.to(dtype=torch.bool)
            attention_logits = attention_logits.masked_fill(
                ~valid, torch.finfo(attention_logits.dtype).min
            )
        attention = torch.softmax(attention_logits, dim=1)
        if mask is not None:
            attention = attention * valid.to(attention.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        track_features = (encoded_track * attention.unsqueeze(1)).sum(dim=2)
        features = [track_features]
        if self.spatial_bins:
            ordered = torch.nn.functional.adaptive_avg_pool1d(
                encoded_track, self.spatial_bins
            ).flatten(1)
            features.append(ordered)
        if self.telemetry_dim:
            if telemetry is None or telemetry.shape != (track.shape[0], self.telemetry_dim):
                raise ValueError(
                    f"telemetry must have shape (batch, {self.telemetry_dim}) when configured"
                )
            telemetry = torch.nan_to_num(telemetry.float())
            if self.legacy_telemetry_layout:
                features.append(self.telemetry(telemetry))
            else:
                parts = torch.split(telemetry, list(self.telemetry_group_dims), dim=-1)
                encoders = cast(nn.ModuleList, self.telemetry)
                features.extend(
                    encoder(part) for encoder, part in zip(encoders, parts, strict=True)
                )
        return cast(torch.Tensor, self.projection(torch.cat(features, dim=-1)))


class TemporalTrackGeometryEncoder(nn.Module):
    """Encode an explicit frame history with a gated recurrent state."""

    def __init__(
        self,
        channels: int,
        telemetry_dim: int,
        *,
        history_length: int,
        hidden_dim: int = 192,
        output_dim: int = 256,
        spatial_bins: int = 0,
        burn_in: int = 0,
        telemetry_group_dims: tuple[int, ...] | None = None,
        telemetry_layer_norm: bool = True,
        legacy_telemetry_layout: bool = False,
    ) -> None:
        super().__init__()
        if history_length < 2 or not 0 <= burn_in < history_length:
            raise ValueError("temporal encoder requires history_length >= 2")
        self.channels = channels
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length
        self.output_dim = output_dim
        self.burn_in = burn_in
        self.frame = TrackGeometryEncoder(
            channels,
            telemetry_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            spatial_bins=spatial_bins,
            telemetry_group_dims=telemetry_group_dims,
            telemetry_layer_norm=telemetry_layer_norm,
            legacy_telemetry_layout=legacy_telemetry_layout,
        )
        self.recurrent = nn.GRU(output_dim, output_dim, batch_first=True)
        self.normalization = nn.LayerNorm(output_dim)

    def forward(
        self,
        track: torch.Tensor,
        telemetry: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_steps(track, telemetry, mask)[:, -1]

    def encode_steps(
        self,
        track: torch.Tensor,
        telemetry: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized recurrent features for every post-burn-in step."""

        if track.ndim != 4 or track.shape[1:3] != (
            self.history_length,
            self.channels,
        ):
            raise ValueError(
                "temporal track must have shape "
                f"(batch, {self.history_length}, {self.channels}, points)"
            )
        batch, history, _, points = track.shape
        if self.telemetry_dim and (
            telemetry is None or telemetry.shape != (batch, history, self.telemetry_dim)
        ):
            raise ValueError(
                f"temporal telemetry must have shape (batch, {history}, {self.telemetry_dim})"
            )
        if mask is not None and mask.shape != (batch, history, points):
            raise ValueError("temporal mask must have shape (batch, history, points)")
        hidden = None
        if self.burn_in:
            with torch.no_grad():
                context = self._encode_frames(track, telemetry, mask, stop=self.burn_in)
                _, hidden = self.recurrent(context)
        encoded = self._encode_frames(track, telemetry, mask, start=self.burn_in)
        recurrent, _ = self.recurrent(encoded, hidden)
        return cast(torch.Tensor, self.normalization(recurrent))

    def _encode_frames(
        self,
        track: torch.Tensor,
        telemetry: torch.Tensor | None,
        mask: torch.Tensor | None,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> torch.Tensor:
        window = track[:, start:stop]
        batch, history = window.shape[:2]
        flat_track = window.reshape(batch * history, *window.shape[2:])
        flat_telemetry = (
            telemetry[:, start:stop].reshape(batch * history, self.telemetry_dim)
            if telemetry is not None and self.telemetry_dim
            else None
        )
        flat_mask = (
            mask[:, start:stop].reshape(batch * history, mask.shape[-1])
            if mask is not None
            else None
        )
        return cast(
            torch.Tensor,
            self.frame(flat_track, flat_telemetry, flat_mask).reshape(
                batch, history, self.output_dim
            ),
        )
