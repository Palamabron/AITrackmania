"""Frame and recurrent track-geometry encoders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Required, TypedDict, Unpack, cast, runtime_checkable

import torch
from torch import nn


class _TelemetryNormalization(Enum):
    NONE = "none"
    LAYER = "layer"


def _telemetry_encoder(
    input_dim: int, hidden_dim: int, normalization: _TelemetryNormalization
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
    if normalization is _TelemetryNormalization.LAYER:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.SiLU())
    return nn.Sequential(*layers)


@runtime_checkable
class ObservationEncoder(Protocol):
    """A replaceable module that maps a batched observation to feature vectors."""

    output_dim: int

    def __call__(self, observation: torch.Tensor) -> torch.Tensor: ...


def _validate_temporal_window(history_length: int, burn_in: int) -> None:
    if history_length < 2:
        raise ValueError("temporal encoder requires history_length >= 2")
    if not 0 <= burn_in < history_length:
        raise ValueError("burn_in must be in [0, history_length)")


def _deterministic_adaptive_mean_pool(value: torch.Tensor, bins: int) -> torch.Tensor:
    points = value.shape[-1]
    pooled = []
    for index in range(bins):
        start = index * points // bins
        stop = ((index + 1) * points + bins - 1) // bins
        pooled.append(value[..., start:stop].mean(dim=-1))
    return torch.stack(pooled, dim=-1)


class _TrackGeometryKwargs(TypedDict, total=False):
    hidden_dim: int
    output_dim: int
    spatial_bins: int
    telemetry_group_dims: tuple[int, ...] | None
    telemetry_layer_norm: bool


@dataclass(frozen=True, slots=True)
class TrackGeometryOptions:
    hidden_dim: int = 192
    output_dim: int = 256
    spatial_bins: int = 0
    telemetry_group_dims: tuple[int, ...] | None = None
    telemetry_layer_norm: bool = True


class _TemporalTrackGeometryKwargs(_TrackGeometryKwargs, total=False):
    history_length: Required[int]
    burn_in: int


@dataclass(frozen=True, slots=True)
class TemporalTrackGeometryOptions:
    history_length: int
    hidden_dim: int = 192
    output_dim: int = 256
    spatial_bins: int = 0
    burn_in: int = 0
    telemetry_group_dims: tuple[int, ...] | None = None
    telemetry_layer_norm: bool = True


@dataclass(frozen=True, slots=True)
class _TemporalInput:
    track: torch.Tensor
    telemetry: torch.Tensor | None
    mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _TemporalDimensions:
    history_length: int
    channels: int
    telemetry_dim: int


@dataclass(frozen=True, slots=True)
class _FrameWindow:
    start: int = 0
    stop: int | None = None


@dataclass(frozen=True, slots=True)
class _FlatFrames:
    track: torch.Tensor
    telemetry: torch.Tensor | None
    mask: torch.Tensor | None
    batch_size: int
    history_length: int


def _flatten_temporal_input(
    values: _TemporalInput, window: _FrameWindow, telemetry_dim: int
) -> _FlatFrames:
    track = values.track[:, window.start : window.stop]
    batch, history = track.shape[:2]
    flat_track = track.reshape(batch * history, *track.shape[2:])
    flat_telemetry = (
        values.telemetry[:, window.start : window.stop].reshape(batch * history, telemetry_dim)
        if values.telemetry is not None and telemetry_dim
        else None
    )
    flat_mask = (
        values.mask[:, window.start : window.stop].reshape(batch * history, values.mask.shape[-1])
        if values.mask is not None
        else None
    )
    return _FlatFrames(flat_track, flat_telemetry, flat_mask, batch, history)


def _validate_temporal_input(values: _TemporalInput, dimensions: _TemporalDimensions) -> None:
    track = values.track
    expected = (dimensions.history_length, dimensions.channels)
    if track.ndim != 4 or track.shape[1:3] != expected:
        raise ValueError(
            "temporal track must have shape "
            f"(batch, {dimensions.history_length}, {dimensions.channels}, points)"
        )
    batch, history, _, points = track.shape
    telemetry_shape = (batch, history, dimensions.telemetry_dim)
    if dimensions.telemetry_dim and (
        values.telemetry is None or values.telemetry.shape != telemetry_shape
    ):
        raise ValueError(f"temporal telemetry must have shape {telemetry_shape}")
    if values.mask is not None and values.mask.shape != (batch, history, points):
        raise ValueError("temporal mask must have shape (batch, history, points)")


def _telemetry_groups(telemetry_dim: int, options: TrackGeometryOptions) -> tuple[int, ...]:
    groups = options.telemetry_group_dims or ((telemetry_dim,) if telemetry_dim else ())
    if any(group < 1 for group in groups) or sum(groups) != telemetry_dim:
        raise ValueError("telemetry groups must be positive and sum to telemetry_dim")
    return groups


def _frame_encoder(
    channels: int,
    telemetry_dim: int,
    options: TemporalTrackGeometryOptions,
) -> TrackGeometryEncoder:
    return TrackGeometryEncoder(
        channels,
        telemetry_dim,
        hidden_dim=options.hidden_dim,
        output_dim=options.output_dim,
        spatial_bins=options.spatial_bins,
        telemetry_group_dims=options.telemetry_group_dims,
        telemetry_layer_norm=options.telemetry_layer_norm,
    )


class TrackGeometryEncoder(nn.Module):
    """Conv1D encoder for equally spaced, car-local track-boundary samples."""

    def __init__(
        self,
        channels: int,
        telemetry_dim: int,
        **kwargs: Unpack[_TrackGeometryKwargs],
    ) -> None:
        super().__init__()
        options = TrackGeometryOptions(**kwargs)
        if channels < 1 or telemetry_dim < 0 or options.spatial_bins < 0:
            raise ValueError("channels must be positive and telemetry_dim non-negative")
        self.channels, self.telemetry_dim = channels, telemetry_dim
        self.telemetry_group_dims = _telemetry_groups(telemetry_dim, options)
        self.output_dim = options.output_dim
        self.spatial_bins = options.spatial_bins
        self.track = self._track_network(channels, options.hidden_dim)
        self.track_attention = self._attention_network(options.hidden_dim)
        self.telemetry = self._telemetry_network(options)
        self.projection = self._projection(options.hidden_dim, options.output_dim)

    @staticmethod
    def _track_network(channels: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(channels, hidden_dim // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    @staticmethod
    def _attention_network(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

    def _telemetry_network(self, options: TrackGeometryOptions) -> nn.Module:
        normalization = (
            _TelemetryNormalization.LAYER
            if options.telemetry_layer_norm
            else _TelemetryNormalization.NONE
        )
        return nn.ModuleList(
            _telemetry_encoder(group, options.hidden_dim, normalization)
            for group in self.telemetry_group_dims
        )

    def _projection(self, hidden_dim: int, output_dim: int) -> nn.Sequential:
        track_feature_count = 1 + self.spatial_bins
        joined = hidden_dim * (track_feature_count + len(self.telemetry_group_dims))
        return nn.Sequential(nn.Linear(joined, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

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
        clean_track = self._validate_track(track)
        encoded_track = self.track(clean_track)
        attention = self._attention(encoded_track, mask)
        features = [(encoded_track * attention.unsqueeze(1)).sum(dim=2)]
        if self.spatial_bins:
            pooled = _deterministic_adaptive_mean_pool(encoded_track, self.spatial_bins)
            features.append(pooled.flatten(1))
        features.extend(self._telemetry_features(telemetry, clean_track.shape[0]))
        return cast(torch.Tensor, self.projection(torch.cat(features, dim=-1)))

    def _validate_track(self, track: torch.Tensor) -> torch.Tensor:
        if track.ndim != 3 or track.shape[1] != self.channels:
            shape = f"(batch, {self.channels}, points)"
            raise ValueError(f"track must have shape {shape}, got {tuple(track.shape)}")
        return torch.nan_to_num(track.float())

    def _attention(self, encoded_track: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attention_logits = self.track_attention(encoded_track).squeeze(1)
        if mask is not None:
            if mask.shape != (encoded_track.shape[0], encoded_track.shape[2]):
                raise ValueError("track mask must have shape (batch, points)")
            valid = mask.to(dtype=torch.bool)
            attention_logits = attention_logits.masked_fill(
                ~valid, torch.finfo(attention_logits.dtype).min
            )
        attention = torch.softmax(attention_logits, dim=1)
        if mask is not None:
            attention = attention * valid.to(attention.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return attention

    def _telemetry_features(
        self, telemetry: torch.Tensor | None, batch_size: int
    ) -> list[torch.Tensor]:
        if not self.telemetry_dim:
            return []
        if telemetry is None or telemetry.shape != (batch_size, self.telemetry_dim):
            shape = f"(batch, {self.telemetry_dim})"
            raise ValueError(f"telemetry must have shape {shape} when configured")
        clean = torch.nan_to_num(telemetry.float())
        parts = torch.split(clean, list(self.telemetry_group_dims), dim=-1)
        encoders = cast(nn.ModuleList, self.telemetry)
        return [encoder(part) for encoder, part in zip(encoders, parts, strict=True)]


class TemporalTrackGeometryEncoder(nn.Module):
    """Encode an explicit frame history with a gated recurrent state."""

    def __init__(
        self,
        channels: int,
        telemetry_dim: int,
        **kwargs: Unpack[_TemporalTrackGeometryKwargs],
    ) -> None:
        super().__init__()
        options = TemporalTrackGeometryOptions(**kwargs)
        _validate_temporal_window(options.history_length, options.burn_in)
        self.channels = channels
        self.telemetry_dim = telemetry_dim
        self.history_length = options.history_length
        self.output_dim = options.output_dim
        self.burn_in = options.burn_in
        self.frame = _frame_encoder(channels, telemetry_dim, options)
        self.recurrent = nn.GRU(options.output_dim, options.output_dim, batch_first=True)
        self.normalization = nn.LayerNorm(options.output_dim)

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

        values = _TemporalInput(track, telemetry, mask)
        self._validate_temporal_input(values)
        hidden = self._burn_in_hidden(values)
        encoded = self._encode_frames(values, _FrameWindow(start=self.burn_in))
        recurrent, _ = self.recurrent(encoded, hidden)
        return cast(torch.Tensor, self.normalization(recurrent))

    def _validate_temporal_input(self, values: _TemporalInput) -> None:
        dimensions = _TemporalDimensions(self.history_length, self.channels, self.telemetry_dim)
        _validate_temporal_input(values, dimensions)

    def _burn_in_hidden(self, values: _TemporalInput) -> torch.Tensor | None:
        if not self.burn_in:
            return None
        with torch.no_grad():
            context = self._encode_frames(values, _FrameWindow(stop=self.burn_in))
            _, hidden = self.recurrent(context)
        return cast(torch.Tensor, hidden)

    def _encode_frames(
        self,
        values: _TemporalInput,
        window: _FrameWindow,
    ) -> torch.Tensor:
        flat = _flatten_temporal_input(values, window, self.telemetry_dim)
        return cast(
            torch.Tensor,
            self.frame(flat.track, flat.telemetry, flat.mask).reshape(
                flat.batch_size, flat.history_length, self.output_dim
            ),
        )
