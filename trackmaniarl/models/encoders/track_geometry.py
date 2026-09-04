"""Track-geometry encoders for lidar and local-frame boundary observations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Required, TypedDict, Unpack, cast

import torch
from torch import nn

from trackmaniarl.models.encoders.track_geometry_frame import (
    TemporalTrackGeometryOptions as _FrameOptions,
)
from trackmaniarl.models.encoders.track_geometry_frame import (
    _flatten_temporal_input,
    _frame_encoder,
    _FrameWindow,
    _TemporalDimensions,
    _TemporalInput,
    _validate_temporal_input,
    _validate_temporal_window,
)


def require_mamba_layer() -> type[nn.Module]:
    """Load the optional Mamba layer only when the encoder is selected."""

    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise RuntimeError(
            "Mamba temporal encoding requires the 'mamba' extra. "
            "Install it on a Linux CUDA learner with: uv sync --extra mamba"
        ) from exc
    return cast(type[nn.Module], Mamba)


@dataclass(slots=True)
class _MambaInferenceState:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    key_value_memory_dict: dict[int, object] = field(default_factory=dict)


class _TemporalMambaKwargs(TypedDict, total=False):
    history_length: Required[int]
    hidden_dim: int
    output_dim: int
    spatial_bins: int
    burn_in: int
    telemetry_group_dims: tuple[int, ...] | None
    telemetry_layer_norm: bool
    d_state: int
    d_conv: int
    expand: int
    mamba_cls: Callable[..., nn.Module] | None


@dataclass(frozen=True, slots=True)
class TemporalMambaOptions:
    history_length: int
    hidden_dim: int = 192
    output_dim: int = 256
    spatial_bins: int = 0
    burn_in: int = 0
    telemetry_group_dims: tuple[int, ...] | None = None
    telemetry_layer_norm: bool = True
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mamba_cls: Callable[..., nn.Module] | None = None


def _frame_options(options: TemporalMambaOptions) -> _FrameOptions:
    return _FrameOptions(
        history_length=options.history_length,
        hidden_dim=options.hidden_dim,
        output_dim=options.output_dim,
        spatial_bins=options.spatial_bins,
        burn_in=options.burn_in,
        telemetry_group_dims=options.telemetry_group_dims,
        telemetry_layer_norm=options.telemetry_layer_norm,
    )


class TemporalMambaTrackGeometryEncoder(nn.Module):
    """Encode frame histories with a causal, opt-in Mamba sequence layer."""

    def __init__(
        self,
        channels: int,
        telemetry_dim: int,
        **kwargs: Unpack[_TemporalMambaKwargs],
    ) -> None:
        super().__init__()
        options = TemporalMambaOptions(**kwargs)
        _validate_temporal_window(options.history_length, options.burn_in)
        if options.d_state < 1 or options.d_conv < 1 or options.expand < 1:
            raise ValueError("d_state, d_conv and expand must be positive")
        self.channels = channels
        self.telemetry_dim = telemetry_dim
        self.history_length = options.history_length
        self.output_dim = options.output_dim
        self.burn_in = options.burn_in
        self.frame = _frame_encoder(channels, telemetry_dim, _frame_options(options))
        self._initialize_temporal(options)

    def _initialize_temporal(self, options: TemporalMambaOptions) -> None:
        layer_cls = options.mamba_cls or require_mamba_layer()
        self.temporal = layer_cls(
            d_model=options.output_dim,
            d_state=options.d_state,
            d_conv=options.d_conv,
            expand=options.expand,
            layer_idx=0,
        )
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
        """Return causal features for the timesteps included in the loss."""

        self._validate_inputs(track, telemetry, mask)
        encoded = self._encode_frames(_TemporalInput(track, telemetry, mask))
        temporal = self._encode_temporal(encoded)
        expected_shape = encoded[:, self.burn_in :].shape
        if temporal.shape != expected_shape:
            raise RuntimeError("Mamba layer must preserve (batch, loss_steps, feature) shape")
        return cast(torch.Tensor, self.normalization(temporal))

    def _encode_temporal(self, encoded: torch.Tensor) -> torch.Tensor:
        if not self.burn_in:
            return cast(torch.Tensor, self.temporal(encoded))
        state = _MambaInferenceState(encoded.shape[1], encoded.shape[0])
        with torch.no_grad():
            self.temporal(encoded[:, : self.burn_in], inference_params=state)
        state.seqlen_offset = self.burn_in
        steps = [
            self.temporal(encoded[:, index : index + 1], inference_params=state)
            for index in range(self.burn_in, self.history_length)
        ]
        return torch.cat(steps, dim=1)

    def _validate_inputs(
        self,
        track: torch.Tensor,
        telemetry: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> None:
        values = _TemporalInput(track, telemetry, mask)
        dimensions = _TemporalDimensions(self.history_length, self.channels, self.telemetry_dim)
        _validate_temporal_input(values, dimensions)

    def _encode_frames(self, values: _TemporalInput) -> torch.Tensor:
        flat = _flatten_temporal_input(values, _FrameWindow(), self.telemetry_dim)
        encoded = self.frame(flat.track, flat.telemetry, flat.mask)
        return cast(
            torch.Tensor,
            encoded.reshape(flat.batch_size, flat.history_length, self.output_dim),
        )
