"""Experimental Mamba model for lidar frame histories."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.models.encoders import TemporalMambaTrackGeometryEncoder
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.lidar import LidarDiscreteQuantileModel


class LidarMambaModel(LidarDiscreteQuantileModel):
    """Lidar model using Mamba for explicit frame histories."""

    def __init__(
        self,
        *,
        cosine_count: int = 64,
        telemetry_dim: int = LidarFeaturePipeline.telemetry_dim,
        history_length: int = 16,
        spatial_bins: int = 0,
        burn_in: int = 4,
        action_ids: tuple[int, ...] | None = None,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        telemetry_layer_norm: bool = True,
        legacy_telemetry_layout: bool = False,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
        masked_telemetry_indices: tuple[int, ...] = (),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_cls: type[nn.Module] | None = None,
    ) -> None:
        if history_length < 2:
            raise ValueError("Mamba requires history_length >= 2")
        super().__init__(
            cosine_count=cosine_count,
            telemetry_dim=telemetry_dim,
            history_length=history_length,
            spatial_bins=spatial_bins,
            burn_in=burn_in,
            action_ids=action_ids,
            lidar_channels=lidar_channels,
            telemetry_group_dims=telemetry_group_dims,
            telemetry_layer_norm=telemetry_layer_norm,
            legacy_telemetry_layout=legacy_telemetry_layout,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_output_dim=encoder_output_dim,
            masked_telemetry_indices=masked_telemetry_indices,
            _temporal_encoder_cls=TemporalMambaTrackGeometryEncoder,
            _temporal_encoder_kwargs={
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "mamba_cls": mamba_cls,
            },
        )


@dataclass(frozen=True, slots=True)
class LidarMambaModelFactory:
    """RunSpec factory for the experimental Mamba lidar model."""

    model_contract = ModelContract.DISCRETE_QUANTILE

    cosine_count: int = 64
    telemetry_dim: int = LidarFeaturePipeline.telemetry_dim
    history_length: int = 16
    spatial_bins: int = 0
    burn_in: int = 4
    action_ids: tuple[int, ...] | None = None
    lidar_channels: int = 4
    telemetry_group_dims: tuple[int, ...] | None = None
    telemetry_layer_norm: bool = True
    legacy_telemetry_layout: bool = False
    encoder_hidden_dim: int = 192
    encoder_output_dim: int = 256
    masked_telemetry_indices: tuple[int, ...] = ()
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    def build(self) -> LidarMambaModel:
        return LidarMambaModel(
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
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
            masked_telemetry_indices=self.masked_telemetry_indices,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
