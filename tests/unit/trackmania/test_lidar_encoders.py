"""Release contracts for reusable lidar encoders."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from trackmaniarl.models.encoders import (
    TemporalMambaTrackGeometryEncoder,
)
from trackmaniarl.models.encoders.track_geometry_frame import (
    _deterministic_adaptive_mean_pool,
)
from trackmaniarl.trackmania.encoders import LidarSimbaSensorEncoder


class _FakeMamba(torch.nn.Module):
    def __init__(self, d_model: int, **_: object) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor, **_: object) -> torch.Tensor:
        return cast(torch.Tensor, self.projection(values))


def _simba_encoder(history_length: int = 1) -> LidarSimbaSensorEncoder:
    return LidarSimbaSensorEncoder(
        sensor={
            "telemetry_dim": 26,
            "spatial_bins": 2,
            "hidden_dim": 16,
            "output_dim": 16,
        },
        backbone={"hidden_dim": 12, "block_count": 1, "expansion": 2},
        history_length=history_length,
    )


def _mamba_encoder() -> TemporalMambaTrackGeometryEncoder:
    return TemporalMambaTrackGeometryEncoder(
        4, 6, history_length=4, burn_in=1, spatial_bins=2, mamba_cls=_FakeMamba
    )


def test_deterministic_spatial_pool_matches_adaptive_average() -> None:
    values = torch.randn(3, 5, 90)

    pooled = _deterministic_adaptive_mean_pool(values, 12)

    expected = torch.nn.functional.adaptive_avg_pool1d(values, 12)
    torch.testing.assert_close(pooled, expected)


def test_lidar_simba_encoder_is_feedforward_and_normalized() -> None:
    encoder = _simba_encoder()

    output = encoder(
        {
            "lidar": torch.randn(3, 4, 16),
            "lidar_mask": torch.ones(3, 16),
            "telemetry": torch.randn(3, 26),
        }
    )

    assert output.shape == (3, 12)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))


def test_mamba_encoder_validates_window_and_returns_loss_steps() -> None:
    with pytest.raises(ValueError, match=r"burn_in must be in \[0, history_length\)"):
        TemporalMambaTrackGeometryEncoder(
            4,
            6,
            history_length=3,
            burn_in=3,
            mamba_cls=_FakeMamba,
        )
    encoder = _mamba_encoder()

    features = encoder.encode_steps(
        torch.randn(2, 4, 4, 16),
        torch.randn(2, 4, 6),
        torch.ones(2, 4, 16),
    )

    assert features.shape == (2, 3, 256)
    assert torch.isfinite(features).all()
