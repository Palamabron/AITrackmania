"""Contract tests for the Mamba-backed lidar IQN path (fake temporal layer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tmrl.algorithms.implicit_quantile_q_learning import ImplicitQuantileQLearning
from tmrl.core.data import TrainingBatch
from tmrl.models.encoders.track_geometry import (
    TemporalMambaTrackGeometryEncoder,
    require_mamba_layer,
)
from tmrl.trackmania.features import LidarFeaturePipeline
from tmrl.trackmania.geometry import build_geometry_asset
from tmrl.trackmania.iqn import LidarIqnMambaModelFactory
from tmrl.trackmania.iqn_mamba import LidarIqnMambaModel


class _FakeMamba(nn.Module):
    def __init__(self, d_model: int, **kwargs: object) -> None:
        del kwargs
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values)


def _geometry_asset(tmp_path: Path) -> Path:
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(0, 11)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    (tmp_path / "test-3.Map.Gbx").write_bytes(b"test-3-map")
    return build_geometry_asset(
        tmp_path / "test-3.npz",
        tmp_path / "left.npy",
        tmp_path / "right.npy",
        map_uid="test-3",
        map_path=tmp_path / "test-3.Map.Gbx",
    )


def test_require_mamba_layer_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(RuntimeError, match="tmrl\\[mamba\\]"):
        require_mamba_layer()


def test_temporal_mamba_encode_steps_respects_burn_in() -> None:
    encoder = TemporalMambaTrackGeometryEncoder(
        4,
        26,
        history_length=8,
        burn_in=2,
        spatial_bins=2,
        mamba_cls=_FakeMamba,
    )
    batch = 3
    points = 16
    track = torch.randn(batch, 8, 4, points)
    telemetry = torch.randn(batch, 8, 26)
    mask = torch.ones(batch, 8, points)
    features = encoder.encode_steps(track, telemetry, mask)

    assert features.shape == (batch, 6, 256)
    assert torch.isfinite(features).all()
    assert encoder(track, telemetry, mask).shape == (batch, 256)


def test_lidar_iqn_mamba_factory_builds_with_fake_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tmrl.models.encoders.track_geometry.require_mamba_layer",
        lambda: _FakeMamba,
    )
    model = LidarIqnMambaModelFactory(
        cosine_count=8,
        telemetry_dim=26,
        history_length=4,
        burn_in=1,
        spatial_bins=2,
        d_state=8,
        d_conv=4,
        expand=2,
    ).build()

    assert model.history_length == 4
    assert model.sequence_burn_in == 1
    assert isinstance(model, LidarIqnMambaModel)


def test_temporal_mamba_iqn_sequence_update(tmp_path: Path) -> None:
    pipeline = LidarFeaturePipeline(
        _geometry_asset(tmp_path),
        expected_map_uid="test-3",
        history_length=1,
        include_track_relative=True,
    )
    raw = np.zeros(33, dtype=np.float32)
    raw[10] = 1.0
    single = pipeline.transform_observation(raw)
    model = LidarIqnMambaModel(
        cosine_count=8,
        telemetry_dim=26,
        history_length=4,
        burn_in=1,
        spatial_bins=2,
        mamba_cls=_FakeMamba,
    )
    learner = ImplicitQuantileQLearning(
        model,
        train_quantile_count=8,
        target_quantile_count=8,
        evaluation_quantile_count=8,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    policy = learner.policy()
    assert isinstance(policy.act(single, deterministic=True), int)
    policy.reset_episode()

    observations = {
        key: value.view(1, 1, *value.shape).repeat(2, 4, *([1] * value.ndim))
        for key, value in single.items()
    }
    batch = TrainingBatch(
        data=observations,
        observations=observations,
        actions=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        rewards=torch.tensor([[0.0, 1.0, 0.5, 0.2], [0.0, 2.0, 0.1, 0.3]]),
        next_observations=observations,
        terminated=torch.zeros((2, 4), dtype=torch.bool),
        truncated=torch.zeros((2, 4), dtype=torch.bool),
        bootstrap_discounts=torch.full((2, 4), 0.99),
        transition_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        importance_weights=torch.ones(2),
        masks=torch.ones(2, 4, dtype=torch.bool),
        metadata={
            "priority_transition_ids": (4, 8),
            "gamma": 0.99,
            "n_step": 1,
        },
    )
    metrics, priorities = learner.update(batch)

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert list(priorities.transition_ids) == [4, 8]
