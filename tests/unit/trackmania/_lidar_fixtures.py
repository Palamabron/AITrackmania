"""Shared geometry fixture for lidar contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.geometry import build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest


def _asset(tmp_path: Path, *, lookahead_points: int = 60) -> Path:
    # Dense enough that opposite-boundary nearest neighbours stay on-station.
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(0, 11)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    (tmp_path / "trackmaniarl-test.Map.Gbx").write_bytes(b"trackmaniarl-test-map")
    return build_geometry_asset(
        GeometryBuildRequest(
            tmp_path / "trackmaniarl-test.npz",
            tmp_path / "left.npy",
            tmp_path / "right.npy",
            "trackmaniarl-test",
            tmp_path / "trackmaniarl-test.Map.Gbx",
            lookahead_points=lookahead_points,
        )
    )
