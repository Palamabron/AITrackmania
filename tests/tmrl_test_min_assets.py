"""Minimal reward/track pickles for ``map_name`` fallback ``tmrl-test`` (pytest / isolated HOME)."""

from __future__ import annotations

import pickle
from pathlib import Path


def write_min_tmrl_test_pickles(base: Path) -> None:
    """Write tiny trajectories so default LIDAR config passes asset checks."""
    reward_dir = base / "reward"
    track_dir = base / "track"
    reward_dir.mkdir(parents=True, exist_ok=True)
    track_dir.mkdir(parents=True, exist_ok=True)
    min_traj = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    min_left = [[0.0, 0.0, -4.0], [10.0, 0.0, -4.0]]
    min_right = [[0.0, 0.0, 4.0], [10.0, 0.0, 4.0]]
    with open(reward_dir / "reward_tmrl-test.pkl", "wb") as f:
        pickle.dump(min_traj, f)
    with open(track_dir / "track_tmrl-test_left.pkl", "wb") as f:
        pickle.dump(min_left, f)
    with open(track_dir / "track_tmrl-test_right.pkl", "wb") as f:
        pickle.dump(min_right, f)


def ensure_min_tmrl_test_pickles_if_missing(base: Path) -> None:
    """Create default ``tmrl-test`` pickles only where a file is already missing."""
    reward_p = base / "reward" / "reward_tmrl-test.pkl"
    left_p = base / "track" / "track_tmrl-test_left.pkl"
    right_p = base / "track" / "track_tmrl-test_right.pkl"
    if reward_p.is_file() and left_p.is_file() and right_p.is_file():
        return
    write_min_tmrl_test_pickles(base)
