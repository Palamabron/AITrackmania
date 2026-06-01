"""Tests for finish-line stopping fixes (boundary extrapolation, near-finish grace)."""

from __future__ import annotations

import importlib.util
import pickle
import tempfile
from pathlib import Path

import numpy as np
from tmrl.tools.geometry_utils import pad_polyline_xz_straight

_REWARD_PATH = (
    Path(__file__).resolve().parents[1] / "tmrl" / "custom" / "tm" / "utils" / "compute_reward.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _straight_track(n: int = 50, spacing: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(n, dtype=np.float64) * spacing
    left = np.stack([xs, xs * 0.0 + 5.0], axis=0)
    right = np.stack([xs, xs * 0.0 - 5.0], axis=0)
    return left, right


def _repeat_padding_old(boundary: np.ndarray, n_pad: int) -> np.ndarray:
    extra = np.full((n_pad, 2), boundary.T[-1])
    return np.concatenate([boundary.T, extra], axis=0).T


BOUNDARY_LOOK_AHEAD = 15


def test_pad_polyline_xz_straight_points_are_distinct():
    left, _ = _straight_track(20)
    extended = pad_polyline_xz_straight(left, BOUNDARY_LOOK_AHEAD)
    tail = extended[:, -BOUNDARY_LOOK_AHEAD:]
    diffs = np.linalg.norm(np.diff(tail, axis=1), axis=0)
    assert np.all(diffs > 0.5), f"expected forward steps, got diffs={diffs}"


def test_pad_polyline_fixes_degenerate_look_ahead_vs_old_repeat():
    left, _ = _straight_track(30)
    new_tail = pad_polyline_xz_straight(left, BOUNDARY_LOOK_AHEAD)[:, -BOUNDARY_LOOK_AHEAD:]
    new_spread = float(np.std(new_tail[0]) + np.std(new_tail[1]))

    old_tail = _repeat_padding_old(left, BOUNDARY_LOOK_AHEAD)[:, -BOUNDARY_LOOK_AHEAD:]
    old_spread = float(np.std(old_tail[0]))

    assert new_spread > 1.0
    assert old_spread < 1e-6


def test_boundary_ahead_slice_at_track_end_is_directional():
    """Integration check: end-of-track look-ahead slice uses extrapolated, not repeated, points."""
    left, right = _straight_track(40)
    look = BOUNDARY_LOOK_AHEAD
    i_l_min = len(left[0]) - 3
    left_ext = pad_polyline_xz_straight(left, look)
    right_ext = pad_polyline_xz_straight(right, look)
    l_x = left_ext[0, i_l_min : i_l_min + look]
    l_z = left_ext[1, i_l_min : i_l_min + look]
    r_x = right_ext[0, i_l_min : i_l_min + look]
    r_z = right_ext[1, i_l_min : i_l_min + look]

    car_pos = [left[0, i_l_min], left[1, i_l_min]]
    cos_a, sin_a = 1.0, 0.0
    lx = (l_x - car_pos[0]) * cos_a - (l_z - car_pos[1]) * sin_a
    rx = (r_x - car_pos[0]) * cos_a - (r_z - car_pos[1]) * sin_a
    assert np.all(np.diff(lx) > 0)
    assert np.all(np.diff(rx) > 0)
    assert float(np.std(lx)) > 1.0


def _make_reward_fn(tmp_path: Path, n_points: int = 100):
    reward_mod = _load_module(_REWARD_PATH, "tmrl_reward_test")
    reward_function_cls = reward_mod.RewardFunction
    near_finish_progress_threshold = reward_mod.NEAR_FINISH_PROGRESS_THRESHOLD

    traj = np.stack(
        [
            np.arange(n_points, dtype=np.float64) * 2.0,
            np.zeros(n_points),
            np.zeros(n_points),
        ],
        axis=1,
    )
    reward_pkl = tmp_path / "reward_test.pkl"
    with open(reward_pkl, "wb") as f:
        pickle.dump(traj, f)
    left = np.stack([traj[:, 0], traj[:, 0] * 0 + 5.0, traj[:, 2]], axis=1)
    right = np.stack([traj[:, 0], traj[:, 0] * 0 - 5.0, traj[:, 2]], axis=1)
    left_pkl = tmp_path / "left.pkl"
    right_pkl = tmp_path / "right.pkl"
    with open(left_pkl, "wb") as f:
        pickle.dump(left, f)
    with open(right_pkl, "wb") as f:
        pickle.dump(right, f)

    rf = reward_function_cls(
        reward_data_path=str(reward_pkl),
        reward_config={
            "min_seconds_before_failure": 10.0,
            "speed_reward_weight": 1.0,
            "max_speed_kmh": 300.0,
        },
        time_step_duration=0.05,
        track_path_left=str(left_pkl),
        track_path_right=str(right_pkl),
    )
    return rf, near_finish_progress_threshold


def test_near_finish_grace_prevents_no_progress_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        rf, progress_threshold = _make_reward_fn(Path(tmp))
        end_pos = rf.data[-1].copy()
        for _ in range(len(rf.data) + 5):
            rf.compute_reward(pos=end_pos, speed=80.0, end_of_track=False)

        assert rf.furthest_race_progress >= progress_threshold

        terminated = False
        for _ in range(250):
            _, terminated, _, _ = rf.compute_reward(pos=end_pos, speed=80.0, end_of_track=False)
            if terminated:
                break

        assert not terminated, (
            f"near-finish grace should prevent no_progress_timeout "
            f"(progress={rf.furthest_race_progress:.3f}, reason={rf._term_reason})"
        )


def test_near_finish_grace_does_not_apply_mid_track():
    with tempfile.TemporaryDirectory() as tmp:
        rf, progress_threshold = _make_reward_fn(Path(tmp))
        start_pos = rf.data[0].copy()
        max_steps = rf._max_no_progress_steps + 5
        terminated = False
        for _ in range(max_steps):
            _, terminated, _, _ = rf.compute_reward(pos=start_pos, speed=80.0, end_of_track=False)
            if terminated:
                break

        assert terminated
        assert rf._term_reason == "no_progress_timeout"
        assert rf.furthest_race_progress < progress_threshold
