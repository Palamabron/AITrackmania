"""Tests for finish-line stopping fixes (boundary extrapolation, near-finish grace)."""

from __future__ import annotations

import importlib.util
import pickle
import tempfile
from pathlib import Path

import numpy as np
from tmrl.tools.track.geometry_utils import pad_polyline_xz_straight

_REWARD_PATH = (
    Path(__file__).resolve().parents[1] / "tmrl" / "custom" / "tm" / "utils" / "compute_reward.py"
)


def _load_module(path: Path, name: str):
    """Dynamically import a Python source file as a module.

    Args:
        path: Filesystem path to the ``.py`` file.
        name: Module name to register in ``sys.modules``.

    Returns:
        The loaded module object.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _straight_track(n: int = 50, spacing: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a straight track with n waypoints at the given spacing, ±5 units wide.

    Args:
        n: Number of waypoints.
        spacing: Distance between consecutive waypoints in metres.

    Returns:
        ``(left, right)`` boundary arrays of shape ``(2, n)``.
    """
    xs = np.arange(n, dtype=np.float64) * spacing
    left = np.stack([xs, xs * 0.0 + 5.0], axis=0)
    right = np.stack([xs, xs * 0.0 - 5.0], axis=0)
    return left, right


def _repeat_padding_old(boundary: np.ndarray, n_pad: int) -> np.ndarray:
    """Pad a boundary by repeating its last point — the degenerate behaviour the new code replaces.

    Args:
        boundary: Shape ``(2, n)`` boundary array.
        n_pad: Number of additional repeated points to append.

    Returns:
        Shape ``(2, n + n_pad)`` array whose tail is collapsed to a single coordinate.
    """
    extra = np.full((n_pad, 2), boundary.T[-1])
    return np.concatenate([boundary.T, extra], axis=0).T


BOUNDARY_LOOK_AHEAD = 15


def test_pad_polyline_xz_straight_points_are_distinct():
    """Extrapolated look-ahead points are distinct — not collapsed to a single coordinate."""
    left, _ = _straight_track(20)
    extended = pad_polyline_xz_straight(left, BOUNDARY_LOOK_AHEAD)
    tail = extended[:, -BOUNDARY_LOOK_AHEAD:]
    diffs = np.linalg.norm(np.diff(tail, axis=1), axis=0)
    assert np.all(diffs > 0.5), f"expected forward steps, got diffs={diffs}"


def test_pad_polyline_fixes_degenerate_look_ahead_vs_old_repeat():
    """The new padding extrapolates the tail ahead; the old approach repeated the last point."""
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


def _make_reward_fn(tmp_path: Path, n_points: int = 100, extra_config: dict | None = None):
    """Instantiate a RewardFunction from a temporary straight track and optional config overrides.

    Args:
        tmp_path: Temporary directory for reward and boundary pickle files.
        n_points: Number of trajectory waypoints (spaced 2 m apart).
        extra_config: Optional dict merged into the default reward_config before construction.

    Returns:
        ``(rf, near_finish_progress_threshold)`` where ``rf`` is a freshly constructed
        ``RewardFunction`` and the threshold is read from the module-level constant.
    """
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

    reward_config = {
        "min_seconds_before_failure": 10.0,
        "speed_reward_weight": 1.0,
        "max_speed_kmh": 300.0,
    }
    if extra_config:
        reward_config.update(extra_config)

    rf = reward_function_cls(
        reward_data_path=str(reward_pkl),
        reward_config=reward_config,
        time_step_duration=0.05,
        track_path_left=str(left_pkl),
        track_path_right=str(right_pkl),
    )
    return rf, near_finish_progress_threshold


def test_near_finish_grace_prevents_no_progress_timeout():
    """A car stalled near the finish line is not terminated by the no-progress timer."""
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
    """Stalling mid-track (below the near-finish threshold) still triggers no_progress_timeout."""
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


def test_terminal_failure_penalty_applied_once_on_timeout():
    """The terminal failure penalty is applied only on the final timeout step, not earlier."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(Path(tmp), extra_config={"terminal_failure_penalty": 3.0})
        start_pos = rf.data[0].copy()
        rewards: list[float] = []
        terminated = False
        for _ in range(rf._max_no_progress_steps + 5):
            r, terminated, _, _ = rf.compute_reward(pos=start_pos, speed=0.0, end_of_track=False)
            rewards.append(float(r))
            if terminated:
                break

        assert terminated
        assert rf._term_reason == "no_progress_timeout"
        # Only the terminal step carries the one-time -3.0; preceding stall
        # steps are unaffected.
        assert abs(rewards[-1] - (rewards[-2] - 3.0)) < 1e-6


def test_terminal_failure_penalty_not_applied_on_finish():
    """end_of_track=True suppresses the failure penalty even when a timeout would otherwise fire."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(Path(tmp), extra_config={"terminal_failure_penalty": 3.0})
        start_pos = rf.data[0].copy()
        # Stand still until the very step the timeout fires, but finish on it:
        # end_of_track must suppress the failure penalty.
        for _ in range(rf._max_no_progress_steps - 1):
            _, terminated, _, _ = rf.compute_reward(pos=start_pos, speed=0.0, end_of_track=False)
            assert not terminated
        r_final, _, _, _ = rf.compute_reward(pos=start_pos, speed=0.0, end_of_track=True)
        # Finish bonus (+ remaining-distance progress payout) with no -3.0.
        assert r_final > 0.0


def test_slow_progress_terminates_creep_that_evades_binary_timer():
    """Creep advancing one trajectory index every ~40 steps resets the binary
    no-progress timer forever; the rate-based cutoff must catch it."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(
            Path(tmp),
            extra_config={
                "min_progress_rate": 3.0,
                "slow_progress_window_seconds": 2.0,
            },
        )
        pos = rf.data[0].copy()
        terminated = False
        steps = 0
        for _ in range(600):
            pos = pos + np.array([0.05, 0.0, 0.0])  # 1 m/s creep (points are 2 m apart)
            _, terminated, _, _ = rf.compute_reward(pos=pos, speed=4.0, end_of_track=False)
            steps += 1
            if terminated:
                break

        assert terminated, "slow creep must be terminated by the rate-based cutoff"
        assert rf._term_reason == "slow_progress"
        # Window is 2 s = 40 steps; termination should fire shortly after.
        assert steps <= 80


def test_slow_progress_does_not_fire_when_driving_fast():
    """The slow-progress cutoff is not triggered when the car advances at a fast, steady pace."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(
            Path(tmp),
            extra_config={
                "min_progress_rate": 3.0,
                "slow_progress_window_seconds": 2.0,
            },
        )
        pos = rf.data[0].copy()
        for _ in range(150):
            pos = pos + np.array([1.0, 0.0, 0.0])  # 20 m/s
            if pos[0] >= rf.data[-1][0]:
                break
            _, terminated, _, _ = rf.compute_reward(pos=pos, speed=72.0, end_of_track=False)
            assert not terminated, f"unexpected termination: {rf._term_reason}"


def test_slow_progress_disabled_by_default():
    """The slow-progress cutoff does not activate when min_progress_rate is not configured."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(Path(tmp))
        pos = rf.data[0].copy()
        for _ in range(120):
            pos = pos + np.array([0.05, 0.0, 0.0])
            _, terminated, _, _ = rf.compute_reward(pos=pos, speed=4.0, end_of_track=False)
            assert not terminated or rf._term_reason != "slow_progress"


def test_terminal_failure_penalty_default_off():
    """Without terminal_failure_penalty, the timeout step reward equals the step before it."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(Path(tmp))
        start_pos = rf.data[0].copy()
        rewards: list[float] = []
        terminated = False
        for _ in range(rf._max_no_progress_steps + 5):
            r, terminated, _, _ = rf.compute_reward(pos=start_pos, speed=0.0, end_of_track=False)
            rewards.append(float(r))
            if terminated:
                break

        assert terminated
        assert abs(rewards[-1] - rewards[-2]) < 1e-6


def test_boundary_shaping_disabled_by_default():
    """All boundary-shaping parameters default to zero when not explicitly configured."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(Path(tmp))
        assert rf._boundary_penalty_weight == 0.0
        assert rf._boundary_crash_penalty == 0.0
        assert rf._wall_hug_penalty_factor == 0.0

        on_track = rf.data[10].copy()
        for _ in range(12):
            rf.compute_reward(pos=on_track, speed=50.0, end_of_track=False)

        off_track = on_track.copy()
        off_track[2] = 12.0
        _, terminated, _, _ = rf.compute_reward(pos=off_track, speed=50.0, end_of_track=False)
        assert not terminated or rf._term_reason != "boundary_crash"


def test_boundary_crash_penalty_subtracted_on_wall_exit():
    """Crossing the boundary with boundary_crash_penalty set terminates with a negative reward."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(
            Path(tmp),
            extra_config={
                "boundary_crash_penalty": 5.0,
                "boundary_penalty_weight": 0.0,
            },
        )
        on_track = rf.data[10].copy()
        for _ in range(12):
            rf.compute_reward(pos=on_track, speed=50.0, end_of_track=False)

        off_track = on_track.copy()
        off_track[2] = 12.0
        r, terminated, _, _ = rf.compute_reward(pos=off_track, speed=50.0, end_of_track=False)
        assert terminated
        assert rf._term_reason == "boundary_crash"
        assert r <= -4.9


def test_boundary_crash_penalty_not_clipped_by_reward_floor():
    """One-time crash penalty must survive reward_clip_floor (default 5.0)."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(
            Path(tmp),
            extra_config={
                "boundary_crash_penalty": 10.0,
                "boundary_penalty_weight": 0.0,
                "reward_clip_floor": 5.0,
            },
        )
        on_track = rf.data[10].copy()
        for _ in range(12):
            rf.compute_reward(pos=on_track, speed=50.0, end_of_track=False)

        off_track = on_track.copy()
        off_track[2] = 12.0
        r, terminated, _, _ = rf.compute_reward(pos=off_track, speed=50.0, end_of_track=False)
        assert terminated
        assert rf._term_reason == "boundary_crash"
        assert r <= -9.9


def test_boundary_soft_penalty_applied_inside_wall():
    """boundary_penalty_weight penalizes wall proximity continuously without a crash event."""
    with tempfile.TemporaryDirectory() as tmp:
        rf, _ = _make_reward_fn(
            Path(tmp),
            extra_config={
                "boundary_penalty_weight": 2.0,
                "boundary_penalty_start": 0.5,
                "boundary_crash_penalty": 0.0,
            },
        )
        on_track = rf.data[10].copy()
        for _ in range(12):
            _, terminated, _, _ = rf.compute_reward(pos=on_track, speed=50.0, end_of_track=False)
            assert not terminated

        near_wall = on_track.copy()
        near_wall[2] = 0.45
        r_near, terminated, _, _ = rf.compute_reward(pos=near_wall, speed=50.0, end_of_track=False)
        assert not terminated
        assert r_near < 0.0
