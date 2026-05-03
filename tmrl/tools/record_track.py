import argparse
import os
import pickle
import sys
from typing import cast

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from tmrl.custom.interfaces.telemetry_indices import (
    TMRL_GRABDATA_FLOAT_COUNT,
    TmrlDataPlugin,
    tmrl_grabdata_payload_nb_floats,
)
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

MIN_POSITIONS_FOR_TRACK = 50
# Minimum distance (metres) driven before a lap-finish signal is accepted.
# Prevents saving "start line only" traces when the finish flag fires shortly after reset.
MIN_TRACK_LENGTH_M = 100.0


def _track_length_m(positions):
    if len(positions) < 2:
        return 0.0
    pts = np.asarray(positions)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.sum(diffs))


def _position_xyz(data: tuple[float, ...]) -> list[float]:
    """Return [x, y, z] for legacy (19f), TQC (20f) and TMRL_GrabData (33f) payloads."""
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        px = int(TmrlDataPlugin.POS_X)
        return [data[px], data[px + 1], data[px + 2]]
    if len(data) >= 20:
        return [data[3], data[4], data[5]]
    return [data[2], data[3], data[4]]


def _is_lap_finished(data: tuple[float, ...]) -> bool:
    """Return finish flag for legacy (19f), TQC (20f) and TMRL_GrabData (33f) payloads."""
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        finish_idx = int(TmrlDataPlugin.FINISH_UI_ACTIVE)
    else:
        finish_idx = 9 if len(data) >= 20 else 8
    return bool(data[finish_idx])


def _filter_origin_points(positions: np.ndarray) -> np.ndarray:
    """Drop ``[0, 0, 0]`` glitch packets (``norm < 1.0``).

    ``retrieve_data()`` already patches most of them, but the very first packets
    before ``_last_good_pos`` is set can still slip through. A jump-distance
    filter is deliberately avoided here because a stale first sample was
    causing every subsequent real position to be rejected as an "outlier".
    """
    pts = cast(np.ndarray, np.asarray(positions, dtype=np.float64))
    if len(pts) < 2:
        return pts
    norms = np.linalg.norm(pts, axis=1)
    mask = norms >= 1.0
    filtered = cast(np.ndarray, pts[mask])
    if len(filtered) < 2:
        return pts
    return filtered


def record_track(path_track: str | None = None) -> None:
    import tmrl.config as cfg

    if path_track is None:
        path_track = cfg.TRACK_PATH_LEFT
    positions: list[list[float]] = []
    client = TM2020OpenPlanetClient(
        port=9000, nb_floats=tmrl_grabdata_payload_nb_floats(cfg.REWARD_CONFIG)
    )
    path = path_track

    recording_announced = False
    while True:
        data = client.retrieve_data(sleep_if_empty=0.01)
        terminated = _is_lap_finished(data)
        if terminated:
            length_m = _track_length_m(positions)
            if len(positions) < MIN_POSITIONS_FOR_TRACK:
                msg = (
                    "Ignoring early lap-finished signal with too few positions "
                    f"({len(positions)}). "
                    f"Need at least {MIN_POSITIONS_FOR_TRACK}; keep driving."
                )
                logger.warning(msg)
                continue
            if length_m < MIN_TRACK_LENGTH_M:
                logger.warning(
                    f"Ignoring lap-finished signal because track is too short ({length_m:.0f} m). "
                    f"Need at least {MIN_TRACK_LENGTH_M:.0f} m; keep driving."
                )
                continue
            logger.info("Computing track checkpoints from captured positions...")
            logger.info(f"Initial number of captured positions: {len(positions)}")
            positions_xyz = np.asarray(positions, dtype=np.float64)

            positions_filtered = _filter_origin_points(positions_xyz)

            length_after = _track_length_m(positions_filtered)
            logger.info(
                f"After filtering: {len(positions_filtered)} positions, "
                f"path length {length_after:.0f} m"
            )

            spaced_points = space_points(positions_filtered)
            smoothed_points = smooth_points(spaced_points)

            logger.info(f"Final number of checkpoints of recorded track: {len(smoothed_points)}")
            if len(smoothed_points) < 2:
                logger.error(
                    "Not enough distinct points. Drive the full track and finish lap again."
                )
                continue
            with open(path, "wb") as f:
                pickle.dump(smoothed_points, f)
            logger.info("All done")
            return

        positions.append(_position_xyz(data))
        if not recording_announced:
            recording_announced = True
            logger.info("Recording started")
            logger.info("Recording track boundary trajectory from telemetry.")
        elif len(positions) % 1000 == 0:
            logger.info(f"Recording in progress: collected {len(positions)} position samples.")


# Arc-length spacing (metres) for resampled track boundaries.
# Dense enough to preserve arcs and 180-degree turns. The previous implementation
# keyed the sample count off ``len(reward_file)``, which under-sampled curves
# into sharp corners.
TRACK_BOUNDARY_SPACING_M = 0.25
# After smoothing, append this many metres along the end tangent (straight runway / pit exit).
TRACK_STRAIGHT_EXTENSION_M = 100.0


def extend_polyline_straight_forward(
    points: np.ndarray,
    extra_m: float = TRACK_STRAIGHT_EXTENSION_M,
    spacing_m: float = TRACK_BOUNDARY_SPACING_M,
) -> np.ndarray:
    """Append colinear samples along the last segment direction for ``extra_m`` metres.

    Uses the unit vector from the second-to-last to the last point. New samples are
    spaced by ``spacing_m`` (same as ``space_points``) so downstream code stays consistent.

    For left+right boundaries, prefer :func:`extend_two_boundaries_parallel` so both
    extensions share one forward direction.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2 or extra_m <= 0:
        return pts
    tang = pts[-1] - pts[-2]
    norm = float(np.linalg.norm(tang))
    if norm < 1e-9:
        return pts
    u = tang / norm
    n_new = max(1, round(extra_m / spacing_m))
    step = extra_m / n_new
    extra_rows = pts[-1] + u * (step * np.arange(1, n_new + 1, dtype=np.float64)[:, np.newaxis])
    return np.vstack([pts, extra_rows])


def _shared_forward_unit(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Unit vector along the track from the last centerline segment (mid of L/R)."""
    left_pts = np.asarray(left, dtype=np.float64)
    right_pts = np.asarray(right, dtype=np.float64)
    if len(left_pts) < 2 or len(right_pts) < 2:
        raise ValueError("left and right boundaries need at least 2 points each")
    c0 = 0.5 * (left_pts[-2] + right_pts[-2])
    c1 = 0.5 * (left_pts[-1] + right_pts[-1])
    v = c1 - c0
    n = float(np.linalg.norm(v))
    if n >= 1e-9:
        return np.asarray(v / n, dtype=np.float64)
    tl = left_pts[-1] - left_pts[-2]
    tr = right_pts[-1] - right_pts[-2]
    v = tl + tr
    n = float(np.linalg.norm(v))
    if n >= 1e-9:
        return np.asarray(v / n, dtype=np.float64)
    tl_n = float(np.linalg.norm(tl))
    if tl_n >= 1e-9:
        return np.asarray(tl / tl_n, dtype=np.float64)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def extend_two_boundaries_parallel(
    left: np.ndarray,
    right: np.ndarray,
    extra_m: float = TRACK_STRAIGHT_EXTENSION_M,
    spacing_m: float = TRACK_BOUNDARY_SPACING_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Append the same straight direction to both sides so the extensions are parallel.

    Forward ``u`` comes from the last segment of the centerline
    ``(left + right) / 2``, then both polylines step by the same offsets along ``u``.
    """
    left_pts = np.asarray(left, dtype=np.float64)
    right_pts = np.asarray(right, dtype=np.float64)
    if extra_m <= 0:
        return left_pts, right_pts
    u = _shared_forward_unit(left_pts, right_pts)
    n_new = max(1, round(extra_m / spacing_m))
    step = extra_m / n_new
    t = step * np.arange(1, n_new + 1, dtype=np.float64)[:, np.newaxis]
    left_ext = np.vstack([left_pts, left_pts[-1] + u * t])
    right_ext = np.vstack([right_pts, right_pts[-1] + u * t])
    return left_ext, right_ext


def space_points(points, spacing_m=TRACK_BOUNDARY_SPACING_M):
    """Resample ``points`` by arc length with ``spacing_m`` metre knots.

    Duplicate consecutive points (``|delta| < 1e-6``) are dropped first to avoid
    ``CubicSpline`` monotonicity errors. The output knot count is clamped to
    200,000 in case the input path length is bogus.
    """
    if len(points) < 2:
        return points.copy()

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mask = distances > 1e-6
    if not np.any(mask):
        return points.copy()

    valid_points = [points[0]]
    for i, m in enumerate(mask):
        if m:
            valid_points.append(points[i + 1])
    points = np.array(valid_points)

    if len(points) < 2:
        return points.copy()

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_length = float(cumulative_distances[-1])
    if total_length <= 0:
        return points
    desired_num_points = max(2, round(total_length / spacing_m))
    desired_num_points = min(desired_num_points, 200_000)
    new_distances = np.linspace(0, total_length, desired_num_points, endpoint=True)
    cs_x = CubicSpline(cumulative_distances, x)
    cs_y = CubicSpline(cumulative_distances, y)
    cs_z = CubicSpline(cumulative_distances, z)
    return np.column_stack((cs_x(new_distances), cs_y(new_distances), cs_z(new_distances)))


def interp_points_with_cubic_spline(sub_array, data_density=3):
    """Cubic-spline interpolate ``sub_array`` (N, 3), upsampled by ``data_density``."""
    if len(sub_array) < 2:
        return sub_array.copy()
    original_x, original_y, original_z = sub_array.T
    original_i = np.arange(0, int(data_density * len(original_x)), step=data_density)
    if len(original_i) < 2:
        return sub_array.copy()
    new_i = np.arange(0, int(data_density * len(original_x) - 1))

    print("Original i:", len(original_i))
    print("Original x:", len(original_x))
    print("Original y:", len(original_y))
    print("Original z:", len(original_z))
    print("new_i:", len(new_i))

    cs_x = CubicSpline(original_i, original_x)
    cs_y = CubicSpline(original_i, original_y)
    cs_z = CubicSpline(original_i, original_z)
    return np.array([cs_x(new_i), cs_y(new_i), cs_z(new_i)]).T


def smooth_points(points, sigma=3):
    """Apply a per-axis Gaussian filter (``sigma`` samples) to (N, 3) ``points``."""
    smoothed_x = gaussian_filter1d(points[:, 0], sigma)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma)
    smoothed_z = gaussian_filter1d(points[:, 2], sigma)
    return np.column_stack((smoothed_x, smoothed_y, smoothed_z))


def line(pt1, pt2, dist):
    """Step along the segment ``pt1 -> pt2`` by ``dist`` metres.

    Returns:
        ``(pt, 0.0)`` when a new point was produced, or ``(None, remaining)``
        when the segment was shorter than ``dist`` and ``remaining`` metres
        still need to be walked on the next segment.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return None, dist - norm
    vec_unit = vec / norm
    pt = pt1 + vec_unit * dist
    return pt, 0.0


def _cli_extend_pkls(paths: list[str], extra_m: float) -> None:
    """Load boundary ``.pkl`` files, append straight extension, overwrite.

    With **exactly two** paths, uses one shared forward vector from the last
    centerline segment so left and right extensions are parallel. Otherwise each
    file is extended along its own end tangent.
    """
    if len(paths) == 2:
        p0, p1 = paths[0], paths[1]
        for p in (p0, p1):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
        with open(p0, "rb") as f:
            raw0 = pickle.load(f)
        with open(p1, "rb") as f:
            raw1 = pickle.load(f)
        a0 = np.asarray(raw0, dtype=np.float64)
        a1 = np.asarray(raw1, dtype=np.float64)
        ext0, ext1 = extend_two_boundaries_parallel(a0, a1, extra_m=extra_m)
        for path, pts, raw in ((p0, ext0, a0), (p1, ext1, a1)):
            with open(path, "wb") as f:
                pickle.dump(pts, f)
            logger.info(
                "Extended {} by {:.1f} m (parallel with pair) ({} -> {} points)",
                path,
                extra_m,
                len(raw),
                len(pts),
            )
        return

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            raw = pickle.load(f)
        pts = extend_polyline_straight_forward(np.asarray(raw, dtype=np.float64), extra_m=extra_m)
        with open(path, "wb") as f:
            pickle.dump(pts, f)
        logger.info(
            "Extended {} by {:.1f} m straight ({} -> {} points)",
            path,
            extra_m,
            len(np.asarray(raw)),
            len(pts),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record track boundary from TM2020 telemetry, or extend existing .pkl files.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_ext = sub.add_parser(
        "extend",
        help=(
            f"Append a straight segment (default {TRACK_STRAIGHT_EXTENSION_M:.0f} m) in place. "
            "Pass **two** paths (left+right) to use one shared direction so both extensions "
            "stay parallel."
        ),
    )
    p_ext.add_argument(
        "pkl",
        nargs="+",
        help="One or more track_*_boundary.pkl paths; use exactly two for parallel L/R extension.",
    )
    p_ext.add_argument(
        "--meters",
        type=float,
        default=TRACK_STRAIGHT_EXTENSION_M,
        help=f"Length of straight extension in metres (default: {TRACK_STRAIGHT_EXTENSION_M}).",
    )

    args = parser.parse_args()
    if args.command == "extend":
        _cli_extend_pkls(args.pkl, extra_m=args.meters)
        sys.exit(0)

    import tmrl.config as cfg

    if not os.path.exists(cfg.REWARD_PATH):
        logger.debug(f" reward not found at path:{cfg.REWARD_PATH}")
    which_track = input("Choose which track do you want to record [left/right] [l/r]: ").lower()
    assert which_track in ("l", "r", "right", "left"), "Input must be left, right, l or r"
    print("Recording starts automatically when telemetry arrives.")
    print("Complete the lap; recording ends automatically on lap finish.")
    if which_track in ("l", "left"):
        record_track(path_track=cfg.TRACK_PATH_LEFT)
    elif which_track in ("r", "right"):
        record_track(path_track=cfg.TRACK_PATH_RIGHT)
