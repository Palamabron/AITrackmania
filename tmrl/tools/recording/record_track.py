from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import tyro
from loguru import logger
from scipy.interpolate import CubicSpline

from tmrl.custom.interfaces.telemetry_indices import tmrl_grabdata_payload_nb_floats
from tmrl.custom.tm.utils.openplanet_client import TM2020OpenPlanetClient
from tmrl.tools.track.geometry_utils import smooth_points
from tmrl.tools.track.track_telemetry import _is_lap_finished, _position_xyz

MIN_POSITIONS_FOR_TRACK = 50
# Minimum distance (metres) driven before a lap-finish signal is accepted.
# Prevents saving "start line only" traces when the finish flag fires shortly after reset.
MIN_TRACK_LENGTH_M = 100.0
# Arc-length spacing (metres) for resampled track boundaries.
# Dense enough to preserve arcs and 180-degree turns.
TRACK_BOUNDARY_SPACING_M = 0.25
# After smoothing, append this many metres along the end tangent (straight runway / pit exit).
TRACK_STRAIGHT_EXTENSION_M = 100.0
# Cap on spline output points to guard against bogus path lengths.
_MAX_SPLINE_POINTS = 200_000
# Log a progress message every N collected positions.
_LOG_INTERVAL = 1000


def _track_length_m(positions) -> float:
    if len(positions) < 2:
        return 0.0
    pts = np.asarray(positions)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.sum(diffs))


def _filter_origin_points(positions: np.ndarray) -> np.ndarray:
    """Drop ``[0, 0, 0]`` glitch packets (``norm < 1.0``).

    ``retrieve_data()`` already patches most of them, but the very first packets
    before ``_last_good_pos`` is set can still slip through.
    """
    from typing import cast

    pts = cast(np.ndarray, np.asarray(positions, dtype=np.float64))
    if len(pts) < 2:
        return pts
    norms = np.linalg.norm(pts, axis=1)
    filtered = cast(np.ndarray, pts[norms >= 1.0])
    if len(filtered) < 2:
        logger.warning(
            "_filter_origin_points: fewer than 2 valid points after filtering "
            "({} of {} kept); falling back to unfiltered data which may include "
            "[0,0,0] glitch packets.",
            len(filtered),
            len(pts),
        )
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

    recording_announced = False
    while True:
        data = client.retrieve_data(sleep_if_empty=0.01)
        terminated = _is_lap_finished(data)
        if terminated:
            length_m = _track_length_m(positions)
            if len(positions) < MIN_POSITIONS_FOR_TRACK:
                logger.warning(
                    "Ignoring early lap-finished signal with too few positions "
                    f"({len(positions)}). "
                    f"Need at least {MIN_POSITIONS_FOR_TRACK}; keep driving."
                )
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
            smoothed_points = smooth_points(spaced_points, sigma=3)

            logger.info(f"Final number of checkpoints of recorded track: {len(smoothed_points)}")
            if len(smoothed_points) < 2:
                logger.error(
                    "Not enough distinct points. Drive the full track and finish lap again."
                )
                continue
            with open(path_track, "wb") as f:
                pickle.dump(smoothed_points, f)
            logger.info("All done")
            return

        positions.append(_position_xyz(data))
        if not recording_announced:
            recording_announced = True
            logger.info("Recording started")
            logger.info("Recording track boundary trajectory from telemetry.")
        elif len(positions) % _LOG_INTERVAL == 0:
            logger.info(f"Recording in progress: collected {len(positions)} position samples.")


def extend_polyline_straight_forward(
    points: np.ndarray,
    extra_m: float = TRACK_STRAIGHT_EXTENSION_M,
    spacing_m: float = TRACK_BOUNDARY_SPACING_M,
) -> np.ndarray:
    """Append colinear samples along the last segment direction for ``extra_m`` metres.

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


def space_points(points: np.ndarray, spacing_m: float = TRACK_BOUNDARY_SPACING_M) -> np.ndarray:
    """Resample ``points`` by arc length with ``spacing_m`` metre knots.

    Duplicate consecutive points (``|delta| < 1e-6``) are dropped first to avoid
    ``CubicSpline`` monotonicity errors. The output knot count is clamped to
    ``_MAX_SPLINE_POINTS`` in case the input path length is bogus.
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
    desired_num_points = min(max(2, round(total_length / spacing_m)), _MAX_SPLINE_POINTS)
    new_distances = np.linspace(0, total_length, desired_num_points, endpoint=True)
    cs_x = CubicSpline(cumulative_distances, x)
    cs_y = CubicSpline(cumulative_distances, y)
    cs_z = CubicSpline(cumulative_distances, z)
    return np.column_stack((cs_x(new_distances), cs_y(new_distances), cs_z(new_distances)))


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


@dataclass
class RecordTrackCli:
    """Record a track boundary from TM2020 telemetry, or extend existing .pkl files."""

    side: str = "left"
    """Which boundary to record: 'left' or 'right'. Ignored when --extend-pkl is given."""

    extend_pkl: list[str] = field(default_factory=list)
    """Extend mode: one or more .pkl paths to extend in place. Pass exactly two for parallel L/R."""

    extend_meters: float = TRACK_STRAIGHT_EXTENSION_M
    """Length of straight extension in metres (only used with --extend-pkl)."""


if __name__ == "__main__":
    import tmrl.config as cfg

    cli = tyro.cli(RecordTrackCli)

    if cli.extend_pkl:
        _cli_extend_pkls(cli.extend_pkl, extra_m=cli.extend_meters)
    else:
        side = cli.side.lower()
        if side not in ("l", "r", "left", "right"):
            raise ValueError(f"--side must be left/right/l/r, got '{side}'")
        print("Recording starts automatically when telemetry arrives.")
        print("Complete the lap; recording ends automatically on lap finish.")
        path = cfg.TRACK_PATH_LEFT if side in ("l", "left") else cfg.TRACK_PATH_RIGHT
        record_track(path_track=path)
