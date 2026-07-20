"""Versioned offline boundary geometry assets for lidar observations."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

GEOMETRY_ASSET_VERSION = "1"


def file_sha256(path: str | Path) -> str:
    """Return the content checksum used to bind a geometry asset to its source map."""

    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_boundary(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("boundary samples must have shape (points, 3)")
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 2:
        raise ValueError("boundary recording contains fewer than two finite positions")
    distance = np.linalg.norm(np.diff(values, axis=0), axis=1)
    keep = np.r_[True, distance > 1e-4]
    values = values[keep]
    if len(values) < 2:
        raise ValueError("boundary recording is degenerate")
    return values


def _validate_geometry_points(points: np.ndarray) -> np.ndarray:
    """Validate a built asset without removing repeated opposite-boundary matches."""

    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("geometry boundaries must have shape (points >= 2, 3)")
    if not np.isfinite(values).all():
        raise ValueError("geometry asset contains non-finite points")
    return values


def _segment_lengths(points: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(np.diff(points, axis=0), axis=1), dtype=np.float64)


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline to ``count`` points uniformly along arc length."""

    segments = _segment_lengths(points)
    arc = np.r_[0.0, np.cumsum(segments)]
    if float(arc[-1]) <= 1e-4:
        raise ValueError("boundary has zero arc length")
    targets = np.linspace(0.0, float(arc[-1]), count, dtype=np.float64)
    return np.stack([np.interp(targets, arc, points[:, axis]) for axis in range(3)], axis=1).astype(
        np.float32
    )


def _resample_matching(
    left: np.ndarray,
    right: np.ndarray,
    center: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-sample paired boundaries uniformly along the centerline arc length."""

    arc = np.r_[0.0, np.cumsum(_segment_lengths(center))]
    if float(arc[-1]) <= 1e-4:
        raise ValueError("centerline has zero arc length")
    targets = np.linspace(0.0, float(arc[-1]), count, dtype=np.float64)

    def take(points: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(targets, arc, points[:, axis]) for axis in range(3)], axis=1
        ).astype(np.float32)

    return take(left), take(right), take(center)


def _smooth_polyline(points: np.ndarray, window: int) -> np.ndarray:
    """Light moving-average smooth along the polyline; endpoints stay fixed."""

    if window == 1:
        return np.asarray(points, dtype=np.float32)
    if window < 1 or window % 2 == 0:
        raise ValueError("smooth_window must be a positive odd integer")
    if len(points) < 3:
        return np.asarray(points, dtype=np.float32)
    radius = window // 2
    values = np.asarray(points, dtype=np.float64)
    out = values.copy()
    for index in range(1, len(values) - 1):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        out[index] = values[start:stop].mean(axis=0)
    out[0] = values[0]
    out[-1] = values[-1]
    return out.astype(np.float32)


def _nearest_index(point: np.ndarray, cloud: np.ndarray) -> int:
    return int(np.argmin(np.sum((cloud - point) ** 2, axis=1)))


def _orient_opposite_boundary(reference: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    """Flip the opposite recording when it was driven against the reference direction."""

    start = _nearest_index(reference[0], opposite)
    end = _nearest_index(reference[-1], opposite)
    if start <= end:
        return opposite
    return np.ascontiguousarray(opposite[::-1])


def _pair_opposite_boundary(
    reference: np.ndarray,
    opposite: np.ndarray,
    *,
    forward_window: int = 256,
    backward_window: int = 64,
) -> np.ndarray:
    """Pair each resampled reference point with a locally nearest opposite point.

    Global nearest-neighbour snaps across parallel map sections and places the
    midpoint centerline between tracks.  Walk the opposite recording with a
    bounded window so matches stay continuous along the driven boundary.
    """

    if forward_window < 1 or backward_window < 0:
        raise ValueError("pairing windows must be non-negative (forward_window >= 1)")
    oriented = _orient_opposite_boundary(reference, opposite)
    indices = np.empty(len(reference), dtype=np.intp)
    indices[0] = _nearest_index(reference[0], oriented)
    for index in range(1, len(reference)):
        previous = int(indices[index - 1])
        start = max(0, previous - backward_window)
        stop = min(len(oriented), previous + forward_window + 1)
        indices[index] = start + _nearest_index(reference[index], oriented[start:stop])
    paired = np.asarray(oriented[indices], dtype=np.float32)
    widths = np.linalg.norm(reference - paired, axis=1)
    if float(np.quantile(widths, 0.1)) <= 0.1:
        raise ValueError("boundary recordings overlap for a substantial portion of the map")
    return paired


def build_geometry_asset(
    output: str | Path,
    left_recording: str | Path,
    right_recording: str | Path,
    *,
    map_uid: str,
    map_path: str | Path,
    spacing_m: float = 2.0,
    smooth_window: int = 5,
) -> Path:
    """Clean, arc-resample and pair two manually recorded map boundaries."""

    if not map_uid:
        raise ValueError("map_uid is required")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError("smooth_window must be a positive odd integer")
    left = _clean_boundary(np.load(left_recording))
    right = _clean_boundary(np.load(right_recording))
    left_length = float(_segment_lengths(left).sum())
    count = max(2, round(left_length / spacing_m) + 1)
    left = _resample(left, count)
    right = _pair_opposite_boundary(left, right)
    center = (left + right) / 2.0
    # Midpoints of a left-arc sample are not uniform on bends (inner/outer path
    # lengths differ). Re-parameterize the triple by centerline arc length so
    # reward/lidar index steps stay ~constant in metres.
    center_length = float(_segment_lengths(center).sum())
    count = max(2, round(center_length / spacing_m) + 1)
    left, right, center = _resample_matching(left, right, center, count)
    if smooth_window > 1:
        left = _smooth_polyline(left, smooth_window)
        right = _smooth_polyline(right, smooth_window)
        center = (left + right) / 2.0
        left, right, center = _resample_matching(left, right, center, count)
    widths = np.linalg.norm(left - right, axis=1)
    if not np.isfinite(center).all() or float(np.median(widths)) <= 0.1:
        raise ValueError("paired boundaries are degenerate or overlap")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_map_sha256 = file_sha256(map_path)
    np.savez_compressed(
        target,
        version=np.asarray(GEOMETRY_ASSET_VERSION),
        map_uid=np.asarray(map_uid),
        map_sha256=np.asarray(source_map_sha256),
        left=left,
        center=center.astype(np.float32),
        right=right,
        spacing_m=np.asarray(spacing_m, dtype=np.float32),
        smooth_window=np.asarray(smooth_window, dtype=np.int32),
        left_sha256=np.asarray(file_sha256(left_recording)),
        right_sha256=np.asarray(file_sha256(right_recording)),
    )
    return target


class BoundaryGeometry:
    """Validated read-only boundary asset used by collection and evaluation."""

    def __init__(self, path: str | Path, *, expected_map_uid: str | None = None) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            required = {"version", "map_uid", "left", "center", "right", "spacing_m"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"geometry asset is missing keys: {sorted(missing)}")
            self.version = str(data["version"].item())
            self.map_uid = str(data["map_uid"].item())
            self.left = _validate_geometry_points(data["left"])
            self.center = _validate_geometry_points(data["center"])
            self.right = _validate_geometry_points(data["right"])
            self.spacing_m = float(data["spacing_m"].item())
            self.map_sha256 = str(data.get("map_sha256", np.asarray("")).item())
        if self.version != GEOMETRY_ASSET_VERSION or self.spacing_m <= 0.0:
            raise ValueError("unsupported or invalid geometry asset")
        if not (len(self.left) == len(self.center) == len(self.right)):
            raise ValueError("geometry asset boundaries must have equal lengths")
        widths = np.linalg.norm(self.left - self.right, axis=1)
        center_length = float(np.linalg.norm(np.diff(self.center, axis=0), axis=1).sum())
        if float(np.median(widths)) <= 0.1 or center_length <= 0.1:
            raise ValueError("geometry asset contains degenerate boundaries or centerline")
        if expected_map_uid is not None and self.map_uid != expected_map_uid:
            raise ValueError("geometry asset map UID does not match evaluation map")
        self.sha256 = file_sha256(self.path)

    def validate_map(self, map_path: str | Path) -> None:
        """Reject missing map files and assets built from a different local map binary."""

        path = Path(map_path)
        if not path.is_file():
            raise ValueError(f"evaluation map file does not exist: {path}")
        if not self.map_sha256:
            raise ValueError("geometry asset is missing the required map checksum")
        if file_sha256(path) != self.map_sha256:
            raise ValueError("geometry asset map checksum does not match evaluation map")
