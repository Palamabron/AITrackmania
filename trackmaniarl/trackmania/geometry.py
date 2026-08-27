"""Versioned offline boundary geometry assets for lidar observations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest

GEOMETRY_ASSET_VERSION = "1"
type _LineTriple = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class _CurvatureProjection:
    origin: np.ndarray
    corridor: np.ndarray
    denominator: np.ndarray


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


def _validate_reward_line(points: np.ndarray) -> None:
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("geometry reward line contains adjacent duplicate points")
    directions = segments / lengths[:, None]
    if len(directions) > 1 and np.any(
        np.linalg.norm(directions[:-1] + directions[1:], axis=1) <= 1.0e-6
    ):
        raise ValueError("geometry reward line contains a zero-length local tangent")


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


def _resample_matching(lines: _LineTriple, count: int) -> _LineTriple:
    """Re-sample paired boundaries uniformly along the centerline arc length."""

    left, right, center = lines
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


def _minimum_curvature_line(left: np.ndarray, right: np.ndarray, center: np.ndarray) -> np.ndarray:
    edge_margin_fraction = 0.12
    line = np.asarray(center, dtype=np.float64).copy()
    inner_left = left + edge_margin_fraction * (right - left)
    inner_right = right + edge_margin_fraction * (left - right)
    corridor = np.asarray(inner_right - inner_left, dtype=np.float64)
    origin = np.asarray(inner_left, dtype=np.float64)
    denominator = np.square(corridor).sum(axis=1).clip(min=1e-8)
    projection = _CurvatureProjection(origin, corridor, denominator)
    closed = _is_closed_loop(center)
    for _ in range(256):
        line = _curvature_iteration(line, projection)
        if not closed:
            line[[0, -1]] = center[[0, -1]]
    return np.asarray(line, dtype=np.float32)


def _curvature_iteration(line: np.ndarray, projection: _CurvatureProjection) -> np.ndarray:
    neighbors = 0.5 * (np.roll(line, 1, axis=0) + np.roll(line, -1, axis=0))
    candidate = 0.35 * line + 0.65 * neighbors
    fraction = (
        np.sum((candidate - projection.origin) * projection.corridor, axis=1)
        / projection.denominator
    )
    result = projection.origin + np.clip(fraction, 0.0, 1.0)[:, None] * projection.corridor
    return np.asarray(result, dtype=np.float64)


def _is_closed_loop(center: np.ndarray) -> bool:
    """True when the recording is a full lap (start and finish are the same place)."""

    gap = float(np.linalg.norm(center[0] - center[-1]))
    length = float(_segment_lengths(center).sum())
    if length <= 1e-4:
        return False
    return gap / length <= 0.05


def _end_tangent(points: np.ndarray) -> np.ndarray:
    direction = points[-1] - points[-2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-5:
        span = min(5, len(points) - 1)
        direction = points[-1] - points[-1 - span]
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-5:
        raise ValueError("cannot extrapolate a degenerate boundary end")
    return np.asarray(direction / norm, dtype=np.float64)


def _extend_polyline(points: np.ndarray, *, count: int, spacing_m: float) -> np.ndarray:
    if count <= 0:
        return np.asarray(points, dtype=np.float32)
    tangent = _end_tangent(points)
    steps = spacing_m * np.arange(1, count + 1, dtype=np.float64)
    extra = points[-1].astype(np.float64) + steps[:, None] * tangent[None, :]
    return np.concatenate([points, extra.astype(np.float32)], axis=0)


def _extend_open_finish(lines: _LineTriple, lookahead_points: int, spacing_m: float) -> _LineTriple:
    """Extrapolate past the finish so lidar look-ahead keeps seeing new samples."""

    left, right, center = lines
    return (
        _extend_polyline(left, count=lookahead_points, spacing_m=spacing_m),
        _extend_polyline(right, count=lookahead_points, spacing_m=spacing_m),
        _extend_polyline(center, count=lookahead_points, spacing_m=spacing_m),
    )


def _nearest_index(point: np.ndarray, cloud: np.ndarray) -> int:
    return int(np.argmin(np.sum((cloud - point) ** 2, axis=1)))


def _orient_opposite_boundary(reference: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    """Flip the opposite recording when it was driven against the reference direction."""

    start = _nearest_index(reference[0], opposite)
    end = _nearest_index(reference[-1], opposite)
    if start <= end:
        return opposite
    return np.ascontiguousarray(opposite[::-1])


def _pair_opposite_boundary(reference: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    """Pair each resampled reference point with a locally nearest opposite point.

    Global nearest-neighbour snaps across parallel map sections and places the
    midpoint centerline between tracks.  Walk the opposite recording with a
    bounded window so matches stay continuous along the driven boundary.
    """

    oriented = _orient_opposite_boundary(reference, opposite)
    indices = _paired_indices(reference, oriented)
    paired = np.asarray(oriented[indices], dtype=np.float32)
    widths = np.linalg.norm(reference - paired, axis=1)
    if float(np.quantile(widths, 0.1)) <= 0.1:
        raise ValueError("boundary recordings overlap for a substantial portion of the map")
    return paired


def _paired_indices(reference: np.ndarray, oriented: np.ndarray) -> np.ndarray:
    indices = np.empty(len(reference), dtype=np.intp)
    indices[0] = _nearest_index(reference[0], oriented)
    for index in range(1, len(reference)):
        previous = int(indices[index - 1])
        start = max(0, previous - 64)
        stop = min(len(oriented), previous + 257)
        indices[index] = start + _nearest_index(reference[index], oriented[start:stop])
    return indices


def build_geometry_asset(request: GeometryBuildRequest) -> Path:
    """Clean, arc-resample and pair two manually recorded map boundaries."""
    from trackmaniarl.trackmania.geometry_build import build

    return build(request)


class BoundaryGeometry:
    """Validated read-only boundary asset used by collection and evaluation."""

    def __init__(self, path: str | Path, *, expected_map_uid: str | None = None) -> None:
        self.path = Path(path)
        self._load_asset()
        self._validate_asset(expected_map_uid)
        self.sha256 = file_sha256(self.path)
        self._initialize_racing_line()

    def _load_asset(self) -> None:
        with np.load(self.path, allow_pickle=False) as data:
            self._validate_archive_files(data.files)
            self.version = str(data["version"].item())
            self.map_uid = str(data["map_uid"].item())
            self.left = _validate_geometry_points(data["left"])
            self.center = _validate_geometry_points(data["center"])
            self.right = _validate_geometry_points(data["right"])
            self.spacing_m = float(data["spacing_m"].item())
            self.map_sha256 = str(data["map_sha256"].item())
            self.recorded_count = int(data["recorded_count"].item())

    @staticmethod
    def _validate_archive_files(files: list[str]) -> None:
        required = {
            "version",
            "map_uid",
            "map_sha256",
            "left",
            "center",
            "right",
            "spacing_m",
            "recorded_count",
        }
        missing = required - set(files)
        if missing:
            raise ValueError(f"geometry asset is missing keys: {sorted(missing)}")

    def _validate_asset(self, expected_map_uid: str | None) -> None:
        if (
            self.version != GEOMETRY_ASSET_VERSION
            or not np.isfinite(self.spacing_m)
            or self.spacing_m <= 0.0
        ):
            raise ValueError("unsupported or invalid geometry asset")
        if not (len(self.left) == len(self.center) == len(self.right)):
            raise ValueError("geometry asset boundaries must have equal lengths")
        if not 2 <= self.recorded_count <= len(self.center):
            raise ValueError("geometry asset recorded_count is out of range")
        widths = np.linalg.norm(self.left - self.right, axis=1)
        center_length = float(np.linalg.norm(np.diff(self.center, axis=0), axis=1).sum())
        if float(np.median(widths)) <= 0.1 or center_length <= 0.1:
            raise ValueError("geometry asset contains degenerate boundaries or centerline")
        _validate_reward_line(self.reward_center)
        if expected_map_uid is not None and self.map_uid != expected_map_uid:
            raise ValueError("geometry asset map UID does not match evaluation map")

    def _initialize_racing_line(self) -> None:
        recorded = slice(0, self.recorded_count)
        self._racing_line = _minimum_curvature_line(
            self.left[recorded],
            self.right[recorded],
            self.center[recorded],
        )
        _validate_reward_line(self._racing_line)

    @property
    def reward_center(self) -> np.ndarray:
        """Centerline used for progress/finish (excludes virtual lidar extension)."""

        return self.center[: self.recorded_count]

    @property
    def racing_line(self) -> np.ndarray:
        """Minimum-curvature reference constrained to the recorded road corridor."""

        return self._racing_line

    def validate_map(self, map_path: str | Path) -> None:
        """Reject missing map files and assets built from a different local map binary."""

        path = Path(map_path)
        if not path.is_file():
            raise ValueError(f"evaluation map file does not exist: {path}")
        if not self.map_sha256:
            raise ValueError("geometry asset is missing the required map checksum")
        if file_sha256(path) != self.map_sha256:
            raise ValueError("geometry asset map checksum does not match evaluation map")
