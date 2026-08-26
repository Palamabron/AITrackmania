"""Build a faster state-compatible trajectory from recorded laps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    load_demonstration,
    resolve_demonstration_paths,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry


@dataclass(frozen=True, slots=True)
class TrajectoryStitchingConfig:
    segment_length_m: float = 25.0
    max_position_gap_m: float = 0.75
    max_velocity_gap_mps: float = 3.0
    max_heading_gap_degrees: float = 3.0
    switch_penalty_s: float = 0.02
    minimum_gain_s: float = 0.10
    require_matching_control: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.segment_length_m,
            self.max_position_gap_m,
            self.max_velocity_gap_mps,
            self.max_heading_gap_degrees,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("trajectory stitching distances must be finite and positive")
        penalties = (self.switch_penalty_s, self.minimum_gain_s)
        if any(not np.isfinite(value) or value < 0.0 for value in penalties):
            raise ValueError("trajectory stitching penalties must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TrajectoryJoin:
    progress_m: float
    progress_fraction: float
    first_source: Path
    second_source: Path
    first_frame_index: int
    second_frame_index: int
    position_gap_m: float
    velocity_gap_mps: float
    heading_gap_degrees: float


@dataclass(frozen=True, slots=True)
class StitchedTrajectory:
    demonstration: Demonstration
    source_paths: tuple[Path, ...]
    joins: tuple[TrajectoryJoin, ...]
    fastest_source_time_s: float

    @property
    def estimated_gain_s(self) -> float:
        return self.fastest_source_time_s - self.demonstration.finish_time_s


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: Path
    demonstration: Demonstration
    progress_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class _JoinCandidate:
    first: _Candidate
    second: _Candidate
    first_index: int
    second_index: int
    finish_time_s: float
    selection_cost_s: float
    join: TrajectoryJoin


@dataclass(frozen=True, slots=True)
class _Boundary:
    progress_index: int
    progress_m: float
    total_distance_m: float


@dataclass(frozen=True, slots=True)
class _JoinState:
    first: _Candidate
    second: _Candidate
    first_index: int
    second_index: int
    gaps: tuple[float, float, float]


type _TimingContract = tuple[int, float | None, str, int]


def build_fastest_compatible_trajectory(
    paths: Sequence[str | Path],
    geometry: BoundaryGeometry,
    config: TrajectoryStitchingConfig | None = None,
) -> StitchedTrajectory:
    """Return the fastest whole lap or one-switch compatible splice."""

    options = config or TrajectoryStitchingConfig()
    candidates = _load_candidates(paths, geometry)
    groups = _group_by_timing_contract(candidates)
    results = [_stitch_group(group, geometry, options) for group in groups.values()]
    best = min(results, key=lambda result: result.demonstration.finish_time_s)
    fastest_source = min(candidate.demonstration.finish_time_s for candidate in candidates)
    return StitchedTrajectory(
        demonstration=best.demonstration,
        source_paths=best.source_paths,
        joins=best.joins,
        fastest_source_time_s=fastest_source,
    )


def _load_candidates(paths: Sequence[str | Path], geometry: BoundaryGeometry) -> list[_Candidate]:
    resolved = resolve_demonstration_paths(paths)
    if not resolved:
        raise ValueError("trajectory stitching requires at least one demonstration")
    candidates = []
    for path in resolved:
        demonstration = load_demonstration(path)
        _validate_map_contract(demonstration, geometry)
        progress = _project_progress(demonstration.frames[:, 4:7], geometry.reward_center)
        candidates.append(_Candidate(path, demonstration, progress))
    return candidates


def _validate_map_contract(demonstration: Demonstration, geometry: BoundaryGeometry) -> None:
    if demonstration.map_uid != geometry.map_uid:
        raise ValueError("stitch source map UID does not match the geometry")
    if demonstration.geometry_sha256 != geometry.sha256:
        raise ValueError("stitch source geometry hash does not match the geometry")


def _project_progress(positions: np.ndarray, centerline: np.ndarray) -> np.ndarray:
    progress = np.empty(len(positions), dtype=np.int64)
    current = 0
    for frame_index, position in enumerate(positions):
        start = max(0, current - 10)
        stop = min(len(centerline), current + 501)
        distances = np.sum(np.square(centerline[start:stop] - position), axis=1)
        current = max(current, start + int(np.argmin(distances)))
        progress[frame_index] = current
    return progress


def _group_by_timing_contract(
    candidates: Sequence[_Candidate],
) -> dict[_TimingContract, list[_Candidate]]:
    groups: dict[_TimingContract, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        demonstration = candidate.demonstration
        contract = (
            demonstration.action_repeat_frames,
            demonstration.decision_interval_ms,
            demonstration.control_alignment,
            demonstration.frames.shape[1],
        )
        groups[contract].append(candidate)
    return dict(groups)


def _stitch_group(
    candidates: Sequence[_Candidate],
    geometry: BoundaryGeometry,
    config: TrajectoryStitchingConfig,
) -> StitchedTrajectory:
    baseline = min(candidates, key=lambda item: item.demonstration.finish_time_s)
    boundaries, distances = _progress_boundaries(geometry.reward_center, config.segment_length_m)
    best = _best_join(candidates, (boundaries, distances), config)
    if (
        best is None
        or best.selection_cost_s > baseline.demonstration.finish_time_s - config.minimum_gain_s
    ):
        return _single_source_result(baseline)
    demonstration = _splice(best)
    return StitchedTrajectory(
        demonstration=demonstration,
        source_paths=(best.first.path, best.second.path),
        joins=(best.join,),
        fastest_source_time_s=baseline.demonstration.finish_time_s,
    )


def _progress_boundaries(
    centerline: np.ndarray, segment_length_m: float
) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1), dtype=np.float64),
    ]
    targets = np.r_[np.arange(0.0, cumulative[-1], segment_length_m), cumulative[-1]]
    indices = np.unique(np.clip(np.searchsorted(cumulative, targets), 0, len(centerline) - 1))
    return indices, cumulative[indices]


def _best_join(
    candidates: Sequence[_Candidate],
    progress: tuple[np.ndarray, np.ndarray],
    config: TrajectoryStitchingConfig,
) -> _JoinCandidate | None:
    boundaries, distances = progress
    best: _JoinCandidate | None = None
    total_distance = float(distances[-1])
    for boundary_index in range(1, len(boundaries) - 1):
        boundary = _Boundary(
            int(boundaries[boundary_index]),
            float(distances[boundary_index]),
            total_distance,
        )
        for first in candidates:
            for second in candidates:
                candidate = _join_at_boundary((first, second), boundary, config)
                if candidate is not None and (
                    best is None or candidate.selection_cost_s < best.selection_cost_s
                ):
                    best = candidate
    return best


def _join_at_boundary(
    pair: tuple[_Candidate, _Candidate],
    boundary: _Boundary,
    config: TrajectoryStitchingConfig,
) -> _JoinCandidate | None:
    first, second = pair
    if first.path == second.path:
        return None
    state = _join_state(first, second, boundary.progress_index)
    if not _compatible(state, config):
        return None
    finish_time = _joined_finish_time(state)
    join = TrajectoryJoin(
        progress_m=boundary.progress_m,
        progress_fraction=boundary.progress_m / boundary.total_distance_m,
        first_source=first.path,
        second_source=second.path,
        first_frame_index=state.first_index,
        second_frame_index=state.second_index,
        position_gap_m=state.gaps[0],
        velocity_gap_mps=state.gaps[1],
        heading_gap_degrees=state.gaps[2],
    )
    return _JoinCandidate(
        first,
        second,
        state.first_index,
        state.second_index,
        finish_time,
        finish_time + config.switch_penalty_s,
        join,
    )


def _frame_at_progress(candidate: _Candidate, progress_index: int) -> int:
    frame_index = int(np.searchsorted(candidate.progress_indices, progress_index, side="left"))
    return min(frame_index, len(candidate.demonstration.frames) - 1)


def _join_state(first: _Candidate, second: _Candidate, progress_index: int) -> _JoinState:
    first_index = _frame_at_progress(first, progress_index)
    second_index = _frame_at_progress(second, progress_index)
    first_frame = first.demonstration.frames[first_index]
    second_frame = second.demonstration.frames[second_index]
    position = float(np.linalg.norm(first_frame[4:7] - second_frame[4:7]))
    velocity = float(np.linalg.norm(first_frame[7:10] - second_frame[7:10]))
    heading = _heading_gap_degrees(first_frame[10:13], second_frame[10:13])
    return _JoinState(first, second, first_index, second_index, (position, velocity, heading))


def _heading_gap_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_horizontal = first[[0, 2]]
    second_horizontal = second[[0, 2]]
    first_horizontal /= max(float(np.linalg.norm(first_horizontal)), 1e-8)
    second_horizontal /= max(float(np.linalg.norm(second_horizontal)), 1e-8)
    cosine = float(np.clip(first_horizontal @ second_horizontal, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _compatible(
    state: _JoinState,
    config: TrajectoryStitchingConfig,
) -> bool:
    within_limits = (
        state.gaps[0] <= config.max_position_gap_m
        and state.gaps[1] <= config.max_velocity_gap_mps
        and state.gaps[2] <= config.max_heading_gap_degrees
    )
    if not within_limits or not config.require_matching_control:
        return within_limits
    first_control = state.first.demonstration.controls[
        min(state.first_index, len(state.first.demonstration.controls) - 1)
    ]
    second_control = state.second.demonstration.controls[
        min(state.second_index, len(state.second.demonstration.controls) - 1)
    ]
    return bool(np.array_equal(first_control, second_control))


def _joined_finish_time(state: _JoinState) -> float:
    first_time = float(state.first.demonstration.frames[state.first_index, 3]) / 1_000.0
    second_time = float(state.second.demonstration.frames[state.second_index, 3]) / 1_000.0
    remainder = state.second.demonstration.finish_time_s - second_time
    return first_time + remainder


def _splice(join: _JoinCandidate) -> Demonstration:
    first = join.first.demonstration
    second = join.second.demonstration
    tail = second.frames[join.second_index + 1 :].copy()
    time_offset = first.frames[join.first_index, 3] - second.frames[join.second_index, 3]
    tail[:, 3] += time_offset
    frames = np.concatenate((first.frames[: join.first_index + 1], tail))
    actions = np.concatenate(
        (first.actions[: join.first_index], second.actions[join.second_index :])
    )
    controls = np.concatenate(
        (first.controls[: join.first_index], second.controls[join.second_index :])
    )
    return Demonstration(
        map_uid=first.map_uid,
        geometry_sha256=first.geometry_sha256,
        action_repeat_frames=first.action_repeat_frames,
        frames=frames,
        actions=actions,
        controls=controls,
        finish_time_s=join.finish_time_s,
        decision_interval_ms=first.decision_interval_ms,
        control_alignment=first.control_alignment,
    )


def _single_source_result(candidate: _Candidate) -> StitchedTrajectory:
    return StitchedTrajectory(
        demonstration=candidate.demonstration,
        source_paths=(candidate.path,),
        joins=(),
        fastest_source_time_s=candidate.demonstration.finish_time_s,
    )
