"""Demonstration validation, replay conversion, and resampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np

from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_DURATION_S,
    BRAKE_TAP_SENTINEL,
    BRAKE_TAP_TABLE_N_STEER,
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstration_data import Demonstration
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


@dataclass(frozen=True, slots=True)
class DemonstrationResamplingConfig:
    action_lead_ms: float = 0.0
    aggregate_controls: bool = False


@dataclass(frozen=True, slots=True)
class DemonstrationResamplingRequest:
    demonstration: Demonstration
    decision_interval_ms: float | None
    config: DemonstrationResamplingConfig = DemonstrationResamplingConfig()


@dataclass(frozen=True, slots=True)
class _ResampleContext:
    settings: DemonstrationResamplingConfig
    decision_interval_ms: float | None


@dataclass(frozen=True, slots=True)
class _ControlWindow:
    controls: np.ndarray
    race_times_ms: np.ndarray
    start_ms: float
    stop_ms: float


def demonstration_timing_summary(demonstration: Demonstration) -> dict[str, float]:
    """Summarize the physical telemetry cadence stored in a demonstration."""

    intervals = np.diff(demonstration.frames[:, 3]).astype(np.float64)
    switches = np.any(demonstration.controls[1:] != demonstration.controls[:-1], axis=1)
    return {
        "transitions": float(len(intervals)),
        "first_frame_ms": float(demonstration.frames[0, 3]),
        "interval_median_ms": float(np.median(intervals)),
        "interval_p95_ms": float(np.quantile(intervals, 0.95)),
        "interval_max_ms": float(np.max(intervals)),
        "control_switches": float(np.count_nonzero(switches)),
        "minimum_control_hold_ms": _minimum_control_hold_ms(demonstration),
    }


def _minimum_control_hold_ms(demonstration: Demonstration) -> float:
    controls = demonstration.controls
    if len(controls) < 2:
        return float(demonstration.frames[-1, 3] - demonstration.frames[0, 3])
    boundaries = np.flatnonzero(np.any(controls[1:] != controls[:-1], axis=1)) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    stops = np.concatenate((boundaries, np.asarray([len(controls)])))
    times = demonstration.frames[:, 3].astype(np.float64)
    durations = times[stops] - times[starts]
    return float(np.min(durations))


def validate_recording_quality(
    demonstration: Demonstration,
) -> None:
    """Reject recordings whose telemetry cadence cannot preserve short inputs."""

    summary = demonstration_timing_summary(demonstration)
    _validate_recording_cadence(summary)
    if demonstration.control_alignment != "frame_start":
        raise ValueError(
            "demonstration does not align each control with its transition start frame"
        )
    native = demonstration.action_repeat_frames == 1 and demonstration.decision_interval_ms is None
    if native:
        _validate_native_cadence(summary)


def _validate_recording_cadence(summary: dict[str, float]) -> None:
    if summary["first_frame_ms"] > 50.0:
        raise ValueError("demonstration recording started more than 50 ms after race start")
    if summary["interval_p95_ms"] > 25.0 or summary["interval_max_ms"] > 50.0:
        raise ValueError(
            "demonstration telemetry cadence is too sparse; require p95 <=25 ms and max <=50 ms"
        )


def _validate_native_cadence(summary: dict[str, float]) -> None:
    if (
        summary["first_frame_ms"] > 15.0
        or not 8.0 <= summary["interval_median_ms"] <= 12.0
        or summary["interval_p95_ms"] > 12.0
        or summary["interval_max_ms"] > 20.0
    ):
        raise ValueError(
            "native demonstration missed its 100 Hz timing contract; require first <=15 ms, "
            "median 8-12 ms, p95 <=12 ms, and max <=20 ms"
        )


def _recording_quality_message(demonstration: Demonstration) -> str:
    summary = demonstration_timing_summary(demonstration)
    return (
        "Recording quality: "
        f"transitions={int(summary['transitions'])}, "
        f"first={summary['first_frame_ms']:.0f}ms, "
        f"interval(median={summary['interval_median_ms']:.1f}ms, "
        f"p95={summary['interval_p95_ms']:.1f}ms, max={summary['interval_max_ms']:.1f}ms), "
        f"control_switches={int(summary['control_switches'])}, "
        f"shortest_control={summary['minimum_control_hold_ms']:.1f}ms, "
        f"alignment={demonstration.control_alignment}"
    )


def reject_outliers(
    demonstrations: Sequence[Demonstration], *, max_gap_s: float = 1.0
) -> list[Demonstration]:
    """Keep laps within ``max_gap_s`` of the best finish time, ranked fastest-first."""

    if max_gap_s < 0.0:
        raise ValueError("max_gap_s must be non-negative")
    if not demonstrations:
        return []
    best = min(demonstration.finish_time_s for demonstration in demonstrations)
    cutoff = best + max_gap_s
    return sorted(
        (
            demonstration
            for demonstration in demonstrations
            if demonstration.finish_time_s <= cutoff
        ),
        key=lambda demonstration: demonstration.finish_time_s,
    )


def validate_demonstration(
    demonstration: Demonstration,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
) -> None:
    _validate_demonstration_identity(demonstration, geometry)
    _validate_demonstration_timing(demonstration, config)
    if demonstration.frames.shape[1] != DEFAULT_TELEMETRY_FIELD_COUNT:
        raise ValueError("demonstration telemetry schema does not match the environment")


def _validate_demonstration_identity(
    demonstration: Demonstration, geometry: BoundaryGeometry
) -> None:
    if demonstration.map_uid != geometry.map_uid:
        raise ValueError("demonstration map UID does not match the configured map")
    if demonstration.geometry_sha256 != geometry.sha256:
        raise ValueError("demonstration geometry hash does not match the configured geometry")


def _validate_demonstration_timing(
    demonstration: Demonstration, config: TrackmaniaEnvironmentConfig
) -> None:
    if (
        config.decision_interval_ms is None
        and demonstration.action_repeat_frames != config.action_repeat_frames
    ):
        raise ValueError("demonstration action repeat does not match the environment")
    if (
        config.decision_interval_ms is not None
        and demonstration.decision_interval_ms is not None
        and not np.isclose(
            demonstration.decision_interval_ms,
            config.decision_interval_ms,
            rtol=0.0,
            atol=0.05,
        )
    ):
        raise ValueError("demonstration decision interval does not match the environment")


def resample_demonstration(
    request: DemonstrationResamplingRequest,
) -> tuple[np.ndarray, np.ndarray]:
    """Select transitions on the physical decision-time grid used online."""

    demonstration = request.demonstration
    settings = request.config
    if not np.isfinite(settings.action_lead_ms) or settings.action_lead_ms < 0.0:
        raise ValueError("demonstration action lead must be finite and non-negative")
    selected = _resample_indices(demonstration, request.decision_interval_ms)
    context = _ResampleContext(settings, request.decision_interval_ms)
    actions = _resampled_actions(demonstration, selected, context)
    return demonstration.frames[selected], actions


def _resampled_actions(
    demonstration: Demonstration, selected: np.ndarray, context: _ResampleContext
) -> np.ndarray:
    race_times = demonstration.frames[:, 3]
    settings = context.settings
    if settings.aggregate_controls and context.decision_interval_ms is not None:
        return _aggregate_demonstration_controls(
            demonstration,
            race_times[selected] + settings.action_lead_ms,
        )
    action_times = race_times[selected[:-1]] + settings.action_lead_ms
    action_indices = np.searchsorted(race_times, action_times, side="left")
    action_indices = np.clip(action_indices, 0, len(demonstration.actions) - 1)
    return cast(np.ndarray, demonstration.actions[action_indices])


def _resample_indices(
    demonstration: Demonstration, decision_interval_ms: float | None
) -> np.ndarray:
    if decision_interval_ms is None:
        return np.arange(len(demonstration.frames), dtype=np.int64)
    race_times = demonstration.frames[:, 3]
    indices = [0]
    while indices[-1] < len(demonstration.actions):
        target = float(race_times[indices[-1]]) + decision_interval_ms
        candidate = int(np.searchsorted(race_times, target, side="left"))
        indices.append(min(max(candidate, indices[-1] + 1), len(demonstration.actions)))
    return np.asarray(indices, dtype=np.int64)


def _aggregate_demonstration_controls(
    demonstration: Demonstration, window_boundaries_ms: np.ndarray
) -> np.ndarray:
    if demonstration.control_alignment != "frame_start":
        raise ValueError("control aggregation requires frame_start demonstration controls")
    race_times = demonstration.frames[:, 3].astype(np.float64)
    actions = [
        _quantize_control_window(
            _integrate_control_window(
                _ControlWindow(demonstration.controls, race_times, float(start), float(stop))
            ),
            float(stop - start),
        )
        for start, stop in pairwise(window_boundaries_ms)
    ]
    return np.asarray(actions, dtype=np.int64)


def _integrate_control_window(window: _ControlWindow) -> np.ndarray:
    duration_ms = window.stop_ms - window.start_ms
    if duration_ms <= 0.0:
        raise ValueError("demonstration control window must have positive duration")
    integral = np.zeros(3, dtype=np.float64)
    cursor = window.start_ms
    while cursor < window.stop_ms:
        index, segment_stop = _control_segment(window, cursor)
        overlap_ms = segment_stop - cursor
        integral[[0, 2]] += window.controls[index, [0, 2]] * overlap_ms
        integral[1] += _brake_overlap_ms(
            float(window.controls[index, 1]),
            (float(window.race_times_ms[index]), cursor, segment_stop),
        )
        cursor = segment_stop
    return (integral / duration_ms).astype(np.float32)


def _control_segment(window: _ControlWindow, cursor: float) -> tuple[int, float]:
    index = int(np.searchsorted(window.race_times_ms, cursor, side="right") - 1)
    index = int(np.clip(index, 0, len(window.controls) - 1))
    next_index = index + 1
    boundary = (
        window.race_times_ms[next_index]
        if next_index < len(window.race_times_ms)
        else window.stop_ms
    )
    segment_stop = min(window.stop_ms, max(cursor, float(boundary)))
    return index, window.stop_ms if segment_stop == cursor else segment_stop


def _brake_overlap_ms(brake: float, timing: tuple[float, float, float]) -> float:
    start_ms, overlap_start, overlap_stop = timing
    if brake != BRAKE_TAP_SENTINEL:
        return float(np.clip(brake, 0.0, 1.0)) * (overlap_stop - overlap_start)
    tap_stop = start_ms + BRAKE_TAP_DURATION_S * 1_000.0
    return max(0.0, min(overlap_stop, tap_stop) - overlap_start)


def _quantize_control_window(control: np.ndarray, duration_ms: float) -> int:
    _, table = build_brake_tap_action_table()
    steer_values = np.linspace(-1.0, 1.0, BRAKE_TAP_TABLE_N_STEER, dtype=np.float32)
    steer = float(steer_values[np.argmin(np.abs(steer_values - control[2]))])
    gas = float(control[0] >= 0.5)
    brake_duties = np.asarray(
        [0.0, BRAKE_TAP_DURATION_S * 1_000.0 / duration_ms, 1.0], dtype=np.float32
    )
    brake_values = (0.0, BRAKE_TAP_SENTINEL, 1.0)
    brake = brake_values[int(np.argmin(np.abs(brake_duties - control[1])))]
    return continuous_control_to_discrete_index(
        np.asarray([gas, brake, steer], dtype=np.float32), table
    )
