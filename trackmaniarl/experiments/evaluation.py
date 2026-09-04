"""Versioned TrackMania evaluation-suite models and standard score calculation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import mean, median

STANDARD_METRICS = (
    "eval/finish_rate",
    "eval/finish_time_s",
    "eval/median_finish_time_s",
    "eval/best_finish_time_s",
    "eval/sub40_rate",
    "eval/sub38_rate",
    "eval/sub36_rate",
    "eval/failure_progress_mean_pct",
    "eval/crash_rate",
    "eval/reward",
    "eval/action_latency_ms",
    "eval/controller_apply_ms",
    "eval/telemetry_wait_ms",
    "eval/telemetry_skipped_frames_total",
    "eval/telemetry_skipped_frames_mean",
    "eval/telemetry_skipped_frames_max",
    "eval/telemetry_steps_with_skipped_frames_fraction",
    "eval/throughput_fps",
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One episode's normalized outcome, independent of the tracker backend."""

    finished: bool
    finish_time_s: float | None
    crashed: bool
    reward: float
    action_latency_ms: float
    throughput_fps: float
    progress_pct: float = 0.0
    map_id: str = ""
    map_uid: str = ""
    trial_index: int = 0
    telemetry_error: str | None = None
    controller_error: str | None = None
    progress_bins: Mapping[str, Mapping[str, float]] | None = None
    steps: int = 0
    controller_apply_ms: float = 0.0
    telemetry_wait_ms: float = 0.0
    telemetry_skipped_frames_total: int = 0
    telemetry_skipped_frames_mean: float = 0.0
    telemetry_skipped_frames_max: int = 0
    telemetry_steps_with_skipped_frames_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class _EvaluationSummary:
    values: list[EvaluationResult]
    finished_times: list[float]
    failed_progress: list[float]
    total_steps: int
    skipped_frames: int


def _summarize(values: list[EvaluationResult]) -> _EvaluationSummary:
    finished_times = [item.finish_time_s for item in values if item.finished and item.finish_time_s]
    return _EvaluationSummary(
        values,
        finished_times,
        [item.progress_pct for item in values if not item.finished],
        sum(item.steps for item in values),
        sum(item.telemetry_skipped_frames_total for item in values),
    )


def _threshold_rate(values: list[EvaluationResult], threshold: float) -> float:
    return mean(
        float(item.finished and item.finish_time_s is not None and item.finish_time_s < threshold)
        for item in values
    )


def _finish_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    times = summary.finished_times
    return {
        "eval/finish_rate": mean(float(item.finished) for item in summary.values),
        "eval/finish_time_s": mean(times) if times else 0.0,
        "eval/median_finish_time_s": median(times) if times else 0.0,
        "eval/best_finish_time_s": min(times) if times else 0.0,
        "eval/sub40_rate": _threshold_rate(summary.values, 40.0),
        "eval/sub38_rate": _threshold_rate(summary.values, 38.0),
        "eval/sub36_rate": _threshold_rate(summary.values, 36.0),
    }


def _step_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    values = summary.values
    steps = summary.total_steps
    skipped = summary.skipped_frames
    return {
        "eval/action_latency_ms": _step_weighted_mean(values, "action_latency_ms", steps),
        "eval/controller_apply_ms": _step_weighted_mean(values, "controller_apply_ms", steps),
        "eval/telemetry_wait_ms": _step_weighted_mean(values, "telemetry_wait_ms", steps),
        "eval/telemetry_skipped_frames_total": float(skipped),
        "eval/telemetry_skipped_frames_mean": skipped / steps if steps else 0.0,
        "eval/telemetry_skipped_frames_max": float(
            max(item.telemetry_skipped_frames_max for item in values)
        ),
        "eval/telemetry_steps_with_skipped_frames_fraction": _step_weighted_mean(
            values, "telemetry_steps_with_skipped_frames_fraction", steps
        ),
    }


def _outcome_metrics(summary: _EvaluationSummary) -> dict[str, float]:
    values = summary.values
    progress = mean(summary.failed_progress) if summary.failed_progress else 100.0
    return {
        "eval/failure_progress_mean_pct": progress,
        "eval/crash_rate": mean(float(item.crashed) for item in values),
        "eval/reward": mean(item.reward for item in values),
    }


def aggregate_results(results: Iterable[EvaluationResult]) -> Mapping[str, float]:
    values = list(results)
    if not values:
        raise ValueError("An evaluation suite must contain at least one episode result")
    summary = _summarize(values)
    metrics = _finish_metrics(summary)
    metrics.update(_outcome_metrics(summary))
    metrics.update(_step_metrics(summary))
    metrics["eval/throughput_fps"] = mean(item.throughput_fps for item in values)
    return metrics


def _step_weighted_mean(values: list[EvaluationResult], field: str, total_steps: int) -> float:
    if not total_steps:
        return 0.0
    return sum(float(getattr(item, field)) * item.steps for item in values) / total_steps
