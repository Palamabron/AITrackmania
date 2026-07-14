"""Versioned TrackMania evaluation-suite models and standard score calculation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import mean

STANDARD_METRICS = (
    "eval/finish_rate",
    "eval/finish_time_s",
    "eval/crash_rate",
    "eval/reward",
    "eval/action_latency_ms",
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
    map_id: str = ""
    map_uid: str = ""
    trial_index: int = 0
    telemetry_error: str | None = None
    controller_error: str | None = None


def aggregate_results(results: Iterable[EvaluationResult]) -> Mapping[str, float]:
    """Calculate the metrics required by every versioned evaluation suite."""

    values = list(results)
    if not values:
        raise ValueError("An evaluation suite must contain at least one episode result")
    finished_times = [item.finish_time_s for item in values if item.finished and item.finish_time_s]
    return {
        "eval/finish_rate": mean(float(item.finished) for item in values),
        "eval/finish_time_s": mean(finished_times) if finished_times else 0.0,
        "eval/crash_rate": mean(float(item.crashed) for item in values),
        "eval/reward": mean(item.reward for item in values),
        "eval/action_latency_ms": mean(item.action_latency_ms for item in values),
        "eval/throughput_fps": mean(item.throughput_fps for item in values),
    }
