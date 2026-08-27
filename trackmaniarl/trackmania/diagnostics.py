"""Progress-binned policy diagnostics for TrackMania runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)

_TIMING_METRICS = (
    "entry_time_s",
    "reference_time_s",
    "time_debt_s",
    "projected_velocity_mps_mean",
    "step_race_ms_mean",
    "step_race_ms_p50",
    "step_race_ms_p95",
    "step_race_ms_max",
    "decision_interval_abs_error_ms_mean",
    "decision_interval_abs_error_ms_p95",
    "decision_interval_abs_error_ms_max",
    "action_switch_rate",
    "steer_switch_rate",
)


@dataclass(frozen=True, slots=True)
class ProgressDiagnosticRecord:
    progress_pct: float
    action: Any
    policy: Any
    info: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ExpertDiagnosticRecord:
    progress_pct: float
    expert_q: float
    greedy_q: float
    expert_rank: int


class ProgressBinDiagnostics:
    def __init__(self, action_count: int, bin_count: int = 10) -> None:
        if action_count < 2 or bin_count < 1:
            raise ValueError("action_count must be at least two and bin_count must be positive")
        self.action_count = action_count
        self.bin_count = bin_count
        self._initialize_action_metrics()
        self._initialize_timing_metrics()
        self._initialize_switch_metrics()

    def _initialize_action_metrics(self) -> None:
        self._actions = [[0] * self.action_count for _ in range(self.bin_count)]
        self._q_margin_totals = [0.0] * self.bin_count
        self._q_margin_minimums = [float("inf")] * self.bin_count
        self._q_margin_samples = [0] * self.bin_count
        self._q_max_totals = [0.0] * self.bin_count
        self._q_max_samples = [0] * self.bin_count

    def _initialize_timing_metrics(self) -> None:
        self._entry_time_s: list[float | None] = [None] * self.bin_count
        self._reference_time_s: list[float | None] = [None] * self.bin_count
        self._time_debt_s: list[float | None] = [None] * self.bin_count
        self._projected_velocity_totals = [0.0] * self.bin_count
        self._step_race_ms: list[list[float]] = [[] for _ in range(self.bin_count)]
        self._decision_interval_errors: list[list[float]] = [[] for _ in range(self.bin_count)]

    def _initialize_switch_metrics(self) -> None:
        self._action_switches = [0] * self.bin_count
        self._steer_switches = [0] * self.bin_count
        self._previous_action: int | None = None
        self._previous_steer: float | None = None

    def record(self, record: ProgressDiagnosticRecord) -> None:
        index = self._index(record.progress_pct)
        action_index = _diagnostic_action_index(record.action, self.action_count)
        if action_index is not None:
            self._actions[index][action_index] += 1
            if self._previous_action is not None and action_index != self._previous_action:
                self._action_switches[index] += 1
            self._previous_action = action_index
        self._record_q(index, record.policy)
        if record.info is not None:
            self._record_telemetry(index, record.info)

    def _record_q(self, index: int, policy: Any) -> None:
        margin = getattr(policy, "last_q_margin", None)
        if margin is not None:
            value = float(margin)
            self._q_margin_totals[index] += value
            self._q_margin_minimums[index] = min(self._q_margin_minimums[index], value)
            self._q_margin_samples[index] += 1
        maximum = getattr(policy, "last_q_max", None)
        if maximum is not None:
            self._q_max_totals[index] += float(maximum)
            self._q_max_samples[index] += 1

    def _record_telemetry(self, index: int, info: Mapping[str, Any]) -> None:
        if self._entry_time_s[index] is None:
            self._entry_time_s[index] = float(info.get("race_time_ms", 0.0)) / 1_000.0
            self._reference_time_s[index] = float(info.get("reference_time_s", 0.0))
            self._time_debt_s[index] = float(info.get("time_debt_s", 0.0))
        self._projected_velocity_totals[index] += float(info.get("projected_velocity_mps", 0.0))
        self._step_race_ms[index].append(float(info.get("step_race_time_ms", 0.0)))
        self._decision_interval_errors[index].append(
            abs(float(info.get("decision_interval_error_ms", 0.0)))
        )
        steer = float(info.get("control_steer", 0.0))
        if self._previous_steer is not None and steer != self._previous_steer:
            self._steer_switches[index] += 1
        self._previous_steer = steer

    def summary(self) -> dict[str, dict[str, float]]:
        return {self._name(index): self._bin_summary(index) for index in range(self.bin_count)}

    def flat_summary(self) -> dict[str, float]:
        return {
            f"progress_bin/{name}/{metric}": value
            for name, metrics in self.summary().items()
            for metric, value in metrics.items()
        }

    def _index(self, progress_pct: float) -> int:
        bounded = min(max(progress_pct, 0.0), 100.0)
        return min(self.bin_count - 1, int(bounded * self.bin_count / 100.0))

    def _name(self, index: int) -> str:
        start = index * 100 // self.bin_count
        end = (index + 1) * 100 // self.bin_count
        return f"{start:02d}_{end:03d}"

    def _bin_summary(self, index: int) -> dict[str, float]:
        counts = self._actions[index]
        samples = sum(counts)
        nonzero = [count for count in counts if count]
        summary = self._action_summary(index, samples, nonzero)
        summary.update(self._timing_summary(index))
        return summary

    def _action_summary(self, index: int, samples: int, nonzero: list[int]) -> dict[str, float]:
        margin_samples = self._q_margin_samples[index]
        maximum_samples = self._q_max_samples[index]
        return {
            "action_count": float(samples),
            "action_entropy": self._entropy(samples, nonzero),
            "action_coverage": len(nonzero) / self.action_count,
            "q_margin_mean": self._q_margin_totals[index] / margin_samples
            if margin_samples
            else 0.0,
            "q_margin_min": self._q_margin_minimums[index] if margin_samples else 0.0,
            "q_max_mean": self._q_max_totals[index] / maximum_samples if maximum_samples else 0.0,
        }

    def _entropy(self, samples: int, nonzero: list[int]) -> float:
        if not samples:
            return 0.0
        entropy = -sum((count / samples) * log(count / samples) for count in nonzero)
        return entropy / log(self.action_count)

    def _timing_summary(self, index: int) -> dict[str, float]:
        durations = self._step_race_ms[index]
        if not durations:
            return {}
        count = len(durations)
        errors = self._decision_interval_errors[index]
        return {
            "entry_time_s": float(self._entry_time_s[index] or 0.0),
            "reference_time_s": float(self._reference_time_s[index] or 0.0),
            "time_debt_s": float(self._time_debt_s[index] or 0.0),
            "projected_velocity_mps_mean": self._projected_velocity_totals[index] / count,
            **_step_distribution(durations),
            **_error_distribution(errors),
            "action_switch_rate": self._action_switches[index] / count,
            "steer_switch_rate": self._steer_switches[index] / count,
        }


def _step_distribution(values: list[float]) -> dict[str, float]:
    return {
        "step_race_ms_mean": float(np.mean(values)),
        "step_race_ms_p50": float(np.quantile(values, 0.5)),
        "step_race_ms_p95": float(np.quantile(values, 0.95)),
        "step_race_ms_max": float(np.max(values)),
    }


def _error_distribution(values: list[float]) -> dict[str, float]:
    return {
        "decision_interval_abs_error_ms_mean": float(np.mean(values)),
        "decision_interval_abs_error_ms_p95": float(np.quantile(values, 0.95)),
        "decision_interval_abs_error_ms_max": float(np.max(values)),
    }


def _diagnostic_action_index(action: Any, action_count: int) -> int | None:
    if isinstance(action, (int, np.integer)):
        index = int(action)
        return index if 0 <= index < action_count else None
    if not isinstance(action, np.ndarray) or action.shape != (3,):
        return None
    canonical_count, action_table = build_brake_tap_action_table()
    if action_count != canonical_count:
        return None
    return continuous_control_to_discrete_index(action, action_table)


def aggregate_progress_bins(
    summaries: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, float]:
    grouped = _group_progress_summaries(summaries)
    result: dict[str, float] = {}
    for name, values in grouped.items():
        result.update(_aggregate_progress_bin(name, values))
    return result


def _group_progress_summaries(
    summaries: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, list[Mapping[str, float]]]:
    grouped: dict[str, list[Mapping[str, float]]] = {}
    for summary in summaries:
        for name, metrics in summary.items():
            grouped.setdefault(name, []).append(metrics)
    return grouped


def _aggregate_progress_bin(name: str, values: list[Mapping[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    counts = [item["action_count"] for item in values]
    total = sum(counts)
    weights = [count / total for count in counts] if total else [0.0] * len(counts)
    prefix = f"progress_bin/{name}"
    result[f"{prefix}/action_count"] = total
    result.update(_weighted_progress_metrics(prefix, values, weights))
    minima = [item["q_margin_min"] for item in values if item["action_count"]]
    result[f"{prefix}/q_margin_min"] = min(minima) if minima else 0.0
    result.update(_observed_progress_metrics(prefix, values))
    return result


def _weighted_progress_metrics(
    prefix: str, values: list[Mapping[str, float]], weights: list[float]
) -> dict[str, float]:
    metrics = ("action_entropy", "action_coverage", "q_margin_mean", "q_max_mean")
    return {
        f"{prefix}/{metric}": sum(
            weight * item[metric] for weight, item in zip(weights, values, strict=True)
        )
        for metric in metrics
    }


def _observed_progress_metrics(prefix: str, values: list[Mapping[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in _TIMING_METRICS:
        observed = [item[metric] for item in values if metric in item]
        if observed:
            result[f"{prefix}/{metric}"] = float(np.mean(observed))
    return result


class ExpertActionDiagnostics:
    def __init__(self, bin_count: int = 10) -> None:
        if bin_count < 1:
            raise ValueError("bin_count must be positive")
        self.bin_count = bin_count
        self._counts = [0] * bin_count
        self._expert_q_totals = [0.0] * bin_count
        self._greedy_q_totals = [0.0] * bin_count
        self._rank_totals = [0.0] * bin_count

    def record(self, record: ExpertDiagnosticRecord) -> None:
        index = min(
            self.bin_count - 1,
            max(0, int(record.progress_pct * self.bin_count / 100.0)),
        )
        self._counts[index] += 1
        self._expert_q_totals[index] += record.expert_q
        self._greedy_q_totals[index] += record.greedy_q
        self._rank_totals[index] += record.expert_rank

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            self._name(index): self._summary_at(index, count)
            for index, count in enumerate(self._counts)
        }

    def _name(self, index: int) -> str:
        start = index * 100 // self.bin_count
        end = (index + 1) * 100 // self.bin_count
        return f"{start:02d}_{end:03d}"

    def _summary_at(self, index: int, count: int) -> dict[str, float]:
        expert_total = self._expert_q_totals[index]
        greedy_total = self._greedy_q_totals[index]
        return {
            "count": float(count),
            "expert_q_mean": expert_total / count if count else 0.0,
            "raw_greedy_q_mean": greedy_total / count if count else 0.0,
            "advantage_gap_mean": (greedy_total - expert_total) / count if count else 0.0,
            "expert_action_rank_mean": self._rank_totals[index] / count if count else 0.0,
        }


def aggregate_expert_bins(
    summaries: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    grouped = _group_progress_summaries(summaries)
    return {name: _aggregate_expert_bin(values) for name, values in grouped.items()}


def _aggregate_expert_bin(values: list[Mapping[str, float]]) -> dict[str, float]:
    total = sum(item["count"] for item in values)
    metrics = (
        "expert_q_mean",
        "raw_greedy_q_mean",
        "advantage_gap_mean",
        "expert_action_rank_mean",
    )
    return {
        "count": total,
        **{
            metric: sum(item["count"] * item[metric] for item in values) / total if total else 0.0
            for metric in metrics
        },
    }
