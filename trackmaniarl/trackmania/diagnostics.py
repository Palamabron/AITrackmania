"""Progress-binned policy diagnostics for TrackMania runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import log
from typing import Any

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)


class ProgressBinDiagnostics:
    def __init__(self, action_count: int, bin_count: int = 10) -> None:
        if action_count < 2 or bin_count < 1:
            raise ValueError("action_count must be at least two and bin_count must be positive")
        self.action_count = action_count
        self.bin_count = bin_count
        self._actions = [[0] * action_count for _ in range(bin_count)]
        self._q_margin_totals = [0.0] * bin_count
        self._q_margin_minimums = [float("inf")] * bin_count
        self._q_margin_samples = [0] * bin_count
        self._q_max_totals = [0.0] * bin_count
        self._q_max_samples = [0] * bin_count
        self._entry_time_s: list[float | None] = [None] * bin_count
        self._reference_time_s: list[float | None] = [None] * bin_count
        self._time_debt_s: list[float | None] = [None] * bin_count
        self._projected_velocity_totals = [0.0] * bin_count
        self._step_race_ms: list[list[float]] = [[] for _ in range(bin_count)]
        self._decision_interval_errors: list[list[float]] = [[] for _ in range(bin_count)]
        self._action_switches = [0] * bin_count
        self._steer_switches = [0] * bin_count
        self._previous_action: int | None = None
        self._previous_steer: float | None = None

    def record(
        self,
        progress_pct: float,
        action: Any,
        policy: Any,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        index = self._index(progress_pct)
        action_index = _diagnostic_action_index(action, self.action_count)
        if action_index is not None:
            self._actions[index][action_index] += 1
            if self._previous_action is not None and action_index != self._previous_action:
                self._action_switches[index] += 1
            self._previous_action = action_index
        self._record_q(index, policy)
        if info is not None:
            self._record_telemetry(index, info)

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
        entropy = 0.0
        if samples:
            entropy = -sum((count / samples) * log(count / samples) for count in nonzero) / log(
                self.action_count
            )
        margin_samples = self._q_margin_samples[index]
        maximum_samples = self._q_max_samples[index]
        summary = {
            "action_count": float(samples),
            "action_entropy": entropy,
            "action_coverage": len(nonzero) / self.action_count,
            "q_margin_mean": self._q_margin_totals[index] / margin_samples
            if margin_samples
            else 0.0,
            "q_margin_min": self._q_margin_minimums[index] if margin_samples else 0.0,
            "q_max_mean": self._q_max_totals[index] / maximum_samples if maximum_samples else 0.0,
        }
        durations = self._step_race_ms[index]
        if durations:
            count = len(durations)
            errors = self._decision_interval_errors[index]
            summary.update(
                {
                    "entry_time_s": float(self._entry_time_s[index] or 0.0),
                    "reference_time_s": float(self._reference_time_s[index] or 0.0),
                    "time_debt_s": float(self._time_debt_s[index] or 0.0),
                    "projected_velocity_mps_mean": self._projected_velocity_totals[index] / count,
                    "step_race_ms_mean": float(np.mean(durations)),
                    "step_race_ms_p50": float(np.quantile(durations, 0.5)),
                    "step_race_ms_p95": float(np.quantile(durations, 0.95)),
                    "step_race_ms_max": float(np.max(durations)),
                    "decision_interval_abs_error_ms_mean": float(np.mean(errors)),
                    "decision_interval_abs_error_ms_p95": float(np.quantile(errors, 0.95)),
                    "decision_interval_abs_error_ms_max": float(np.max(errors)),
                    "action_switch_rate": self._action_switches[index] / count,
                    "steer_switch_rate": self._steer_switches[index] / count,
                }
            )
        return summary


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
    grouped: dict[str, list[Mapping[str, float]]] = {}
    for summary in summaries:
        for name, metrics in summary.items():
            grouped.setdefault(name, []).append(metrics)
    result: dict[str, float] = {}
    for name, values in grouped.items():
        counts = [item["action_count"] for item in values]
        total = sum(counts)
        weights = [count / total for count in counts] if total else [0.0] * len(counts)
        result[f"progress_bin/{name}/action_count"] = total
        for metric in ("action_entropy", "action_coverage", "q_margin_mean", "q_max_mean"):
            result[f"progress_bin/{name}/{metric}"] = sum(
                weight * item[metric] for weight, item in zip(weights, values, strict=True)
            )
        minima = [item["q_margin_min"] for item in values if item["action_count"]]
        result[f"progress_bin/{name}/q_margin_min"] = min(minima) if minima else 0.0
        for metric in (
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
        ):
            observed = [item[metric] for item in values if metric in item]
            if observed:
                result[f"progress_bin/{name}/{metric}"] = float(np.mean(observed))
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

    def record(
        self, progress_pct: float, expert_q: float, greedy_q: float, expert_rank: int
    ) -> None:
        index = min(self.bin_count - 1, max(0, int(progress_pct * self.bin_count / 100.0)))
        self._counts[index] += 1
        self._expert_q_totals[index] += expert_q
        self._greedy_q_totals[index] += greedy_q
        self._rank_totals[index] += expert_rank

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for index, count in enumerate(self._counts):
            start = index * 100 // self.bin_count
            end = (index + 1) * 100 // self.bin_count
            name = f"{start:02d}_{end:03d}"
            result[name] = {
                "count": float(count),
                "expert_q_mean": self._expert_q_totals[index] / count if count else 0.0,
                "raw_greedy_q_mean": self._greedy_q_totals[index] / count if count else 0.0,
                "advantage_gap_mean": (
                    (self._greedy_q_totals[index] - self._expert_q_totals[index]) / count
                    if count
                    else 0.0
                ),
                "expert_action_rank_mean": self._rank_totals[index] / count if count else 0.0,
            }
        return result


def aggregate_expert_bins(
    summaries: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, float]]] = {}
    for summary in summaries:
        for name, metrics in summary.items():
            grouped.setdefault(name, []).append(metrics)
    result: dict[str, dict[str, float]] = {}
    for name, values in grouped.items():
        total = sum(item["count"] for item in values)
        result[name] = {
            "count": total,
            **{
                metric: (
                    sum(item["count"] * item[metric] for item in values) / total if total else 0.0
                )
                for metric in (
                    "expert_q_mean",
                    "raw_greedy_q_mean",
                    "advantage_gap_mean",
                    "expert_action_rank_mean",
                )
            },
        }
    return result
