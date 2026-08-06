"""Progress-binned policy diagnostics for TrackMania runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import log
from typing import Any


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

    def record(self, progress_pct: float, action: int, policy: Any) -> None:
        index = self._index(progress_pct)
        if 0 <= action < self.action_count:
            self._actions[index][action] += 1
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
        return {
            "action_count": float(samples),
            "action_entropy": entropy,
            "action_coverage": len(nonzero) / self.action_count,
            "q_margin_mean": self._q_margin_totals[index] / margin_samples
            if margin_samples
            else 0.0,
            "q_margin_min": self._q_margin_minimums[index] if margin_samples else 0.0,
            "q_max_mean": self._q_max_totals[index] / maximum_samples if maximum_samples else 0.0,
        }


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
