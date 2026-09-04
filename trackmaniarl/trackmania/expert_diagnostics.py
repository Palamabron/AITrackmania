"""Expert-action agreement diagnostics for TrackMania policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpertDiagnosticRecord:
    progress_pct: float
    expert_q: float
    greedy_q: float
    expert_rank: int
    expert_action: int
    greedy_action: int
    expert_steering_bin: int
    greedy_steering_bin: int


@dataclass(frozen=True, slots=True)
class _ExpertSwitches:
    eligible: bool
    expert_action: bool
    policy_action: bool
    expert_steering: bool
    policy_steering: bool


@dataclass(frozen=True, slots=True)
class _ExpertMatches:
    exact_action: bool
    steering_bin: bool


@dataclass(frozen=True, slots=True)
class _ExpertDenominators:
    total: int
    action_switches: int
    action_steady: int
    steering_switches: int
    steering_steady: int


@dataclass(slots=True)
class _ExpertAgreement:
    count: int = 0
    exact_action_matches: int = 0
    steering_bin_matches: int = 0
    switch_comparisons: int = 0
    expert_action_switches: int = 0
    policy_action_switches: int = 0
    action_switch_true_positives: int = 0
    expert_steering_switches: int = 0
    policy_steering_switches: int = 0
    steering_switch_true_positives: int = 0
    action_switch_step_matches: int = 0
    action_steady_step_matches: int = 0
    steering_switch_step_matches: int = 0
    steering_steady_step_matches: int = 0

    def record(self, record: ExpertDiagnosticRecord, switches: _ExpertSwitches) -> None:
        matches = _ExpertMatches(
            record.expert_action == record.greedy_action,
            record.expert_steering_bin == record.greedy_steering_bin,
        )
        self.count += 1
        self.exact_action_matches += int(matches.exact_action)
        self.steering_bin_matches += int(matches.steering_bin)
        if switches.eligible:
            self._record_switches(switches)
            self._record_step_matches(switches, matches)

    def _record_switches(self, switches: _ExpertSwitches) -> None:
        self.switch_comparisons += 1
        self.expert_action_switches += int(switches.expert_action)
        self.policy_action_switches += int(switches.policy_action)
        self.action_switch_true_positives += int(switches.expert_action and switches.policy_action)
        self.expert_steering_switches += int(switches.expert_steering)
        self.policy_steering_switches += int(switches.policy_steering)
        self.steering_switch_true_positives += int(
            switches.expert_steering and switches.policy_steering
        )

    def _record_step_matches(self, switches: _ExpertSwitches, matches: _ExpertMatches) -> None:
        self.action_switch_step_matches += int(switches.expert_action and matches.exact_action)
        self.action_steady_step_matches += int(not switches.expert_action and matches.exact_action)
        self.steering_switch_step_matches += int(switches.expert_steering and matches.steering_bin)
        self.steering_steady_step_matches += int(
            not switches.expert_steering and matches.steering_bin
        )

    def summary(self) -> dict[str, float]:
        return _expert_agreement_summary(self._counts())

    def _counts(self) -> dict[str, int]:
        return {
            "count": self.count,
            "exact_action_match_count": self.exact_action_matches,
            "steering_bin_match_count": self.steering_bin_matches,
            "switch_comparison_count": self.switch_comparisons,
            "expert_action_switch_count": self.expert_action_switches,
            "policy_action_switch_count": self.policy_action_switches,
            "action_switch_true_positive_count": self.action_switch_true_positives,
            "expert_steering_switch_count": self.expert_steering_switches,
            "policy_steering_switch_count": self.policy_steering_switches,
            "steering_switch_true_positive_count": self.steering_switch_true_positives,
            "expert_action_switch_step_exact_match_count": self.action_switch_step_matches,
            "expert_action_steady_step_exact_match_count": self.action_steady_step_matches,
            "expert_steering_switch_step_match_count": self.steering_switch_step_matches,
            "expert_steering_steady_step_match_count": self.steering_steady_step_matches,
        }


class ExpertActionDiagnostics:
    def __init__(self, bin_count: int = 10) -> None:
        if bin_count < 1:
            raise ValueError("bin_count must be positive")
        self.bin_count = bin_count
        self._counts = [0] * bin_count
        self._expert_q_totals = [0.0] * bin_count
        self._greedy_q_totals = [0.0] * bin_count
        self._rank_totals = [0.0] * bin_count
        self._agreements = [_ExpertAgreement() for _ in range(bin_count)]
        self._global_agreement = _ExpertAgreement()
        self._previous_expert_action: int | None = None
        self._previous_greedy_action: int | None = None
        self._previous_expert_steering: int | None = None
        self._previous_greedy_steering: int | None = None

    def record(self, record: ExpertDiagnosticRecord) -> None:
        index = self._index(record.progress_pct)
        self._counts[index] += 1
        self._expert_q_totals[index] += record.expert_q
        self._greedy_q_totals[index] += record.greedy_q
        self._rank_totals[index] += record.expert_rank
        switches = self._switches(record)
        self._agreements[index].record(record, switches)
        self._global_agreement.record(record, switches)
        self._remember_actions(record)

    def _index(self, progress_pct: float) -> int:
        return min(
            self.bin_count - 1,
            max(0, int(progress_pct * self.bin_count / 100.0)),
        )

    def _switches(self, record: ExpertDiagnosticRecord) -> _ExpertSwitches:
        expert_action = self._previous_expert_action
        greedy_action = self._previous_greedy_action
        expert_steering = self._previous_expert_steering
        greedy_steering = self._previous_greedy_steering
        eligible = expert_action is not None
        return _ExpertSwitches(
            eligible,
            eligible and expert_action != record.expert_action,
            eligible and greedy_action != record.greedy_action,
            eligible and expert_steering != record.expert_steering_bin,
            eligible and greedy_steering != record.greedy_steering_bin,
        )

    def _remember_actions(self, record: ExpertDiagnosticRecord) -> None:
        self._previous_expert_action = record.expert_action
        self._previous_greedy_action = record.greedy_action
        self._previous_expert_steering = record.expert_steering_bin
        self._previous_greedy_steering = record.greedy_steering_bin

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            self._name(index): self._summary_at(index, count)
            for index, count in enumerate(self._counts)
        }

    def action_summary(self) -> dict[str, float]:
        return self._global_agreement.summary()

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
            **self._agreements[index].summary(),
        }


def aggregate_expert_bins(
    summaries: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    grouped = _group_expert_summaries(summaries)
    return {name: _aggregate_expert_bin(values) for name, values in grouped.items()}


def _group_expert_summaries(
    summaries: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, list[Mapping[str, float]]]:
    grouped: dict[str, list[Mapping[str, float]]] = {}
    for summary in summaries:
        for name, metrics in summary.items():
            grouped.setdefault(name, []).append(metrics)
    return grouped


def _aggregate_expert_bin(values: list[Mapping[str, float]]) -> dict[str, float]:
    total = sum(item["count"] for item in values)
    metrics = (
        "expert_q_mean",
        "raw_greedy_q_mean",
        "advantage_gap_mean",
        "expert_action_rank_mean",
    )
    return {
        **_weighted_q_metrics(values, metrics, total),
        **aggregate_expert_actions(values),
    }


def _weighted_q_metrics(
    values: list[Mapping[str, float]], metrics: tuple[str, ...], total: float
) -> dict[str, float]:
    return {
        metric: sum(item["count"] * item[metric] for item in values) / total if total else 0.0
        for metric in metrics
    }


_EXPERT_COUNT_METRICS = (
    "count",
    "exact_action_match_count",
    "steering_bin_match_count",
    "switch_comparison_count",
    "expert_action_switch_count",
    "policy_action_switch_count",
    "action_switch_true_positive_count",
    "expert_steering_switch_count",
    "policy_steering_switch_count",
    "steering_switch_true_positive_count",
    "expert_action_switch_step_exact_match_count",
    "expert_action_steady_step_exact_match_count",
    "expert_steering_switch_step_match_count",
    "expert_steering_steady_step_match_count",
)


def aggregate_expert_actions(
    summaries: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    values = list(summaries)
    counts = {
        metric: int(sum(item.get(metric, 0.0) for item in values))
        for metric in _EXPERT_COUNT_METRICS
    }
    return _expert_agreement_summary(counts)


def _expert_agreement_summary(counts: Mapping[str, int]) -> dict[str, float]:
    denominators = _expert_denominators(counts)
    return {
        **{metric: float(counts[metric]) for metric in _EXPERT_COUNT_METRICS},
        **_base_accuracy(counts, denominators),
        **_action_step_accuracy(counts, denominators),
        **_steering_step_accuracy(counts, denominators),
    }


def _expert_denominators(counts: Mapping[str, int]) -> _ExpertDenominators:
    comparisons = counts["switch_comparison_count"]
    action_switches = counts["expert_action_switch_count"]
    steering_switches = counts["expert_steering_switch_count"]
    return _ExpertDenominators(
        counts["count"],
        action_switches,
        comparisons - action_switches,
        steering_switches,
        comparisons - steering_switches,
    )


def _base_accuracy(
    counts: Mapping[str, int], denominators: _ExpertDenominators
) -> dict[str, float]:
    return {
        "exact_action_accuracy": _ratio(counts["exact_action_match_count"], denominators.total),
        "steering_bin_accuracy": _ratio(counts["steering_bin_match_count"], denominators.total),
        "action_switch_recall": _ratio(
            counts["action_switch_true_positive_count"], denominators.action_switches
        ),
        "steering_switch_recall": _ratio(
            counts["steering_switch_true_positive_count"], denominators.steering_switches
        ),
    }


def _action_step_accuracy(
    counts: Mapping[str, int], denominators: _ExpertDenominators
) -> dict[str, float]:
    return {
        "expert_action_steady_step_count": float(denominators.action_steady),
        "expert_action_switch_step_exact_accuracy": _ratio(
            counts["expert_action_switch_step_exact_match_count"],
            denominators.action_switches,
        ),
        "expert_action_steady_step_exact_accuracy": _ratio(
            counts["expert_action_steady_step_exact_match_count"],
            denominators.action_steady,
        ),
    }


def _steering_step_accuracy(
    counts: Mapping[str, int], denominators: _ExpertDenominators
) -> dict[str, float]:
    return {
        "expert_steering_steady_step_count": float(denominators.steering_steady),
        "expert_steering_switch_step_accuracy": _ratio(
            counts["expert_steering_switch_step_match_count"],
            denominators.steering_switches,
        ),
        "expert_steering_steady_step_accuracy": _ratio(
            counts["expert_steering_steady_step_match_count"],
            denominators.steering_steady,
        ),
    }


def _ratio(numerator: int, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0
