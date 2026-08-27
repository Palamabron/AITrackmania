from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

METRIC_GROUPS = (
    "episode",
    "learner",
    "training",
    "replay",
    "performance",
    "actor",
    "eval",
    "evaluation",
)


@dataclass(frozen=True, slots=True)
class MetricStats:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    last: float
    p05: float
    p95: float
    recent_mean: float
    prior_mean: float | None
    recent_delta: float | None
    recent_relative_delta: float | None


@dataclass(frozen=True, slots=True)
class MetricTrend:
    recent_mean: float
    prior_mean: float | None
    delta: float | None
    relative_delta: float | None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_trend(values: list[float]) -> MetricTrend:
    recent_size = min(len(values), max(10, math.ceil(len(values) * 0.1)))
    recent = values[-recent_size:]
    prior = values[-2 * recent_size : -recent_size]
    recent_mean = sum(recent) / len(recent)
    prior_mean = sum(prior) / len(prior) if prior else None
    delta = recent_mean - prior_mean if prior_mean is not None else None
    relative_delta = (
        (recent_mean - prior_mean) / abs(prior_mean)
        if prior_mean is not None and prior_mean != 0.0
        else None
    )
    return MetricTrend(recent_mean, prior_mean, delta, relative_delta)


def metric_stats(values: list[float]) -> MetricStats:
    if not values:
        raise ValueError("metric statistics require at least one value")
    trend = _metric_trend(values)
    return MetricStats(
        len(values),
        min(values),
        max(values),
        sum(values) / len(values),
        median(values),
        values[-1],
        _percentile(values, 0.05),
        _percentile(values, 0.95),
        trend.recent_mean,
        trend.prior_mean,
        trend.delta,
        trend.relative_delta,
    )


def _metric_group(name: str) -> str:
    prefix = name.partition("/")[0]
    return prefix if prefix in METRIC_GROUPS else "custom"


def history_values(
    history: Iterable[Mapping[str, Any]],
) -> tuple[int, dict[str, list[float]], dict[str, dict[str, int]]]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    categories: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows = 0
    for row in history:
        rows += 1
        for name, value in row.items():
            number = _numeric_value(value)
            if number is not None and not name.startswith("_"):
                values[name].append(number)
            elif isinstance(value, str) and value and not name.startswith("_"):
                categories[name][value] += 1
    return rows, dict(values), {name: dict(counts) for name, counts in categories.items()}


def metric_summaries(
    values: dict[str, list[float]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    metrics = {name: asdict(metric_stats(series)) for name, series in sorted(values.items())}
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for name in metrics:
        groups[_metric_group(name)].append(name)
    return metrics, dict(sorted(groups.items()))
