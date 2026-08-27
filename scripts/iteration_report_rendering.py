from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SECTION_TITLES = {
    "episode_health": "Episode health",
    "learner_stability": "Learner stability",
    "replay_and_schedule": "Replay and schedule",
    "throughput_and_backlog": "Throughput and backlog",
    "actor_health": "Actor and policy health",
    "evaluation": "Evaluation",
}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _format_number(value: float | None) -> str:
    return f"{value:.4g}" if value is not None else "n/a"


def _render_metrics(title: str, rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "",
        f"### {title}",
        "",
        "| Metric | Last | Recent mean | Prior mean | Delta | P05-P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(_metric_row(row))
    return lines


def _metric_row(row: Mapping[str, Any]) -> str:
    delta = _number(row["recent_delta"])
    delta_text = f"{delta:+.4g}" if delta is not None else "n/a"
    interval = f"{_format_number(_number(row['p05']))}-{_format_number(_number(row['p95']))}"
    return (
        f"| `{row['name']}` | {_format_number(_number(row['last']))} | "
        f"{_format_number(_number(row['recent_mean']))} | "
        f"{_format_number(_number(row['prior_mean']))} | {delta_text} | {interval} |"
    )


def _render_categories(categories: Mapping[str, Mapping[str, int]]) -> list[str]:
    termination = categories.get("episode/termination")
    if not termination:
        return []
    rows = ", ".join(f"`{name}`: {count}" for name, count in sorted(termination.items()))
    return ["", "### Termination reasons", "", rows]


def _comparison_row(experiment: Mapping[str, Any]) -> str:
    section = experiment["sections"]
    episode = {row["name"]: row for row in section["episode_health"]}
    learner = {row["name"]: row for row in section["learner_stability"]}
    run = experiment["run"] if isinstance(experiment["run"], Mapping) else {}
    values = (
        _recent(episode.get("episode/return") or episode.get("episode/reward")),
        _recent(episode.get("episode/progress_pct")),
        _last(episode.get("episode/finish_rate")),
        _recent(learner.get("learner/loss/iqn")),
    )
    metrics = " | ".join(_format_number(value) for value in values)
    return f"| {experiment['exp_id']} | {run.get('state', 'unknown')} | {metrics} |"


def _comparison_rows(experiments: Iterable[Mapping[str, Any]]) -> list[str]:
    rows = list(experiments)
    if len(rows) < 2:
        return []
    header = [
        "",
        "## Cross-run comparison",
        "",
        "| Run | State | Return | Progress | Finish rate | Loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    return [*header, *map(_comparison_row, rows)]


def _recent(row: Mapping[str, Any] | None) -> float | None:
    return _number(row.get("recent_mean")) if row is not None else None


def _last(row: Mapping[str, Any] | None) -> float | None:
    return _number(row.get("last")) if row is not None else None


def _alert_lines(alerts: list[str]) -> list[str]:
    if not alerts:
        return []
    return ["", "### Alerts", "", *(f"- {alert}" for alert in alerts)]


def _experiment_lines(experiment: Mapping[str, Any]) -> list[str]:
    run = experiment["run"] if isinstance(experiment["run"], Mapping) else {}
    history = experiment["history"]
    lines = [
        "",
        f"## {experiment['exp_id']}",
        f"State: {run.get('state', 'unknown')} | History rows: "
        f"{history.get('rows_scanned', 0)} | Numeric metrics: {history.get('metric_count', 0)}",
    ]
    for key, title in SECTION_TITLES.items():
        lines.extend(_render_metrics(title, experiment["sections"][key]))
    lines.extend(_render_categories(experiment["categories"]))
    lines.extend(_alert_lines(experiment["alerts"]))
    return lines


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# W&B RL Diagnostic Report",
        "",
        f"Experiments analyzed: {report['experiment_count']}",
    ]
    lines.extend(_comparison_rows(report["experiments"]))
    for experiment in report["experiments"]:
        lines.extend(_experiment_lines(experiment))
    return "\n".join(lines)
