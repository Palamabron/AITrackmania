#!/usr/bin/env python3
"""Render health reports from normalized W&B analysis artifacts.

Usage:
    uv run python scripts/iteration_report.py
    uv run python scripts/iteration_report.py --format json --out reports/wandb.json
    uv run python scripts/iteration_report.py --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
SUPPORTED_SCHEMA_VERSION = "2.0"


def _load_analyses() -> dict[str, dict[str, Any]]:
    analyses: dict[str, dict[str, Any]] = {}
    unsupported_count = 0
    for path in sorted(ANALYSIS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: ignoring invalid JSON: {path}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or data.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            unsupported_count += 1
            continue
        exp_id = data.get("exp_id")
        if not isinstance(exp_id, str) or not exp_id:
            print(f"WARNING: ignoring analysis without exp_id: {path}", file=sys.stderr)
            continue
        analyses[exp_id] = data
    if unsupported_count:
        print(
            f"WARNING: ignored {unsupported_count} legacy analysis artifacts; "
            "refresh them with fetch_analysis.py to include them.",
            file=sys.stderr,
        )
    return analyses


def _metric_matches(name: str, keywords: Iterable[str]) -> bool:
    lower_name = name.lower()
    return all(keyword in lower_name for keyword in keywords)


def _first_metric(
    metrics: Mapping[str, Any], *keywords: str
) -> tuple[str, Mapping[str, Any]] | None:
    matches = [
        (name, values)
        for name, values in metrics.items()
        if isinstance(values, Mapping) and _metric_matches(name, keywords)
    ]
    return sorted(matches, key=lambda item: item[0])[0] if matches else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _metric(metrics: Mapping[str, Any], *names: str) -> tuple[str, Mapping[str, Any]] | None:
    for name in names:
        values = metrics.get(name)
        if isinstance(values, Mapping):
            return name, values
    return None


def _metric_rows(
    metrics: Mapping[str, Any], exact: tuple[tuple[str, ...], ...], prefixes: tuple[str, ...]
) -> list[dict[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for names in exact:
        match = _metric(metrics, *names)
        if match is not None:
            selected[match[0]] = match[1]
    for name, values in metrics.items():
        if isinstance(values, Mapping) and name.startswith(prefixes):
            selected[name] = values
    return [_metric_row(name, values) for name, values in sorted(selected.items())]


def _metric_row(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "count": int(values["count"]),
        "last": _number(values["last"]),
        "recent_mean": _number(values["recent_mean"]),
        "prior_mean": _number(values["prior_mean"]),
        "recent_delta": _number(values["recent_delta"]),
        "recent_relative_delta": _number(values["recent_relative_delta"]),
        "p05": _number(values["p05"]),
        "p95": _number(values["p95"]),
    }


def _section(metrics: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    sections = {
        "episode_health": (
            (
                ("episode/return", "episode/reward"),
                ("episode/progress_pct",),
                ("episode/finish_rate",),
                ("episode/best_finish_time_s",),
                ("episode/finish_time_s",),
                ("episode/steps", "episode/transitions"),
                ("episode/race_time_s",),
                ("episode/exploration_epsilon",),
            ),
            ("episode/reward/", "episode/termination/", "episode/velocity/"),
        ),
        "learner_stability": (
            (
                ("learner/q_mean", "learner/debug/q_selected_mean"),
                ("learner/q_abs_max",),
                ("learner/gradient_norm_max", "learner/debug/gradient_norm"),
                ("learner/clipped_fraction", "learner/debug/gradient_clipped_fraction"),
                ("learner/debug/td_abs_mean",),
                ("learner/debug/td_abs_max",),
            ),
            ("learner/loss/",),
        ),
        "replay_and_schedule": (
            (
                ("replay/size",),
                ("replay/fill_fraction",),
                ("replay/per_beta",),
                ("training/update_credit",),
                ("training/finish_rate",),
            ),
            (),
        ),
        "throughput_and_backlog": (
            (
                ("training/transitions_per_s",),
                ("training/updates_per_s",),
                ("training/update_throughput_ratio",),
                ("training/update_backlog_s",),
                ("training/rollout_queue_depth",),
                ("performance/learner_update_s",),
                ("performance/replay_wait_s",),
            ),
            (),
        ),
        "actor_health": (
            (
                ("actor/ingest_fps",),
                ("actor/policy_lag_updates",),
                ("actor/queue_delay_s",),
                ("actor/rollout_queue_depth",),
                ("actor/heartbeat/spool_bytes",),
                ("actor/timeout/silence_s",),
            ),
            ("actor/policy/",),
        ),
        "evaluation": (
            (
                ("eval/suite/eval/finish_rate", "eval/episode/finish_rate"),
                ("eval/suite/eval/median_finish_time_s",),
                ("eval/suite/eval/finish_time_s", "eval/episode/finish_time_s"),
                ("eval/suite/eval/crash_rate",),
                ("eval/suite/eval/reward", "eval/episode/return"),
            ),
            (),
        ),
    }
    exact, prefixes = sections[key]
    return _metric_rows(metrics, exact, prefixes)


def _categories(analysis: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    categories = analysis.get("categories")
    if not isinstance(categories, Mapping):
        return {}
    return {
        str(name): {str(value): int(count) for value, count in values.items()}
        for name, values in categories.items()
        if isinstance(values, Mapping)
    }


def _alerts(analysis: Mapping[str, Any], sections: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    history = analysis["history"]
    run = analysis.get("run")
    metrics = analysis["metrics"]
    alerts: list[str] = []
    if int(history["rows_scanned"]) < 10:
        alerts.append("History is too short for reliable trend estimates.")
    if isinstance(run, Mapping) and run.get("state") not in {"running", "finished"}:
        alerts.append(f"Run state is '{run.get('state')}', so the final history may be incomplete.")
    for row in sections["learner_stability"]:
        if row["name"].startswith("learner/loss/") and (row["recent_relative_delta"] or 0.0) > 0.5:
            alerts.append(f"{row['name']} rose by more than 50% in the recent window.")
    clipped = _metric(metrics, "learner/clipped_fraction")
    clipping_rate = _number(clipped[1]["recent_mean"]) if clipped is not None else None
    if clipping_rate is not None and clipping_rate > 0.5:
        alerts.append("More than half of recent learner updates were gradient-clipped.")
    backlog = _metric(metrics, "training/update_backlog_s")
    if backlog is not None and (_number(backlog[1]["recent_mean"]) or 0.0) > 60.0:
        alerts.append("Learner update backlog exceeds one minute.")
    no_progress = _section_metric(sections["episode_health"], "episode/termination/no_progress")
    if _recent(no_progress) is not None and _recent(no_progress) > 0.5:
        alerts.append("No-progress termination is the dominant recent episode outcome.")
    finish_rate = _section_metric(sections["episode_health"], "episode/finish_rate")
    if _last(finish_rate) == 0.0 and finish_rate is not None and int(finish_rate["count"]) >= 25:
        alerts.append("No training episode has finished despite at least 25 recorded episodes.")
    eval_finish_rate = _section_metric(sections["evaluation"], "eval/episode/finish_rate")
    if _last(eval_finish_rate) == 0.0 and eval_finish_rate is not None:
        alerts.append("Deterministic evaluation has not recorded a finish.")
    transitions = _section_metric(sections["throughput_and_backlog"], "training/transitions_per_s")
    if transitions is not None and (_number(transitions["recent_relative_delta"]) or 0.0) < -0.3:
        alerts.append("Recent collection throughput is more than 30% below the prior window.")
    update_time = _section_metric(
        sections["throughput_and_backlog"], "performance/learner_update_s"
    )
    if update_time is not None and (_number(update_time["recent_relative_delta"]) or 0.0) > 0.5:
        alerts.append("Recent learner update latency is more than 50% above the prior window.")
    return alerts


def _section_metric(rows: Iterable[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if row["name"] == name), None)


def _diagnostics(exp_id: str, analysis: Mapping[str, Any]) -> dict[str, Any]:
    history = analysis.get("history")
    metrics = analysis.get("metrics")
    if not isinstance(history, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError(f"Analysis for {exp_id} has an invalid normalized schema")
    sections = {
        key: _section(metrics, key)
        for key in (
            "episode_health",
            "learner_stability",
            "replay_and_schedule",
            "throughput_and_backlog",
            "actor_health",
            "evaluation",
        )
    }
    return {
        "exp_id": exp_id,
        "run": analysis.get("run", {}),
        "history": dict(history),
        "categories": _categories(analysis),
        "sections": sections,
        "alerts": _alerts(analysis, sections),
    }


def build_json_report(analyses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    experiments = [_diagnostics(exp_id, analysis) for exp_id, analysis in sorted(analyses.items())]
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


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
        delta = _number(row["recent_delta"])
        delta_text = f"{delta:+.4g}" if delta is not None else "n/a"
        interval = f"{_format_number(_number(row['p05']))}-{_format_number(_number(row['p95']))}"
        lines.append(
            f"| `{row['name']}` | {_format_number(_number(row['last']))} | "
            f"{_format_number(_number(row['recent_mean']))} | "
            f"{_format_number(_number(row['prior_mean']))} | {delta_text} | {interval} |"
        )
    return lines


def _render_categories(categories: Mapping[str, Mapping[str, int]]) -> list[str]:
    termination = categories.get("episode/termination")
    if not termination:
        return []
    rows = ", ".join(f"`{name}`: {count}" for name, count in sorted(termination.items()))
    return ["", "### Termination reasons", "", rows]


def _comparison_rows(experiments: Iterable[Mapping[str, Any]]) -> list[str]:
    rows = list(experiments)
    if len(rows) < 2:
        return []
    lines = [
        "",
        "## Cross-run comparison",
        "",
        "| Run | State | Return | Progress | Finish rate | Loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in rows:
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
        lines.append(
            f"| {experiment['exp_id']} | {run.get('state', 'unknown')} | "
            + " | ".join(_format_number(value) for value in values)
            + " |"
        )
    return lines


def _recent(row: Mapping[str, Any] | None) -> float | None:
    return _number(row.get("recent_mean")) if row is not None else None


def _last(row: Mapping[str, Any] | None) -> float | None:
    return _number(row.get("last")) if row is not None else None


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# W&B RL Diagnostic Report",
        "",
        f"Experiments analyzed: {report['experiment_count']}",
    ]
    lines.extend(_comparison_rows(report["experiments"]))
    titles = {
        "episode_health": "Episode health",
        "learner_stability": "Learner stability",
        "replay_and_schedule": "Replay and schedule",
        "throughput_and_backlog": "Throughput and backlog",
        "actor_health": "Actor and policy health",
        "evaluation": "Evaluation",
    }
    for experiment in report["experiments"]:
        run = experiment["run"] if isinstance(experiment["run"], Mapping) else {}
        history = experiment["history"]
        lines.extend(
            [
                "",
                f"## {experiment['exp_id']}",
                f"State: {run.get('state', 'unknown')} | "
                f"History rows: {history.get('rows_scanned', 0)} | "
                f"Numeric metrics: {history.get('metric_count', 0)}",
            ]
        )
        for key, title in titles.items():
            lines.extend(_render_metrics(title, experiment["sections"][key]))
        lines.extend(_render_categories(experiment["categories"]))
        if experiment["alerts"]:
            lines.extend(["", "### Alerts", ""])
            lines.extend(f"- {alert}" for alert in experiment["alerts"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render normalized W&B experiment analyses")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("--out", default="", help="Write to file instead of stdout")
    args = parser.parse_args()

    analyses = _load_analyses()
    if not analyses:
        print(
            "No normalized analysis files found. Run: "
            "uv run python scripts/fetch_analysis.py --run ENTITY/PROJECT/RUN_ID",
            file=sys.stderr,
        )
        sys.exit(1)
    report = build_json_report(analyses)
    if args.format == "json":
        output = json.dumps(report, indent=2, default=str)
    else:
        output = build_markdown_report(report)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
