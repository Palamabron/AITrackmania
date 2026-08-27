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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING or __name__ != "__main__":
    from scripts.iteration_report_rendering import build_markdown_report
else:
    from iteration_report_rendering import build_markdown_report

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
SUPPORTED_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class ReportArguments:
    format: str
    output: str


def _read_analysis(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid analysis JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"analysis must be a JSON object: {path}")
    if data.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"analysis must use schema {SUPPORTED_SCHEMA_VERSION}: {path}")
    exp_id = data.get("exp_id")
    if not isinstance(exp_id, str) or not exp_id:
        raise ValueError(f"analysis must contain a non-empty exp_id: {path}")
    return cast(dict[str, Any], data)


def _load_analyses() -> dict[str, dict[str, Any]]:
    analyses: dict[str, dict[str, Any]] = {}
    for path in sorted(ANALYSIS_DIR.glob("*.json")):
        analysis = _read_analysis(path)
        analyses[analysis["exp_id"]] = analysis
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


SECTION_SPECS: dict[str, tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]] = {
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
SECTION_KEYS = tuple(SECTION_SPECS)


def _section(metrics: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    exact, prefixes = SECTION_SPECS[key]
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


def _history_alerts(analysis: Mapping[str, Any]) -> list[str]:
    history = analysis["history"]
    run = analysis.get("run")
    alerts: list[str] = []
    if int(history["rows_scanned"]) < 10:
        alerts.append("History is too short for reliable trend estimates.")
    if isinstance(run, Mapping) and run.get("state") not in {"running", "finished"}:
        alerts.append(f"Run state is '{run.get('state')}', so the final history may be incomplete.")
    return alerts


def _learner_alerts(
    metrics: Mapping[str, Any], sections: Mapping[str, list[dict[str, Any]]]
) -> list[str]:
    alerts: list[str] = []
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
    return alerts


def _episode_alerts(sections: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    alerts: list[str] = []
    no_progress = _section_metric(sections["episode_health"], "episode/termination/no_progress")
    no_progress_rate = _recent(no_progress)
    if no_progress_rate is not None and no_progress_rate > 0.5:
        alerts.append("No-progress termination is the dominant recent episode outcome.")
    finish_rate = _section_metric(sections["episode_health"], "episode/finish_rate")
    if _last(finish_rate) == 0.0 and finish_rate is not None and int(finish_rate["count"]) >= 25:
        alerts.append("No training episode has finished despite at least 25 recorded episodes.")
    eval_finish_rate = _section_metric(sections["evaluation"], "eval/episode/finish_rate")
    if _last(eval_finish_rate) == 0.0 and eval_finish_rate is not None:
        alerts.append("Deterministic evaluation has not recorded a finish.")
    return alerts


def _throughput_alerts(sections: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    alerts: list[str] = []
    transitions = _section_metric(sections["throughput_and_backlog"], "training/transitions_per_s")
    if transitions is not None and (_number(transitions["recent_relative_delta"]) or 0.0) < -0.3:
        alerts.append("Recent collection throughput is more than 30% below the prior window.")
    update_time = _section_metric(
        sections["throughput_and_backlog"], "performance/learner_update_s"
    )
    if update_time is not None and (_number(update_time["recent_relative_delta"]) or 0.0) > 0.5:
        alerts.append("Recent learner update latency is more than 50% above the prior window.")
    return alerts


def _alerts(analysis: Mapping[str, Any], sections: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    metrics = analysis["metrics"]
    return [
        *_history_alerts(analysis),
        *_learner_alerts(metrics, sections),
        *_episode_alerts(sections),
        *_throughput_alerts(sections),
    ]


def _section_metric(rows: Iterable[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if row["name"] == name), None)


def _diagnostic_sections(metrics: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {key: _section(metrics, key) for key in SECTION_KEYS}


def _diagnostics(exp_id: str, analysis: Mapping[str, Any]) -> dict[str, Any]:
    history = analysis.get("history")
    metrics = analysis.get("metrics")
    if not isinstance(history, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError(f"Analysis for {exp_id} has an invalid normalized schema")
    sections = _diagnostic_sections(metrics)
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


def _recent(row: Mapping[str, Any] | None) -> float | None:
    return _number(row.get("recent_mean")) if row is not None else None


def _last(row: Mapping[str, Any] | None) -> float | None:
    return _number(row.get("last")) if row is not None else None


def _parse_args() -> ReportArguments:
    parser = argparse.ArgumentParser(description="Render normalized W&B experiment analyses")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("--out", default="", help="Write to file instead of stdout")
    args = parser.parse_args()
    return ReportArguments(args.format, args.out)


def _require_analyses() -> dict[str, dict[str, Any]]:
    analyses = _load_analyses()
    if not analyses:
        print(
            "No normalized analysis files found. Run: "
            "uv run python scripts/fetch_analysis.py --run ENTITY/PROJECT/RUN_ID",
            file=sys.stderr,
        )
        sys.exit(1)
    return analyses


def _render_report(analyses: Mapping[str, Mapping[str, Any]], output_format: str) -> str:
    report = build_json_report(analyses)
    if output_format == "json":
        return json.dumps(report, indent=2, default=str)
    return build_markdown_report(report)


def _write_report(output: str, target: str) -> None:
    if target:
        Path(target).write_text(output, encoding="utf-8")
        print(f"Report written to {target}", file=sys.stderr)
    else:
        print(output)


def main() -> None:
    args = _parse_args()
    analyses = _require_analyses()
    _write_report(_render_report(analyses, args.format), args.output)


if __name__ == "__main__":
    main()
