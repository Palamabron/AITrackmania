#!/usr/bin/env python3
"""Fetch normalized TMRL experiment analyses from Weights & Biases.

Usage:
    uv run python scripts/fetch_analysis.py --run dsc-pjatk-warsaw/my-trackmania-agent/z67iytmc
    uv run python scripts/fetch_analysis.py --run ENTITY/PROJECT/RUN_ID --force
    uv run python scripts/fetch_analysis.py --exp-id registered-experiment
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"
ANALYSIS_SCHEMA_VERSION = "2.0"
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


class WandbRun(Protocol):
    id: str
    name: str
    state: str
    url: str
    config: Mapping[str, Any]
    summary: Mapping[str, Any]

    def scan_history(self, *, page_size: int) -> Iterable[Mapping[str, Any]]: ...


class WandbApi(Protocol):
    def run(self, path: str) -> WandbRun: ...


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
class RunRequest:
    path: str
    output_id: str


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


def _read_registry() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def parse_run_path(value: str) -> str:
    """Validate an entity/project/run identifier accepted by W&B."""

    parts = value.strip("/").split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("run path must have the form ENTITY/PROJECT/RUN_ID")
    return "/".join(parts)


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


def metric_stats(values: list[float]) -> MetricStats:
    """Summarize a metric and compare its most recent observations."""

    if not values:
        raise ValueError("metric statistics require at least one value")
    recent_size = min(len(values), max(10, math.ceil(len(values) * 0.1)))
    recent = values[-recent_size:]
    prior = values[-2 * recent_size : -recent_size]
    recent_mean = sum(recent) / len(recent)
    prior_mean = sum(prior) / len(prior) if prior else None
    delta = recent_mean - prior_mean if prior_mean is not None else None
    relative_delta = delta / abs(prior_mean) if prior_mean not in (None, 0.0) else None
    return MetricStats(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        median=median(values),
        last=values[-1],
        p05=_percentile(values, 0.05),
        p95=_percentile(values, 0.95),
        recent_mean=recent_mean,
        prior_mean=prior_mean,
        recent_delta=delta,
        recent_relative_delta=relative_delta,
    )


def _metric_group(name: str) -> str:
    prefix = name.partition("/")[0]
    return prefix if prefix in METRIC_GROUPS else "custom"


def _history_values(
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


def _run_field(run: WandbRun, name: str) -> Any:
    return getattr(run, name, None)


def analyze_run(run: WandbRun, path: str, output_id: str) -> dict[str, Any]:
    """Build a tracker-neutral analysis document for one W&B run."""

    rows, values, categories = _history_values(run.scan_history(page_size=1_000))
    metrics = {name: asdict(metric_stats(series)) for name, series in sorted(values.items())}
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for name in metrics:
        groups[_metric_group(name)].append(name)
    warnings = (
        ["History is empty; only run metadata and W&B summary are available."] if rows == 0 else []
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "exp_id": output_id,
        "wandb_run_path": path,
        "run": {
            "id": _run_field(run, "id"),
            "name": _run_field(run, "name"),
            "state": _run_field(run, "state"),
            "url": _run_field(run, "url"),
            "created_at": _run_field(run, "created_at"),
            "updated_at": _run_field(run, "updated_at"),
        },
        "config": dict(run.config or {}),
        "wandb_summary": dict(run.summary or {}),
        "history": {
            "rows_scanned": rows,
            "metric_count": len(metrics),
            "metric_groups": dict(sorted(groups.items())),
        },
        "metrics": metrics,
        "categories": dict(sorted(categories.items())),
        "warnings": warnings,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def _registry_request(entry: Mapping[str, Any], entity: str, project: str) -> RunRequest:
    exp_id = str(entry["exp_id"])
    explicit_path = entry.get("wandb_run_path")
    if isinstance(explicit_path, str):
        return RunRequest(parse_run_path(explicit_path), exp_id)
    run_id = str(entry.get("wandb_run_id", exp_id))
    return RunRequest(parse_run_path(f"{entity}/{project}/{run_id}"), exp_id)


def _requests_from_args(args: argparse.Namespace) -> list[RunRequest]:
    if args.run:
        if args.exp_id:
            raise ValueError("--run and --exp-id cannot be used together")
        return [
            RunRequest(parse_run_path(path), parse_run_path(path).split("/")[-1])
            for path in args.run
        ]
    entries = _read_registry()
    selected = [entry for entry in entries if not args.exp_id or entry["exp_id"] == args.exp_id]
    if args.exp_id and not selected:
        raise ValueError(f"No registry entry found for experiment '{args.exp_id}'")
    requests = [_registry_request(entry, args.entity, args.project) for entry in selected]
    if not requests:
        raise ValueError("provide --run or register an experiment in experiments/registry.jsonl")
    return requests


def _should_fetch(path: Path, force: bool, stale_hours: float) -> bool:
    if force or not path.exists():
        return True
    if stale_hours <= 0:
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours >= stale_hours


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _create_api(timeout: int) -> WandbApi:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install the W&B SDK with `uv sync --group dev`.") from exc
    api_type = getattr(wandb, "Api", None)
    if api_type is None:
        raise RuntimeError("The installed W&B SDK is incomplete; run `uv sync --group dev`.")
    return api_type(timeout=timeout)


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Fetch normalized analyses from Weights & Biases")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ENTITY/PROJECT/RUN_ID",
        help="Fetch an explicit W&B run; may be repeated",
    )
    parser.add_argument("--exp-id", default="", help="Fetch one registered experiment")
    parser.add_argument("--entity", default="dsc-pjatk-warsaw")
    parser.add_argument("--project", default="my-trackmania-agent")
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis files")
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=0,
        help="Refresh existing files only when they are at least N hours old (0=never)",
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set. Export it or add to .env")
        raise SystemExit(2)

    try:
        api = _create_api(args.timeout)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc
    fetched, skipped, failed = 0, 0, 0
    try:
        requests = _requests_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    for request in requests:
        out_path = ANALYSIS_DIR / f"{request.output_id}.json"
        if not _should_fetch(out_path, args.force, args.stale_hours):
            print(f"  [SKIP] {request.path} (existing analysis is current)")
            skipped += 1
            continue
        print(f"  Fetching {request.path}...")
        try:
            result = analyze_run(api.run(request.path), request.path, request.output_id)
        except Exception as exc:
            print(f"  [FAIL] {request.path}: {exc}")
            failed += 1
            continue
        _write_json(out_path, result)
        print(
            f"  [OK]   {request.output_id}: {result['history']['rows_scanned']} history rows, "
            f"{result['history']['metric_count']} numeric metrics"
        )
        fetched += 1

    print(f"\nDone: fetched={fetched}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
