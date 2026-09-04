#!/usr/bin/env python3
"""Fetch normalized TrackmaniaRL experiment analyses from Weights & Biases.

Usage:
    uv run python scripts/fetch_analysis.py --run dsc-pjatk-warsaw/my-trackmania-agent/z67iytmc
    uv run python scripts/fetch_analysis.py --run ENTITY/PROJECT/RUN_ID --force
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING or __name__ != "__main__":
    from scripts.fetch_analysis_metrics import (
        history_values as _history_values,
    )
    from scripts.fetch_analysis_metrics import (
        metric_summaries as _metric_summaries,
    )
else:
    from fetch_analysis_metrics import (
        history_values as _history_values,
    )
    from fetch_analysis_metrics import (
        metric_summaries as _metric_summaries,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
ANALYSIS_SCHEMA_VERSION = "2.0"


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
class RunRequest:
    path: str
    output_id: str


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    force: bool
    stale_hours: float


@dataclass(frozen=True, slots=True)
class FetchArguments:
    runs: list[str]
    timeout: int
    policy: FetchPolicy


class FetchOutcome(Enum):
    FETCHED = "fetched"
    SKIPPED = "skipped"


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


def parse_run_path(value: str) -> str:
    """Validate an entity/project/run identifier accepted by W&B."""

    parts = value.strip("/").split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("run path must have the form ENTITY/PROJECT/RUN_ID")
    return "/".join(parts)


def _run_field(run: WandbRun, name: str) -> Any:
    return getattr(run, name, None)


def _run_metadata(run: WandbRun) -> dict[str, Any]:
    names = ("id", "name", "state", "url", "created_at", "updated_at")
    return {name: _run_field(run, name) for name in names}


def _history_summary(
    rows: int, metrics: dict[str, dict[str, Any]], groups: dict[str, list[str]]
) -> dict[str, Any]:
    return {"rows_scanned": rows, "metric_count": len(metrics), "metric_groups": groups}


def analyze_run(run: WandbRun, path: str, output_id: str) -> dict[str, Any]:
    rows, values, categories = _history_values(run.scan_history(page_size=1_000))
    metrics, groups = _metric_summaries(values)
    warnings = (
        ["History is empty; only run metadata and W&B summary are available."] if not rows else []
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "exp_id": output_id,
        "wandb_run_path": path,
        "run": _run_metadata(run),
        "config": dict(run.config or {}),
        "wandb_summary": dict(run.summary or {}),
        "history": _history_summary(rows, metrics, groups),
        "metrics": metrics,
        "categories": dict(sorted(categories.items())),
        "warnings": warnings,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def _requests_from_args(args: FetchArguments) -> list[RunRequest]:
    if not args.runs:
        raise ValueError("provide at least one --run")
    paths = [parse_run_path(path) for path in args.runs]
    return [RunRequest(path, path.split("/")[-1]) for path in paths]


def _should_fetch(path: Path, policy: FetchPolicy) -> bool:
    if policy.force or not path.exists():
        return True
    if policy.stale_hours <= 0:
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours >= policy.stale_hours


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
    return cast(WandbApi, api_type(timeout=timeout))


def _add_refresh_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis files")
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=0,
        help="Refresh existing files only when they are at least N hours old (0=never)",
    )
    parser.add_argument("--timeout", type=int, default=180)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch normalized analyses from Weights & Biases")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ENTITY/PROJECT/RUN_ID",
        help="Fetch an explicit W&B run; may be repeated",
    )
    _add_refresh_options(parser)
    return parser


def _parse_args() -> tuple[argparse.ArgumentParser, FetchArguments]:
    parser = _argument_parser()
    args = parser.parse_args()
    policy = FetchPolicy(args.force, args.stale_hours)
    parsed = FetchArguments(args.run, args.timeout, policy)
    return parser, parsed


def _require_api_key() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set. Export it or add to .env")
        raise SystemExit(2)


def _api_or_exit(timeout: int) -> WandbApi:
    try:
        return _create_api(timeout)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc


def _requests_or_error(parser: argparse.ArgumentParser, args: FetchArguments) -> list[RunRequest]:
    try:
        return _requests_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))


def _fetch_request(api: WandbApi, request: RunRequest, policy: FetchPolicy) -> FetchOutcome:
    out_path = ANALYSIS_DIR / f"{request.output_id}.json"
    if not _should_fetch(out_path, policy):
        print(f"  [SKIP] {request.path} (existing analysis is current)")
        return FetchOutcome.SKIPPED
    print(f"  Fetching {request.path}...")
    result = analyze_run(api.run(request.path), request.path, request.output_id)
    _write_json(out_path, result)
    history = result["history"]
    print(
        f"  [OK]   {request.output_id}: {history['rows_scanned']} history rows, "
        f"{history['metric_count']} numeric metrics"
    )
    return FetchOutcome.FETCHED


def _fetch_all(
    api: WandbApi, requests: list[RunRequest], policy: FetchPolicy
) -> dict[FetchOutcome, int]:
    counts = dict.fromkeys(FetchOutcome, 0)
    for request in requests:
        counts[_fetch_request(api, request, policy)] += 1
    return counts


def main() -> None:
    _load_env()
    parser, args = _parse_args()
    _require_api_key()
    api = _api_or_exit(args.timeout)
    counts = _fetch_all(api, _requests_or_error(parser, args), args.policy)

    fetched = counts[FetchOutcome.FETCHED]
    skipped = counts[FetchOutcome.SKIPPED]
    print(f"\nDone: fetched={fetched}, skipped={skipped}")


if __name__ == "__main__":
    main()
