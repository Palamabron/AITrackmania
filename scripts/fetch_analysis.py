#!/usr/bin/env python3
"""Fetch/refresh analysis JSON files for all experiments from wandb.

Reusable every autoresearch iteration.  Pulls trainer + worker history from
the wandb API, computes summary statistics, and writes per-experiment JSON
files to ``experiments/analysis/``.

Usage:
    python scripts/fetch_analysis.py                  # refresh all experiments
    python scripts/fetch_analysis.py --exp-id foo     # refresh one experiment
    python scripts/fetch_analysis.py --force           # overwrite existing
    python scripts/fetch_analysis.py --stale-hours 2   # only refresh >2h old
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"
BASELINE_PATH = ANALYSIS_DIR / "gtn-baseline.json"


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


def _read_registry() -> list[dict]:
    entries = []
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _safe_stats(h: Any, col: str) -> dict[str, Any] | None:
    if col not in h.columns:
        return None
    s = h[col].astype(float).dropna()
    if len(s) == 0:
        return None
    return {
        "count": len(s),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "last": float(s.iloc[-1]),
        "p5": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


def _phase_analysis(h: Any) -> dict[str, Any]:
    if len(h) < 100:
        return {}
    n = len(h)
    phases = {
        "early": h.iloc[: n // 3],
        "mid": h.iloc[n // 3 : 2 * n // 3],
        "late": h.iloc[2 * n // 3 :],
    }
    result: dict[str, Any] = {}
    for metric in ("loss/iqn_loss", "q/max_q", "q/mean_q", "metrics/return_train"):
        if metric not in h.columns:
            continue
        pmeans: dict[str, float] = {}
        for pname, pdf in phases.items():
            s = pdf[metric].astype(float).dropna()
            if len(s) > 0:
                pmeans[pname] = round(float(s.mean()), 4)
        if len(pmeans) < 2:
            continue
        early_v = pmeans.get("early", 0)
        late_v = pmeans.get("late", 0)
        pct = round((late_v - early_v) / abs(early_v) * 100, 1) if early_v != 0 else 0.0
        if metric == "loss/iqn_loss":
            direction = (
                "improving"
                if late_v < early_v * 0.8
                else "degrading"
                if late_v > early_v * 1.3
                else "stable"
            )
        elif metric == "metrics/return_train":
            direction = (
                "improving"
                if late_v > early_v * 1.3
                else "degrading"
                if late_v < early_v * 0.7
                else "stable"
            )
        else:
            direction = "stable"
        result[metric] = {"phases": pmeans, "pct_change": pct, "direction": direction}
    return result


def _vs_baseline(summary: dict, baseline: dict, key_metrics: list[str]) -> dict:
    deltas: dict[str, Any] = {}
    for mk in key_metrics:
        bs = baseline.get("metrics", {}).get(mk, {})
        es = summary.get("metrics", {}).get(mk, {})
        if bs.get("last") is not None and es.get("last") is not None:
            abs_d = round(es["last"] - bs["last"], 4)
            pct_d = round(abs_d / abs(bs["last"]) * 100, 1) if bs["last"] != 0 else 0.0
            deltas[mk] = {
                "baseline": round(bs["last"], 4),
                "experiment": round(es["last"], 4),
                "delta": abs_d,
                "pct": pct_d,
            }
    return deltas


def fetch_one(
    api: Any,
    entity: str,
    project: str,
    exp_id: str,
    entry: dict | None,
    history_samples: int = 100_000,
) -> dict[str, Any] | None:
    """Fetch and analyze a single experiment from wandb."""
    import pandas as pd

    path = f"{entity}/{project}"
    wandb_run_id = (entry or {}).get("wandb_run_id", exp_id)

    filters = {
        "$or": [
            {"display_name": {"$regex": f"^{wandb_run_id}"}},
            {"name": {"$regex": f"^{wandb_run_id}"}},
        ]
    }
    try:
        runs = list(api.runs(path, filters=filters))
    except Exception as exc:
        print(f"  [WARN] Failed to query runs for {exp_id}: {exc}")
        return None

    trainers = [r for r in runs if (r.name or "").upper().endswith("TRAINER")]
    workers = [r for r in runs if (r.name or "").upper().endswith("WORKER")]

    if not trainers:
        print(f"  [WARN] No TRAINER run found for {exp_id}")
        return None

    trainer = trainers[0]
    h = trainer.history(samples=history_samples)
    if isinstance(h, pd.DataFrame) and h.empty:
        print(f"  [WARN] Empty history for {exp_id}")
        return None

    wandb_config = dict(trainer.config) if trainer.config else {}

    summary: dict[str, Any] = {
        "exp_id": exp_id,
        "wandb_run_id": wandb_run_id,
        "trainer_state": trainer.state,
        "total_rows": len(h),
        "columns": list(h.columns),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    if wandb_config:
        summary["full_config"] = wandb_config

    key_metrics = [
        "loss/iqn_loss",
        "loss/total_loss",
        "loss/monotonicity_penalty",
        "q/mean_q",
        "q/max_q",
        "q/min_q",
        "q/std_q",
        "exploration/epsilon",
        "metrics/return_train",
        "metrics/return_test",
        "eval/return_deterministic",
        "eval/finish_time_test_s",
        "debug/td_abs_mean",
        "debug/grad_norm",
    ]
    summary["metrics"] = {}
    for m in key_metrics:
        stats = _safe_stats(h, m)
        if stats:
            summary["metrics"][m] = stats

    # Finish time (filter out 0 = did not finish, and wandb interpolation artifacts)
    if "eval/finish_time_test_s" in h.columns:
        ft_s = h["eval/finish_time_test_s"].astype(float).dropna()
        ft_s = ft_s[ft_s > 0]

        # wandb history(samples=N) can interpolate between 0 (no finish) and a
        # real finish, producing fake small values. Filter by requiring
        # finished_track_count_test to be close to an integer (>= 0.9 rounds to
        # a real eval cycle, not interpolation).
        if "eval/finished_track_count_test" in h.columns:
            fcount = h.loc[ft_s.index, "eval/finished_track_count_test"].astype(float)
            is_real = (fcount - fcount.round()).abs() < 0.1
            interpolated = (~is_real).sum()
            if interpolated > 0:
                print(f"  [NOTE] Filtered {interpolated} wandb-interpolated finish time rows")
            ft_s = ft_s[is_real]

        if len(ft_s) > 0:
            summary["best_finish_time_s"] = float(ft_s.min())
            summary["last_finish_time_s"] = float(ft_s.iloc[-1])
            summary["median_finish_time_s"] = float(ft_s.median())
        else:
            summary["best_finish_time_s"] = None
    else:
        summary["best_finish_time_s"] = None

    # Grad norm pre-clip stats (important for gradient clipping analysis)
    if "debug/grad_norm_pre_clip" in h.columns:
        pre_clip = _safe_stats(h, "debug/grad_norm_pre_clip")
        if pre_clip:
            summary["metrics"]["debug/grad_norm_pre_clip"] = pre_clip

    # Worker data
    if workers:
        worker = workers[0]
        hw = worker.history(samples=50_000)
        worker_summary: dict[str, Any] = {"worker_state": worker.state}

        if "run/term_reason" in hw.columns:
            vc = hw["run/term_reason"].astype(str).value_counts(dropna=False)
            worker_summary["termination_counts"] = vc.to_dict()
        if "run/finished_track" in hw.columns:
            ft = hw["run/finished_track"].astype(float).dropna()
            worker_summary["finish_rate"] = float(ft.mean()) if len(ft) > 0 else 0.0
        if "run/finish_time" in hw.columns:
            wft = hw["run/finish_time"].astype(float).dropna()
            wft = wft[wft > 0]
            if len(wft) > 0:
                w_best = float(wft.min())
                worker_summary["best_finish_time_s"] = w_best
                worker_summary["finish_count"] = len(wft)
                if summary["best_finish_time_s"] is None or w_best < summary["best_finish_time_s"]:
                    summary["best_finish_time_s"] = w_best
                if summary.get("last_finish_time_s") is None:
                    summary["last_finish_time_s"] = float(wft.iloc[-1])
        if "run/steps" in hw.columns:
            st = hw["run/steps"].astype(float).dropna()
            if len(st) > 0:
                worker_summary["avg_episode_steps"] = float(st.mean())

        summary["worker"] = worker_summary

    # Phase analysis
    trends = _phase_analysis(h)
    if trends:
        summary["training_trends"] = trends

    # Vs baseline comparison
    if BASELINE_PATH.exists() and exp_id != "gtn-baseline":
        try:
            ba = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            deltas = _vs_baseline(summary, ba, key_metrics)
            if deltas:
                summary["vs_baseline"] = deltas
        except Exception:
            pass

    return summary


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Fetch analysis data from wandb")
    parser.add_argument("--exp-id", default="", help="Fetch only this experiment")
    parser.add_argument("--entity", default="dsc-pjatk-warsaw")
    parser.add_argument("--project", default="tmrl")
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis files")
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=0,
        help="Only refresh files older than N hours (0=always)",
    )
    parser.add_argument("--history-samples", type=int, default=100_000)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set. Export it or add to .env")
        raise SystemExit(2)

    import wandb

    api = wandb.Api(timeout=args.timeout)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    entries = _read_registry()
    registry_map = {e["exp_id"]: e for e in entries}

    exp_ids = [args.exp_id] if args.exp_id else [e["exp_id"] for e in entries]

    fetched, skipped, failed = 0, 0, 0
    for exp_id in exp_ids:
        out_path = ANALYSIS_DIR / f"{exp_id}.json"

        if out_path.exists() and not args.force:
            if args.stale_hours > 0:
                age_h = (time.time() - out_path.stat().st_mtime) / 3600
                if age_h < args.stale_hours:
                    print(f"  [SKIP] {exp_id} (age {age_h:.1f}h < {args.stale_hours}h)")
                    skipped += 1
                    continue
            else:
                print(f"  [SKIP] {exp_id} (exists, use --force to overwrite)")
                skipped += 1
                continue

        print(f"  Fetching {exp_id}...")
        entry = registry_map.get(exp_id)
        try:
            result = fetch_one(api, args.entity, args.project, exp_id, entry, args.history_samples)
        except Exception as exc:
            print(f"  [FAIL] {exp_id}: {exc}")
            failed += 1
            continue

        if result is None:
            failed += 1
            continue

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        ft = result.get("best_finish_time_s")
        ft_str = f"{ft:.2f}s" if ft else "DNF"
        print(f"  [OK]   {exp_id} -> {ft_str}")
        fetched += 1

    print(f"\nDone: fetched={fetched}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
