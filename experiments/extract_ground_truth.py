#!/usr/bin/env python3
"""Extract deterministic ground-truth metrics from experiments/analysis/*.json.

Produces a stable, machine-readable summary that agents can rely on instead of
interpreting snapshot data at checkpoint time.  The output is written to
``experiments/ground_truth.json`` and also printed to stdout.

Usage:
    python experiments/extract_ground_truth.py
    python experiments/extract_ground_truth.py --format table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "ground_truth.json"


def _load_registry() -> list[dict]:
    entries = []
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _load_analysis(exp_id: str) -> dict | None:
    p = ANALYSIS_DIR / f"{exp_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def extract_experiment(exp_id: str, registry_entry: dict | None, analysis: dict | None) -> dict:
    """Extract canonical metrics for one experiment."""
    result: dict = {"exp_id": exp_id}

    if registry_entry:
        result["status"] = registry_entry.get("status")
        result["stop_reason"] = registry_entry.get("stop_reason")
        result["hypothesis"] = registry_entry.get("hypothesis")
        result["config_overrides"] = registry_entry.get("config_overrides", {})
        sm = registry_entry.get("summary_metrics") or {}
        result["registry_best_finish_time_s"] = sm.get("best_finish_time_s")

    if not analysis:
        result["has_analysis"] = False
        return result

    result["has_analysis"] = True

    best_ft = analysis.get("best_finish_time_s")
    if best_ft == 0 or best_ft == 0.0:
        best_ft = None
    result["best_finish_time_s"] = best_ft
    result["last_finish_time_s"] = analysis.get("last_finish_time_s")
    result["median_finish_time_s"] = analysis.get("median_finish_time_s")

    worker = analysis.get("worker", {})
    result["worker_best_finish_time_s"] = worker.get("best_finish_time_s")
    result["worker_finish_count"] = worker.get("finish_count", 0)
    result["worker_finish_rate"] = worker.get("finish_rate", 0)

    metrics = analysis.get("metrics", {})
    result["loss_iqn_median"] = _safe_get(metrics, "loss/iqn_loss", "median")
    result["loss_iqn_last"] = _safe_get(metrics, "loss/iqn_loss", "last")
    result["loss_iqn_mean"] = _safe_get(metrics, "loss/iqn_loss", "mean")
    result["max_q_last"] = _safe_get(metrics, "q/max_q", "last")
    result["mean_q_last"] = _safe_get(metrics, "q/mean_q", "last")
    result["epsilon_last"] = _safe_get(metrics, "exploration/epsilon", "last")
    result["grad_norm_last"] = _safe_get(metrics, "debug/grad_norm", "last")
    result["return_train_last"] = _safe_get(metrics, "metrics/return_train", "last")
    result["return_test_last"] = _safe_get(metrics, "metrics/return_test", "last")
    result["eval_finish_time_max"] = _safe_get(metrics, "eval/finish_time_test_s", "max")

    trends = analysis.get("training_trends", {})
    result["loss_trend"] = _safe_get(trends, "loss/iqn_loss", "direction")
    result["return_trend"] = _safe_get(trends, "metrics/return_train", "direction")

    return result


def build_ground_truth() -> dict:
    registry_entries = _load_registry()
    registry_map = {e["exp_id"]: e for e in registry_entries}

    all_analysis_files = sorted(ANALYSIS_DIR.glob("*.json"))
    all_exp_ids = set(registry_map.keys())
    for f in all_analysis_files:
        all_exp_ids.add(f.stem)

    experiments = []
    for exp_id in sorted(all_exp_ids):
        analysis = _load_analysis(exp_id)
        registry_entry = registry_map.get(exp_id)
        experiments.append(extract_experiment(exp_id, registry_entry, analysis))

    with_finish = sorted(
        [e for e in experiments if e.get("best_finish_time_s") and e["best_finish_time_s"] > 0],
        key=lambda e: e["best_finish_time_s"],
    )
    without_finish = [
        e for e in experiments if not e.get("best_finish_time_s") or e["best_finish_time_s"] <= 0
    ]

    leaderboard = []
    for rank, e in enumerate(with_finish, 1):
        leaderboard.append(
            {
                "rank": rank,
                "exp_id": e["exp_id"],
                "best_finish_time_s": e["best_finish_time_s"],
                "worker_finish_rate": e.get("worker_finish_rate"),
                "status": e.get("status"),
            }
        )
    for e in without_finish:
        leaderboard.append(
            {
                "rank": None,
                "exp_id": e["exp_id"],
                "best_finish_time_s": None,
                "worker_finish_rate": e.get("worker_finish_rate"),
                "status": e.get("status"),
            }
        )

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "experiments_with_finish": len(with_finish),
        "best_overall": with_finish[0] if with_finish else None,
        "leaderboard": leaderboard,
        "experiments": experiments,
    }


def print_table(gt: dict) -> None:
    print(f"\n{'=' * 90}")
    print(f"EXPERIMENT GROUND TRUTH  (generated: {gt['generated_at']})")
    print(f"{'=' * 90}")
    print(f"\n{'Rank':<5} {'Experiment':<40} {'Best Time':>10} {'Finish%':>8} {'Status':<15}")
    print(f"{'-' * 5} {'-' * 40} {'-' * 10} {'-' * 8} {'-' * 15}")
    for entry in gt["leaderboard"]:
        rank = str(entry["rank"]) if entry["rank"] else "-"
        ft = f"{entry['best_finish_time_s']:.2f}s" if entry["best_finish_time_s"] else "DNF"
        fr = (
            f"{entry['worker_finish_rate'] * 100:.1f}%" if entry.get("worker_finish_rate") else "0%"
        )
        status = entry.get("status") or "unknown"
        print(f"{rank:<5} {entry['exp_id']:<40} {ft:>10} {fr:>8} {status:<15}")

    if gt.get("best_overall"):
        best = gt["best_overall"]
        print(f"\n** BEST: {best['exp_id']} = {best['best_finish_time_s']:.2f}s **")
    print()


def main():
    parser = argparse.ArgumentParser(description="Extract ground truth from analysis JSONs")
    parser.add_argument("--format", choices=["json", "table"], default="table")
    args = parser.parse_args()

    gt = build_ground_truth()

    OUTPUT_PATH.write_text(json.dumps(gt, indent=2, default=str), encoding="utf-8")
    print(f"Ground truth written to {OUTPUT_PATH}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(gt, indent=2, default=str))
    else:
        print_table(gt)
        for exp in gt["experiments"]:
            if exp.get("best_finish_time_s") and exp["best_finish_time_s"] > 0:
                print(f"  {exp['exp_id']}:")
                print(f"    best_finish_time_s = {exp['best_finish_time_s']:.2f}")
                if exp.get("worker_best_finish_time_s"):
                    print(f"    worker_best        = {exp['worker_best_finish_time_s']:.2f}")
                print(f"    worker_finish_rate = {exp.get('worker_finish_rate', 0) * 100:.1f}%")
                print(f"    loss_median        = {exp.get('loss_iqn_median', 'N/A')}")
                print(f"    return_train_last  = {exp.get('return_train_last', 'N/A')}")
                print(
                    f"    trend: loss={exp.get('loss_trend', '?')}, "
                    f"return={exp.get('return_trend', '?')}"
                )


if __name__ == "__main__":
    main()
