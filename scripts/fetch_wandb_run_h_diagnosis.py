"""
Fetch wandb runs for GnnEffNet_TQC_gSDE_run_H (TRAINER and WORKER).
Load WANDB_API_KEY from .env. Summarize metrics to diagnose model collapse.

Usage:
  python scripts/fetch_wandb_run_h_diagnosis.py

Output:
  - docs/wandb_run_h/diagnosis.json
  - docs/wandb_run_h/history_*.csv (optional)
  - Printed summary (entropy_coef, log_pi, return_test, best_race_progress, etc.)
"""

import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
_env = os.path.join(_repo_root, ".env")
if os.path.isfile(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "WANDB_API_KEY":
                    os.environ["WANDB_API_KEY"] = v
                    break

if _repo_root in sys.path:
    sys.path.remove(_repo_root)
wandb = __import__("wandb")

ENTITY_PROJECT = "tmrl/tmrl"
RUN_BASE = "GnnEffNet_TQC_gSDE_run_H"
RUN_PATHS = [
    f"{ENTITY_PROJECT}/{RUN_BASE} TRAINER",
    f"{ENTITY_PROJECT}/{RUN_BASE} WORKER",
]

# Metrics that indicate collapse (from INVESTIGATION_REPORT_Dv3)
TRAINER_COLLAPSE_METRICS = [
    "entropy_coef",
    "debug/log_pi",
    "debug/log_pi_std",
    "metrics/return_test",
    "metrics/return_train",
    "losses/loss_actor",
    "losses/loss_critic",
    "debug/q_a1",
    "debug/demo_fraction_in_batch",
]
WORKER_COLLAPSE_METRICS = [
    "run/reward",
    "run/best_race_progress",
    "run/episode_time",
]


def summarize_series(series, name=""):
    s = series.dropna()
    if len(s) < 1:
        return None
    n = len(s)
    first_10 = s.head(max(1, n // 10))
    last_10 = s.tail(max(1, n // 10))
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "first_10pct_avg": float(first_10.mean()),
        "last_10pct_avg": float(last_10.mean()),
        "trend": "up" if last_10.mean() > first_10.mean() else "down",
    }


def main():
    api = wandb.Api()
    out_dir = os.path.join(_repo_root, "docs", "wandb_run_h")
    os.makedirs(out_dir, exist_ok=True)
    results = {"runs": {}, "collapse_diagnosis": {}}

    for run_path in RUN_PATHS:
        name = "WORKER" if "WORKER" in run_path else "TRAINER"
        print(f"Fetching {run_path} ...")
        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            results["runs"][name] = {"error": str(e)}
            continue

        hist = run.history(samples=5000)
        results["runs"][name] = {
            "path": run.path,
            "state": run.state,
            "history_rows": len(hist),
        }

        metrics = WORKER_COLLAPSE_METRICS if name == "WORKER" else TRAINER_COLLAPSE_METRICS
        summaries = {}
        for key in metrics:
            if key in hist.columns:
                s = hist[key]
                sum_ = summarize_series(s, key)
                if sum_:
                    summaries[key] = sum_
        results["collapse_diagnosis"][name] = summaries

        # Save raw history for inspection
        csv_path = os.path.join(out_dir, f"history_{name}.csv")
        hist.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path} ({len(hist)} rows)")

    # Print collapse summary
    print("\n=== COLLAPSE DIAGNOSIS (first 10% vs last 10% of steps) ===\n")
    for name in ("TRAINER", "WORKER"):
        if name not in results["collapse_diagnosis"]:
            continue
        print(f"--- {name} ---")
        for k, v in results["collapse_diagnosis"][name].items():
            first = v["first_10pct_avg"]
            last = v["last_10pct_avg"]
            trend = v["trend"]
            print(f"  {k}: first_10%={first:.4f}  last_10%={last:.4f}  trend={trend}")
        print()

    # Entropy collapse check
    if "TRAINER" in results["collapse_diagnosis"]:
        td = results["collapse_diagnosis"]["TRAINER"]
        if "entropy_coef" in td:
            ec = td["entropy_coef"]
            if ec["last_10pct_avg"] < 0.005 and ec["trend"] == "down":
                print(
                    "  [COLLAPSE] entropy_coef collapsed (<0.005) -> policy likely deterministic."
                )
        if "debug/log_pi" in td:
            lp = td["debug/log_pi"]
            if lp["last_10pct_avg"] > -0.01 and lp["trend"] == "up":
                print("  [COLLAPSE] log_pi -> 0 -> policy probability ~1 (deterministic).")

    out_path = os.path.join(out_dir, "diagnosis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Diagnosis written to {out_path}")


if __name__ == "__main__":
    main()
