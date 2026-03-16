"""
Fetch wandb TRAINER and WORKER runs for GnnEffNet_SophyResidual_TQC_run_Dv3.
Load WANDB_API_KEY from .env. Apply running average (window 20) and sample every 50 steps.
Output: JSON summary + optional CSVs for the pipeline investigation report.
"""

import json
import os
import sys

import pandas as pd

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

RUN_BASE = "GnnEffNet_TQC_gSDE_run_D_fixed_per"
ENTITY_PROJECT = "tmrl/tmrl"
RUN_PATHS = [
    f"{ENTITY_PROJECT}/{RUN_BASE} WORKER",
    f"{ENTITY_PROJECT}/{RUN_BASE} TRAINER",
]
RUN_WINDOW = 20
RUN_STEP_EVERY = 50


def load_env_api():
    """Ensure .env is loaded; return wandb API."""
    return wandb.Api()


def running_avg_downsample(
    df: pd.DataFrame, window: int = RUN_WINDOW, step_every: int = RUN_STEP_EVERY
) -> pd.DataFrame:
    """
    For numeric columns: compute rolling mean (window). Then keep every step_every-th row.
    Preserve _step (or first column) for x-axis; others get rolling mean. NaNs dropped in rolling.
    """
    out = df.copy()
    step_col = "_step" if "_step" in out.columns else out.columns[0]
    numeric = [c for c in out.select_dtypes(include=["number"]).columns if c != step_col]
    for c in numeric:
        if c in out.columns:
            out[f"{c}_smoothed"] = out[c].rolling(window=window, min_periods=1).mean()
    smoothed_cols = [c for c in out.columns if c.endswith("_smoothed")]
    keep = [step_col] + smoothed_cols
    keep = [c for c in keep if c in out.columns]
    out = out[keep].rename(columns={c: c.replace("_smoothed", "") for c in smoothed_cols})
    out = out.iloc[::step_every].reset_index(drop=True)
    return out


def summarize_series(series: pd.Series):
    s = series.dropna()
    if len(s) < 2:
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


def fetch_run(api, path: str):
    try:
        run = api.run(path)
        return run
    except Exception as e:
        print(f"Failed to fetch {path}: {e}", file=sys.stderr)
        return None


def main():
    api = load_env_api()
    out_dir = os.path.join(_repo_root, "docs", "wandb_dv3")
    os.makedirs(out_dir, exist_ok=True)

    results = {"runs": {}, "smoothed_summaries": {}}

    for run_path in RUN_PATHS:
        name = "WORKER" if "WORKER" in run_path else "TRAINER"
        print(f"Fetching {run_path} ...")
        run = fetch_run(api, run_path)
        if run is None:
            results["runs"][name] = {"error": f"Could not fetch run: {run_path}"}
            continue

        hist = run.history(samples=5000)
        results["runs"][name] = {
            "path": run.path,
            "state": run.state,
            "history_rows": len(hist),
            "columns": [c for c in hist.columns if not c.startswith("_")],
        }

        # Running average (window 20) and every 50 steps
        smoothed = running_avg_downsample(hist, window=RUN_WINDOW, step_every=RUN_STEP_EVERY)
        csv_path = os.path.join(
            out_dir, f"history_{name}_smoothed_w{RUN_WINDOW}_every{RUN_STEP_EVERY}.csv"
        )
        smoothed.to_csv(csv_path, index=False)
        print(f"  Smoothed history saved: {csv_path} ({len(smoothed)} rows)")

        summaries = {}
        for c in smoothed.columns:
            if c.startswith("_"):
                continue
            try:
                s = smoothed[c]
                sum_ = summarize_series(s)
                if sum_:
                    summaries[c] = sum_
            except Exception as e:
                summaries[c] = {"error": str(e)}
        results["smoothed_summaries"][name] = summaries

    report_path = os.path.join(out_dir, "dv3_metrics_summary.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {report_path}")

    # Print history sample for report
    print("\n--- TRAINER history (raw) sample ---")
    if "TRAINER" in results["runs"] and "error" not in results["runs"]["TRAINER"]:
        run = fetch_run(api, RUN_PATHS[1])
        if run:
            h = run.history(samples=5000)
            print(h.head(3).to_string())
            print("...")
            print(h.tail(3).to_string())
    print("\n--- WORKER history (raw) sample ---")
    if "WORKER" in results["runs"] and "error" not in results["runs"]["WORKER"]:
        run = fetch_run(api, RUN_PATHS[0])
        if run:
            h = run.history(samples=5000)
            print(h.head(3).to_string())
            print("...")
            print(h.tail(3).to_string())

    return results


if __name__ == "__main__":
    main()
