"""Analyze wandb trainer run: .env for WANDB_API_KEY, fetch run history, report metrics."""

import importlib
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
wandb = importlib.import_module("wandb")

# Run path: entity/project/run_id (no leading slash; "runs" is not part of api path)
RUN_PATH = "tmrl/tmrl/SophyResidual_runv40_RUN_F TRAINER"


def main():
    api = wandb.Api()
    run = api.run(RUN_PATH)

    print("=== RUN INFO ===")
    print("path:", run.path)
    print("state:", run.state)
    print("created:", getattr(run, "created_at", "N/A"))
    print("duration (s):", getattr(run, "summary", {}).get("_runtime", "N/A"))
    config = getattr(run, "config", {}) or {}
    if config:
        print(
            "config (relevant):",
            {
                k: config.get(k)
                for k in [
                    "RESIDUAL_MLP_NUM_BLOCKS",
                    "RESIDUAL_MLP_NUM_BLOCKS_ACTOR",
                    "RESIDUAL_MLP_NUM_BLOCKS_CRITIC",
                ]
                if k in config
            },
        )

    h = run.history()
    n = len(h)
    print("\n=== ROWS (steps) ===", n)
    cols = [c for c in h.columns if not c.startswith("_")]
    print("=== COLUMNS (metrics) ===", cols)

    def summarize(series):
        s = series.dropna()
        if len(s) < 2:
            return None
        first_10 = s.head(max(1, len(s) // 10))
        last_10 = s.tail(max(1, len(s) // 10))
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "first_avg": float(first_10.mean()),
            "last_avg": float(last_10.mean()),
            "trend": "up" if last_10.mean() > first_10.mean() else "down",
        }

    print("\n=== METRIC SUMMARIES ===")
    summaries = {}
    for c in cols:
        try:
            s = h[c]
            summary = summarize(s)
            if summary:
                summaries[c] = summary
        except Exception as e:
            summaries[c] = {"error": str(e)}
    print(json.dumps(summaries, indent=2))

    # Learning signals
    print("\n=== LEARNING SIGNALS ===")
    if "_step" in h.columns:
        steps = h["_step"].dropna()
        print("step range:", int(steps.min()), "-", int(steps.max()))
    for key in ["return_train", "return_test", "episode_length_train", "episode_length_test"]:
        if key in h.columns:
            r = h[key].dropna()
            if len(r) > 0:
                first_avg = float(r.head(max(1, len(r) // 10)).mean())
                last_avg = float(r.tail(max(1, len(r) // 10)).mean())
                trend = "improving" if last_avg > first_avg else "declining"
                print(f"{key}: first_avg={first_avg:.2f}, last_avg={last_avg:.2f} -> {trend}")
    for key in ["losses/loss_actor", "losses/loss_critic"]:
        if key in h.columns:
            r = h[key].dropna()
            if len(r) > 0:
                first_avg = float(r.head(max(1, len(r) // 10)).mean())
                last_avg = float(r.tail(max(1, len(r) // 10)).mean())
                print(f"{key}: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}")

    return run, h, summaries


if __name__ == "__main__":
    main()
