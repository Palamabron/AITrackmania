"""Fetch wandb run history and print metrics summary for report. Uses WANDB_API_KEY from .env."""

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

api = wandb.Api()
path = "tmrl/tmrl/SophyResidual_runv23_RUN_L TRAINER"
run = api.run(path)

print("=== RUN INFO ===")
print("path:", run.path)
print("state:", run.state)
print("created:", getattr(run, "created_at", "N/A"))
print("duration:", getattr(run, "summary", {}).get("_runtime", "N/A"))

h = run.history()
n = len(h)
print("\n=== ROWS (steps) ===", n)
print("\n=== COLUMNS (all metrics) ===")
cols = [c for c in h.columns if not c.startswith("_")]
print(cols)


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


print("\n=== METRIC SUMMARIES (for report) ===")
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

# Also print return_train / episode_length by epoch if _step exists
if "_step" in h.columns and "return_train" in h.columns:
    steps = h["_step"].dropna()
    print("\n=== STEPS RANGE ===", int(steps.min()), "-", int(steps.max()))
if "return_train" in h.columns:
    r = h["return_train"].dropna()
    print(
        "return_train: min",
        float(r.min()),
        "max",
        float(r.max()),
        "last",
        float(r.iloc[-1]) if len(r) else "N/A",
    )
if "episode_length_train" in h.columns:
    e = h["episode_length_train"].dropna()
    print(
        "episode_length_train: min",
        float(e.min()),
        "max",
        float(e.max()),
        "last",
        float(e.iloc[-1]) if len(e) else "N/A",
    )
