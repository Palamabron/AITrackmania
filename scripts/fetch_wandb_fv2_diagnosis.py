"""
Fetch WandB runs GnnEffNet_TQC_gSDE_run_Fv2 TRAINER and WORKER.
Load WANDB_API_KEY from repo .env. Print metrics summary and diagnosis.
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
RUN_BASE = "GnnEffNet_TQC_gSDE_run_Fv2"
PATHS = [
    f"{ENTITY_PROJECT}/{RUN_BASE} TRAINER",
    f"{ENTITY_PROJECT}/{RUN_BASE} WORKER",
]


def main():
    api = wandb.Api()
    out = {"runs": {}, "diagnosis": []}

    for path in PATHS:
        name = "TRAINER" if "TRAINER" in path else "WORKER"
        try:
            run = api.run(path)
        except Exception as e:
            out["runs"][name] = {"error": str(e)}
            out["diagnosis"].append(f"{name}: Failed to fetch - {e}")
            print(f"{name}: Failed - {e}")
            continue

        out["runs"][name] = {
            "state": run.state,
            "path": run.path,
            "summary": dict(run.summary._json_dict) if run.summary else {},
        }
        hist = run.history(samples=5000)
        if hist is None or len(hist) == 0:
            out["runs"][name]["history_rows"] = 0
            out["runs"][name]["cols"] = []
            print(f"{name}: no history")
            continue

        out["runs"][name]["history_rows"] = len(hist)
        cols = [c for c in hist.columns if not c.startswith("_")]
        out["runs"][name]["cols"] = cols

        # Key metrics for diagnosis
        step_col = (
            "_step" if "_step" in hist.columns else (hist.columns[0] if len(hist.columns) else None)
        )
        if step_col is not None:
            last_step = hist[step_col].dropna().iloc[-1] if len(hist) else 0
            out["runs"][name]["last_step"] = int(last_step)

        diag_keys = [
            "reward",
            "reward_mean",
            "test_reward",
            "eval/reward_mean",
            "losses/actor",
            "losses/critic",
            "entropy_coef",
            "debug/q_a1",
            "debug/backup",
        ]
        for key in diag_keys:
            for c in cols:
                if key in c or c == key:
                    s = hist[c].dropna()
                    if len(s) > 0:
                        out["runs"][name].setdefault("series", {})[c] = {
                            "first_100_avg": float(s.head(100).mean()),
                            "last_100_avg": float(s.tail(100).mean()),
                            "min": float(s.min()),
                            "max": float(s.max()),
                        }
                    break

        last = out["runs"][name].get("last_step", "?")
        print(f"{name}: state={run.state}, rows={len(hist)}, last_step={last}")

    # Diagnosis text
    if "WORKER" in out["runs"] and "error" not in out["runs"]["WORKER"]:
        wr = out["runs"]["WORKER"]
        series = wr.get("series", {})
        rwd = series.get("reward") or series.get("reward_mean") or series.get("eval/reward_mean")
        if rwd:
            out["diagnosis"].append(
                f"WORKER reward: first_100_avg={rwd.get('first_100_avg')}, "
                f"last_100_avg={rwd.get('last_100_avg')}"
            )
        if wr.get("state") == "crashed":
            out["diagnosis"].append("WORKER run crashed.")
    if "TRAINER" in out["runs"] and "error" not in out["runs"]["TRAINER"]:
        tr = out["runs"]["TRAINER"]
        series = tr.get("series", {})
        if tr.get("state") == "crashed":
            out["diagnosis"].append("TRAINER run crashed.")
        for k in ["losses/actor", "losses/critic"]:
            if k in series:
                avg = series[k].get("last_100_avg")
                out["diagnosis"].append(f"TRAINER {k}: last_100_avg={avg}")

    report_path = os.path.join(_repo_root, "docs", "wandb_fv2_diagnosis.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Report: {report_path}")

    for line in out["diagnosis"]:
        print("  ", line)
    return out


if __name__ == "__main__":
    main()
