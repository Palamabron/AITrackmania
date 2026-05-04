#!/usr/bin/env python3
"""Inspect Weights & Biases runs for TMRL IQN training: loss curves, eval vs train,
worker termination breakdown, truncation rate.

Requires: pip install wandb pandas numpy
Auth: export WANDB_API_KEY=... or put it in TmrlData/.env next to configs (many setups
already load it before tmrl starts); this script also loads .env from repo root when present.

Example:
  cd AITrackmania
  export WANDB_API_KEY=...
  tmrl-inspect-wandb --entity tmrl --project tmrl \\
      --name-contains miqncrossing-testv1.5 --history-samples 100000 --api-timeout 180

Suggestions printed with --print-suggestions are heuristics derived from exported metrics —
tune thresholds for your workload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TRAINER_METRICS = [
    "loss/iqn_loss",
    "loss/total_loss",
    "loss/monotonicity_penalty",
    "q/mean_q",
    "q/max_q",
    "debug/td_abs_mean",
    "debug/td_abs_p95",
    "exploration/epsilon",
    "metrics/return_train",
    "metrics/return_test",
    "eval/return_deterministic",
]

DEFAULT_WORKER_METRICS = [
    "run/truncated",
    "run/reward",
    "run/steps",
    "run/finished_track",
    "run/term_reason",
]


def _load_dotenv_optional() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


def _series(df: Any, col: str):
    """Return pandas Series or None."""
    try:
        s = df[col].astype(float)
    except Exception:
        return None
    return s


def describe_series(s: Any, prefix: str = "") -> str:
    if s is None or len(s.dropna()) == 0:
        return f"{prefix}<no data>"
    x = s.dropna()
    return (
        f"{prefix}n={len(x)} min={float(x.min()):.6g} p50={float(x.median()):.6g} "
        f"p95={float(x.quantile(0.95)):.6g} max={float(x.max()):.6g}"
    )


def find_argmax_timeseries(df: Any, col: str, after_step: float = 500.0) -> dict[str, Any] | None:
    s = _series(df, col)
    if s is None or "_step" not in df.columns:
        return None
    sub = df[df["_step"] > after_step]
    ss = _series(sub, col)
    if ss is None or len(ss.dropna()) == 0:
        return None
    idx = ss.idxmax()
    row = df.loc[idx]
    return {"col": col, "_step": float(row["_step"]), "value": float(ss.loc[idx])}


def heuristic_suggestions(payload: dict[str, Any]) -> list[str]:
    """Return bullet strings."""
    suggestions: list[str] = []

    wm = payload.get("worker_summary") or {}
    trunc_rate = wm.get("truncated_rate_mean")
    nprog = wm.get("no_progress_times")
    trunc_c = wm.get("truncated_times")

    tm = payload.get("trainer_phase") or {}

    eps_last = tm.get("explorationEpsilon_last_phase_median")

    tm_full = payload.get("trainer_numeric") or {}
    eval_med = tm_full.get("eval_return_deterministic", {}).get("median")
    train_med = tm_full.get("return_train_median_overall")

    # Worker / rollout
    if isinstance(trunc_rate, (int, float)) and trunc_rate > 0.55:
        suggestions.append(
            "Worker: bardzo częsty `truncated` — ustaw wyżej `environment.rtgym.ep_max_length` "
            "(np. już masz 2k) oraz upewnij się, że worker używa `ep_max_length` z env "
            "(patrz rollout `max_samples_per_episode`)."
        )

    if (
        isinstance(nprog, int)
        and isinstance(trunc_c, int)
        and (nprog + trunc_c) > 0
        and nprog > trunc_c * 1.2
    ):
        suggestions.append(
            "Worker: częstszy jest `no_progress_timeout` niż `truncated` — rozważ łagodniejszą "
            "diagnostykę stagnacji przez dłuższy `reward.min_seconds_before_failure` "
            "(pola `reward.slow_progress_window_seconds` i `reward.min_progress_rate` "
            "są legacy/no-op; po analizie, czy to nie sztuka "
            "'utknął na poboczu')."
        )

    # Trainer exploitation
    if isinstance(eps_last, (int, float)) and eps_last > 0.12:
        suggestions.append(
            "Trainer: median ε w późnej fazie nadal wyższe — rozważ niższą wartość docelową "
            "`algorithm.iqn_epsilon_end` (np. 0.01 lub 0.005) lub wydłużenie "
            "`algorithm.iqn_epsilon_decay_steps`, żeby dłużej eksplorować zanim zbijesz greedy."
        )

    # Train/eval mismatch
    if (
        isinstance(eval_med, (int, float))
        and isinstance(train_med, (int, float))
        and eval_med > train_med * 3
        and train_med > 0
    ):
        suggestions.append(
            "Medianowy eval return znacząco wyższy niż train rollout — sprawdź greedy vs ε, "
            "mix demonstracji (`demo_max_batch_fraction`) i deterministyczny rollout ewalów."
        )

    suggestions.append(
        "Ogólne (IQN stabilność): jeśli wykresy `loss/monotonicity_penalty` budzą niepokój, "
        "spróbuj `algorithm.iqn_monotonicity_lambda` 0.005 zamiast 0.01; jeśli Q rośnie zbyt "
        "agresywnie, obniż `algorithm.iqn_lr` (np. 5e-5→3e-5) albo dopasuj "
        "`algorithm.iqn_soft_target_tau`. Asynchroniczny trening — rozważ obniżenie "
        "`training.max_training_steps_per_environment_step`, żeby trener nie wyprzedzał danych."
    )

    return suggestions


def main() -> None:
    _load_dotenv_optional()
    p = argparse.ArgumentParser(description="W&B IQN / TMRL run inspector")
    p.add_argument("--entity", default="tmrl")
    p.add_argument("--project", default="tmrl")
    p.add_argument("--name-contains", default="", help="Substring on run display name/id")
    p.add_argument("--run-id-exact", default="", help="Exact W&B run id (alternative to substring)")
    p.add_argument("--history-samples", type=int, default=100_000)
    p.add_argument("--api-timeout", type=int, default=120)
    p.add_argument("--early-step-cap", type=float, default=500.0, help="Phase split for loss stats")
    p.add_argument("--print-suggestions", action="store_true")
    p.add_argument("--json-out", default="", help="Optional path to dump summaries as JSON")

    args = p.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set (.env/repo or environment). Abort.")
        raise SystemExit(2)

    import wandb

    api = wandb.Api(timeout=max(19, int(args.api_timeout)))

    if args.run_id_exact:
        trainers = [
            api.run(f"{args.entity}/{args.project}/{args.run_id_exact} TRAINER".strip()),
        ]
        workers = [api.run(f"{args.entity}/{args.project}/{args.run_id_exact} WORKER".strip())]
    elif args.name_contains:
        path = f"{args.entity}/{args.project}"
        filters = {
            "$or": [
                {"display_name": {"$regex": args.name_contains}},
                {"name": {"$regex": args.name_contains}},
            ]
        }
        runs = list(api.runs(path, filters=filters))
        trainers = [r for r in runs if (r.name or "").upper().endswith("TRAINER")]
        workers = [r for r in runs if (r.name or "").upper().endswith("WORKER")]
    else:
        print("Specify --name-contains or --run-id-exact (experiment prefix bez sufiksu).")
        raise SystemExit(2)

    if not trainers:
        print("No TRAINER run found.")
        raise SystemExit(1)

    trainer = trainers[0]
    print(f"TRAINER id={trainer.id} state={trainer.state} url={trainer.url}")
    h = trainer.history(samples=int(args.history_samples))

    warmup = h[h["_step"] <= float(args.early_step_cap)] if "_step" in h.columns else h
    late = h[h["_step"] > float(args.early_step_cap)] if "_step" in h.columns else h

    trainer_payload: dict[str, Any] = {
        "trainer_run_id": trainer.id,
        "rows": len(h),
        "columns": list(h.columns),
    }

    print("\n--- TRAINER: phase split loss (early vs late) ---")
    for metric in ["loss/iqn_loss", "loss/total_loss", "loss/monotonicity_penalty"]:
        we = _series(warmup, metric)
        wl = _series(late, metric)
        print(metric)
        print(" ", describe_series(we, "early "))
        print(" ", describe_series(wl, "late "))
        am_late = find_argmax_timeseries(h, metric, after_step=float(args.early_step_cap))
        if am_late:
            print(f"  max-after-warmup: step={am_late['_step']:.0f} value={am_late['value']:.6g}")

    # Key scalars snapshot
    def last_non_nan(series: Any) -> float | None:
        if series is None:
            return None
        v = series.dropna()
        if len(v) == 0:
            return None
        return float(v.iloc[-1])

    trainer_payload["trainer_numeric"] = {}
    snap_cols = [
        "metrics/return_train",
        "metrics/return_test",
        "eval/return_deterministic",
        "exploration/epsilon",
    ]
    print("\n--- TRAINER: last logged values ---")
    for col in snap_cols:
        if col in h.columns:
            s = _series(h, col)
            lc = last_non_nan(s)
            med = float(s.dropna().median()) if s is not None and len(s.dropna()) else None
            trainer_payload["trainer_numeric"][col.replace("/", "_")] = {"last": lc, "median": med}
            print(f"{col}: last={lc} median={med}")

    if "metrics/return_train" in h.columns:
        s = _series(h, "metrics/return_train")
        if s is not None:
            trainer_payload["trainer_numeric"]["return_train_median_overall"] = float(
                s.dropna().median()
            )

    trainer_payload["trainer_phase"] = {
        "warmupStepsCap": args.early_step_cap,
        "explorationEpsilon_lateMedian": (
            float(_series(late, "exploration/epsilon").dropna().median())
            if _series(late, "exploration/epsilon") is not None
            and len(_series(late, "exploration/epsilon").dropna())
            else None
        ),
    }

    if workers:
        w = workers[0]
        print(f"\nWORKER id={w.id} state={w.state} url={w.url}")
        hw = w.history(samples=50_000)
        trainer_payload["worker_summary"] = {
            "worker_run_id": w.id,
            "rows": len(hw),
            "termination_counts": {},
        }
        print("\n--- WORKER ---")
        if "run/term_reason" in hw.columns:
            vc = hw["run/term_reason"].astype(str).value_counts(dropna=False)
            print(vc.to_string())
            trainer_payload["worker_summary"]["termination_counts"] = vc.to_dict()
        if "run/truncated" in hw.columns:
            tr = hw["run/truncated"].astype(float).dropna()
            if len(tr):
                mr = float(tr.mean())
                trainer_payload["worker_summary"]["truncated_rate_mean"] = mr
                print(f"run/truncated mean (episode logs): {mr:.3f}")
        if "run/steps" in hw.columns:
            st = hw["run/steps"].astype(float).dropna()
            if len(st):
                print(describe_series(st, "run/steps "))

        for col in DEFAULT_WORKER_METRICS:
            if col == "run/term_reason":
                continue
            if col in hw.columns:
                s = _series(hw, col)
                if col == "run/term_reason":
                    continue
                if col == "run/truncated":
                    trainer_payload["worker_summary"]["truncated_times"] = int((s > 0.5).sum())
                trainer_payload.setdefault("worker_last", {})[col] = last_non_nan(s)

        # Prefer episode counts from run/term_reason (same source as heuristic)
        trc = trainer_payload["worker_summary"].get("termination_counts") or {}
        if isinstance(trc, dict) and "no_progress_timeout" in trc:
            trainer_payload["worker_summary"]["no_progress_times"] = int(trc["no_progress_timeout"])
        if isinstance(trc, dict) and "truncated" in trc:
            trainer_payload["worker_summary"]["truncated_times"] = int(trc["truncated"])

    if args.print_suggestions:
        print("\n--- HEURISTIC SUGGESTIONS ---")
        for line in heuristic_suggestions(trainer_payload):
            print(f"- {line}")

    if args.json_out:
        out_path = Path(args.json_out)
        # JSON-serialize only summary bits
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(trainer_payload, f, indent=2, default=str)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
