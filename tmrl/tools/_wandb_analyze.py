"""W&B-based experiment analysis commands: register, status, analyze."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any

import yaml

from tmrl.tools._exp_config_utils import (
    ANALYSIS_DIR,
    CONFIGS_DIR,
    _load_target_time,
    _next_exp_id,
    _retry,
    _safe_float_series,
    _warn,
)
from tmrl.tools._experiment_io import (
    _atomic_write,
)
from tmrl.tools._experiment_io import (
    append_registry as _append_registry,
)
from tmrl.tools._experiment_io import (
    load_dotenv as _load_dotenv,
)
from tmrl.tools._experiment_io import (
    read_registry as _read_registry,
)
from tmrl.tools._experiment_io import (
    update_registry_entry as _update_registry_entry,
)
from tmrl.tools._wandb_snapshot import _find_wandb_run


def cmd_register(args: argparse.Namespace) -> None:
    """Register a new experiment in the registry and write its config YAML.

    Creates a ``planned`` entry in the JSONL registry and a corresponding
    YAML file under ``experiments/configs/``.
    """
    try:
        overrides = json.loads(args.overrides) if args.overrides else {}
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid --overrides JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    exp_id = args.exp_id or _next_exp_id()
    parent = args.parent or "baseline"
    hypothesis = args.hypothesis or ""
    now = datetime.datetime.now(datetime.UTC).isoformat()

    entry: dict[str, Any] = {
        "exp_id": exp_id,
        "parent_exp_id": parent,
        "status": "planned",
        "created_at": now,
        "stopped_at": None,
        "wandb_run_id": None,
        "hypothesis": hypothesis,
        "config_overrides": overrides,
        "summary_metrics": None,
        "stop_reason": None,
    }
    _append_registry(entry)

    config_path = CONFIGS_DIR / f"{exp_id}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overrides, f, sort_keys=False, default_flow_style=False)

    print(f"Registered {exp_id} (parent={parent})")
    print(f"  Config: {config_path}")
    print(f"  Hypothesis: {hypothesis}")
    if overrides:
        print(f"  Overrides: {json.dumps(overrides, indent=2)}")


def cmd_status(args: argparse.Namespace) -> None:
    """Print a tabular status of all registered experiments and highlight target achievement."""
    entries = _read_registry()
    if not entries:
        print("No experiments registered yet.")
        return

    fmt = "{:<8} {:<12} {:<12} {:<20} {}"
    print(fmt.format("EXP_ID", "STATUS", "PARENT", "CREATED", "HYPOTHESIS"))
    print("-" * 100)
    for e in entries:
        created = e.get("created_at", "")[:19]
        hyp = (e.get("hypothesis") or "")[:50]
        print(
            fmt.format(
                e.get("exp_id", "?"),
                e.get("status", "?"),
                e.get("parent_exp_id", "?"),
                created,
                hyp,
            )
        )

    target_time = _load_target_time()
    target_met = [
        e
        for e in entries
        if e.get("status") == "completed"
        and e.get("summary_metrics")
        and isinstance(e["summary_metrics"].get("best_finish_time_s"), (int, float))
        and e["summary_metrics"]["best_finish_time_s"] > 0
        and e["summary_metrics"]["best_finish_time_s"] <= target_time
    ]
    if target_met:
        print(
            f"\n*** TARGET MET: {len(target_met)} experiment(s) "
            f"achieved <={target_time}s finish time ***"
        )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Fetch trainer + worker W&B history and write a full analysis JSON.

    Output is saved under ``experiments/analysis/<exp_id>.json``.  Computes
    per-metric statistics, finish-time extraction, three-phase training trends
    (early/mid/late), and a delta comparison against the baseline experiment if
    available.  Updates the registry entry with summary metrics.
    """
    _load_dotenv()

    exp_id = args.exp_id
    entries = {e.get("exp_id"): e for e in _read_registry()}
    entry = entries.get(exp_id)
    if not entry:
        print(f"ERROR: {exp_id} not found in registry.", file=sys.stderr)
        sys.exit(1)

    wandb_run_id = entry.get("wandb_run_id")
    if not wandb_run_id:
        print(f"ERROR: {exp_id} has no wandb_run_id yet.", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    import wandb

    api = wandb.Api(timeout=120)
    entity = args.entity or "tmrl"
    project = args.project or "tmrl"

    trainer_run = None
    worker_run = None
    for suffix, target in [(" TRAINER", "trainer"), (" WORKER", "worker")]:
        try:
            r = _find_wandb_run(api, entity, project, f"{wandb_run_id}{suffix}")
            if target == "trainer":
                trainer_run = r
            else:
                worker_run = r
        except Exception as exc:
            _warn(f"Could not fetch {target} run for {wandb_run_id}: {exc}")

    if not trainer_run:
        print(f"ERROR: Could not find W&B trainer run for {wandb_run_id}", file=sys.stderr)
        sys.exit(1)

    try:
        h = _retry(
            lambda: trainer_run.history(samples=100_000),
            retries=3,
            base_delay=10.0,
            label="fetch trainer history",
        )
    except Exception as exc:
        print(f"ERROR: Could not fetch trainer history: {exc}", file=sys.stderr)
        sys.exit(1)

    summary: dict[str, Any] = {
        "exp_id": exp_id,
        "wandb_run_id": wandb_run_id,
        "trainer_state": trainer_run.state,
        "total_rows": len(h),
        "columns": list(h.columns),
    }

    wandb_config = dict(trainer_run.config or {})
    if wandb_config:
        summary["full_config"] = wandb_config

    if entry.get("git"):
        summary["git"] = entry["git"]

    def _safe_stats(series_name: str) -> dict[str, Any] | None:
        """Return descriptive stats for *series_name* in *h*, or ``None`` if missing/empty."""
        if series_name not in h.columns:
            return None
        try:
            s = _safe_float_series(h[series_name]).dropna()
        except Exception:
            return None
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
        stats = _safe_stats(m)
        if stats:
            summary["metrics"][m] = stats

    if "eval/finish_time_test_s" in h.columns:
        try:
            ft_series = _safe_float_series(h["eval/finish_time_test_s"]).dropna()
            ft_series = ft_series[ft_series > 0]
            if len(ft_series) > 0:
                summary["best_finish_time_s"] = float(ft_series.min())
                summary["last_finish_time_s"] = float(ft_series.iloc[-1])
                summary["median_finish_time_s"] = float(ft_series.median())
            else:
                summary["best_finish_time_s"] = None
        except Exception as exc:
            _warn(f"Error processing finish time series: {exc}")
            summary["best_finish_time_s"] = None
    else:
        summary["best_finish_time_s"] = None

    if worker_run:
        try:
            hw = _retry(
                lambda: worker_run.history(samples=50_000),
                retries=3,
                base_delay=10.0,
                label="fetch worker history",
            )
            worker_summary: dict[str, Any] = {"worker_state": worker_run.state}
            if "run/term_reason" in hw.columns:
                vc = hw["run/term_reason"].astype(str).value_counts(dropna=False)
                worker_summary["termination_counts"] = vc.to_dict()
            if "run/finished_track" in hw.columns:
                ft = _safe_float_series(hw["run/finished_track"]).dropna()
                worker_summary["finish_rate"] = float(ft.mean()) if len(ft) > 0 else 0.0
            if "run/finish_time" in hw.columns:
                wft = _safe_float_series(hw["run/finish_time"]).dropna()
                wft = wft[wft > 0]
                if len(wft) > 0:
                    w_best = float(wft.min())
                    worker_summary["best_finish_time_s"] = w_best
                    worker_summary["finish_count"] = len(wft)
                    if (
                        summary["best_finish_time_s"] is None
                        or w_best < summary["best_finish_time_s"]
                    ):
                        summary["best_finish_time_s"] = w_best
                    if summary.get("last_finish_time_s") is None:
                        summary["last_finish_time_s"] = float(wft.iloc[-1])
            if "run/steps" in hw.columns:
                st = _safe_float_series(hw["run/steps"]).dropna()
                if len(st) > 0:
                    worker_summary["avg_episode_steps"] = float(st.mean())
            summary["worker"] = worker_summary
        except Exception as exc:
            _warn(f"Error fetching/processing worker history: {exc}")
            summary["worker"] = {"worker_state": "error", "error": str(exc)}

    if len(h) >= 100:
        try:
            n = len(h)
            phases = {
                "early": h.iloc[: n // 3],
                "mid": h.iloc[n // 3 : 2 * n // 3],
                "late": h.iloc[2 * n // 3 :],
            }
            phase_analysis: dict[str, dict[str, Any]] = {}
            for metric_name in ("loss/iqn_loss", "q/max_q", "q/mean_q", "metrics/return_train"):
                if metric_name not in h.columns:
                    continue
                phase_means: dict[str, float] = {}
                for pname, pdf in phases.items():
                    s = _safe_float_series(pdf[metric_name]).dropna()
                    if len(s) > 0:
                        phase_means[pname] = round(float(s.mean()), 4)
                if len(phase_means) >= 2:
                    early_v = phase_means.get("early", 0)
                    late_v = phase_means.get("late", 0)
                    pct = round((late_v - early_v) / abs(early_v) * 100, 1) if early_v != 0 else 0.0
                    if metric_name == "loss/iqn_loss":
                        direction = (
                            "improving"
                            if late_v < early_v * 0.8
                            else "degrading"
                            if late_v > early_v * 1.3
                            else "stable"
                        )
                    elif metric_name == "metrics/return_train":
                        direction = (
                            "improving"
                            if late_v > early_v * 1.3
                            else "degrading"
                            if late_v < early_v * 0.7
                            else "stable"
                        )
                    else:
                        direction = "stable"
                    phase_analysis[metric_name] = {
                        "phases": phase_means,
                        "pct_change": pct,
                        "direction": direction,
                    }
            if phase_analysis:
                summary["training_trends"] = phase_analysis
        except Exception as exc:
            _warn(f"Error in phase analysis: {exc}")

    baseline_analysis = ANALYSIS_DIR / "gtn-baseline.json"
    if baseline_analysis.exists() and exp_id != "gtn-baseline":
        try:
            ba = json.loads(baseline_analysis.read_text(encoding="utf-8"))
            deltas: dict[str, dict[str, Any]] = {}
            for mk in key_metrics:
                bs = ba.get("metrics", {}).get(mk, {})
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
            if deltas:
                summary["vs_baseline"] = deltas
        except Exception as exc:
            _warn(f"Error comparing to baseline: {exc}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{exp_id}.json"
    _atomic_write(out_path, json.dumps(summary, indent=2, default=str))

    try:
        _update_registry_entry(
            exp_id,
            {
                "summary_metrics": {
                    "best_finish_time_s": summary.get("best_finish_time_s"),
                    "last_finish_time_s": summary.get("last_finish_time_s"),
                }
            },
        )
    except Exception as exc:
        _warn(f"Could not update registry with summary metrics: {exc}")

    print(f"Analysis saved to {out_path}")
    if summary.get("best_finish_time_s"):
        target_time = _load_target_time()
        print(f"  Best finish time: {summary['best_finish_time_s']:.2f}s (target: {target_time}s)")
