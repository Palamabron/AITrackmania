"""Experiment manager CLI for the agentic tuning framework.

Usage:
    python -m tmrl.tools.experiment_manager register \
        --parent baseline --hypothesis "..." --overrides '{...}'
    python -m tmrl.tools.experiment_manager status
    python -m tmrl.tools.experiment_manager analyze --exp-id EXP001
    python -m tmrl.tools.experiment_manager compare --exp-a EXP001 --exp-b EXP002
    python -m tmrl.tools.experiment_manager suggest
    python -m tmrl.tools.experiment_manager snapshot --exp-id EXP001
    python -m tmrl.tools.experiment_manager briefing [--json]
    python -m tmrl.tools.experiment_manager leaderboard [--json]
    python -m tmrl.tools.experiment_manager reset incomplete|all --yes [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import shutil
import sys
import time
from typing import Any

import yaml

from tmrl.tools._experiment_io import (
    EXPERIMENTS_DIR,
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
from tmrl.tools._experiment_io import (
    write_registry as _write_registry,
)

CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
EXPERIMENTS_LOGS_DIR = EXPERIMENTS_DIR / "logs"
AGENT_CONTEXT_PATH = EXPERIMENTS_DIR / "_agent_context.json"
BASELINE_PATH = EXPERIMENTS_DIR / "baseline.yaml"
DECISIONS_PATH = EXPERIMENTS_DIR / "decisions.md"
SEARCH_SPACE_PATH = EXPERIMENTS_DIR / "search_space.yaml"
ORCHESTRATOR_CONFIG_PATH = EXPERIMENTS_DIR / "orchestrator_config.yaml"


def _warn(msg: str) -> None:
    print(f"[experiment_manager WARNING] {msg}", file=sys.stderr, flush=True)


def _retry(fn, *, retries: int = 3, base_delay: float = 5.0, label: str = ""):
    """Call *fn()* with exponential-backoff retries on any exception.

    Returns the result on success, or re-raises on final failure.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                _warn(
                    f"{label or 'retry'}: attempt {attempt}/{retries} failed "
                    f"({exc!r}), retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _safe_float_series(series):
    """Convert a pandas series to float, coercing non-numeric values to NaN."""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").astype(float)


def _load_target_time() -> float:
    """Read target_finish_time_s from orchestrator_config.yaml."""
    if ORCHESTRATOR_CONFIG_PATH.exists():
        with ORCHESTRATOR_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("target_finish_time_s", 36.0))
    return 36.0


def _orch_defaults() -> tuple[str, str]:
    """Read default entity/project from orchestrator_config.yaml."""
    if ORCHESTRATOR_CONFIG_PATH.exists():
        with ORCHESTRATOR_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("wandb_entity", "tmrl"), cfg.get("wandb_project", "tmrl")
    return "tmrl", "tmrl"


def _next_exp_id() -> str:
    entries = _read_registry()
    if not entries:
        return "EXP001"
    max_num = 0
    for e in entries:
        eid = e.get("exp_id", "")
        # Support both EXP### and custom names -- only auto-increment numeric
        if eid.startswith("EXP") and eid[3:].isdigit():
            max_num = max(max_num, int(eid[3:]))
    return f"EXP{max_num + 1:03d}"


def _load_baseline() -> dict[str, Any]:
    with BASELINE_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_diff(base: dict, other: dict, prefix: str = "") -> dict[str, tuple[Any, Any]]:
    """Return {dotted.key: (base_val, other_val)} for all leaves that differ."""
    diffs: dict[str, tuple[Any, Any]] = {}
    all_keys = set(base) | set(other)
    for k in sorted(all_keys):
        path = f"{prefix}.{k}" if prefix else k
        bv = base.get(k)
        ov = other.get(k)
        if isinstance(bv, dict) and isinstance(ov, dict):
            diffs.update(_deep_diff(bv, ov, path))
        elif bv != ov:
            diffs[path] = (bv, ov)
    return diffs


def _build_full_config(exp_id: str) -> dict[str, Any]:
    """Reconstruct the full config for an experiment by walking the parent chain."""
    entries = {e["exp_id"]: e for e in _read_registry()}
    overrides_chain: list[dict] = []
    current = exp_id
    while current and current != "baseline":
        entry = entries.get(current)
        if not entry:
            break
        overrides_chain.append(entry.get("config_overrides", {}))
        current = entry.get("parent_exp_id", "baseline")
    base = _load_baseline()
    for ovr in reversed(overrides_chain):
        base = _deep_merge(base, ovr)
    return base


def _build_config_overrides_json(overrides: dict[str, Any]) -> str:
    """Convert nested override dict into TMRL_CONFIG_OVERRIDES JSON."""
    return json.dumps(overrides, default=str)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_register(args: argparse.Namespace) -> None:
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

    # Training phase analysis (early / mid / late)
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

    # Comparison vs baseline
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


def cmd_compare(args: argparse.Namespace) -> None:
    exp_a, exp_b = args.exp_a, args.exp_b

    cfg_a = _build_full_config(exp_a)
    cfg_b = _build_full_config(exp_b)
    config_diff = _deep_diff(cfg_a, cfg_b)

    analysis_a_path = ANALYSIS_DIR / f"{exp_a}.json"
    analysis_b_path = ANALYSIS_DIR / f"{exp_b}.json"
    try:
        analysis_a = (
            json.loads(analysis_a_path.read_text(encoding="utf-8"))
            if analysis_a_path.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError) as exc:
        _warn(f"Could not load analysis for {exp_a}: {exc}")
        analysis_a = {}
    try:
        analysis_b = (
            json.loads(analysis_b_path.read_text(encoding="utf-8"))
            if analysis_b_path.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError) as exc:
        _warn(f"Could not load analysis for {exp_b}: {exc}")
        analysis_b = {}

    result: dict[str, Any] = {
        "exp_a": exp_a,
        "exp_b": exp_b,
        "config_diff": {k: {"a": v[0], "b": v[1]} for k, v in config_diff.items()},
        "metrics_comparison": {},
    }

    metrics_a = analysis_a.get("metrics", {})
    metrics_b = analysis_b.get("metrics", {})
    all_metric_keys = set(metrics_a) | set(metrics_b)
    for mk in sorted(all_metric_keys):
        sa = metrics_a.get(mk, {})
        sb = metrics_b.get(mk, {})
        result["metrics_comparison"][mk] = {
            "a_last": sa.get("last"),
            "b_last": sb.get("last"),
            "a_mean": sa.get("mean"),
            "b_mean": sb.get("mean"),
        }

    ft_a = analysis_a.get("best_finish_time_s")
    ft_b = analysis_b.get("best_finish_time_s")
    result["finish_time"] = {"a": ft_a, "b": ft_b}
    if ft_a is not None and ft_b is not None:
        result["finish_time"]["better"] = exp_a if ft_a < ft_b else exp_b

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"=== Comparison: {exp_a} vs {exp_b} ===\n")
        if config_diff:
            print("Config differences:")
            for k, (va, vb) in config_diff.items():
                print(f"  {k}: {va} -> {vb}")
        else:
            print("Config: identical")
        print()
        if ft_a is not None or ft_b is not None:
            print(f"Best finish time: {exp_a}={ft_a}s  {exp_b}={ft_b}s")
            if ft_a is not None and ft_b is not None:
                winner = exp_a if ft_a < ft_b else exp_b
                print(f"  Winner: {winner}")
        print()
        for mk in sorted(all_metric_keys):
            sa = metrics_a.get(mk, {})
            sb = metrics_b.get(mk, {})
            a_last = sa.get("last", "N/A")
            b_last = sb.get("last", "N/A")
            print(f"  {mk}: {exp_a}={a_last}  {exp_b}={b_last}")


def cmd_suggest(args: argparse.Namespace) -> None:
    entries = _read_registry()
    completed = [e for e in entries if e.get("status") in ("completed", "stopped_early")]

    if not completed:
        print("No completed experiments to analyze. Suggestions based on search space only.")
        print("\nStart with the baseline and try:")
        print("  1. Lower iqn_lr (3e-5) if loss is unstable")
        print("  2. Increase batch_size (1024) if memory allows")
        print("  3. Increase n_steps (5) for longer horizon")
        return

    suggestions: list[str] = []

    analyses: list[dict[str, Any]] = []
    for e in completed:
        ap = ANALYSIS_DIR / f"{e['exp_id']}.json"
        if ap.exists():
            analyses.append(json.loads(ap.read_text()))

    finish_times = [
        (a["exp_id"], a["best_finish_time_s"])
        for a in analyses
        if a.get("best_finish_time_s") is not None
    ]
    if finish_times:
        finish_times.sort(key=lambda x: x[1])
        best_id, best_time = finish_times[0]
        target_time = _load_target_time()
        suggestions.append(
            f"Best finish time so far: {best_time:.2f}s ({best_id}). Target: {target_time}s. "
            f"Delta: {best_time - target_time:.2f}s."
        )

    for a in analyses:
        m = a.get("metrics", {})
        iqn_loss = m.get("loss/iqn_loss", {})
        if iqn_loss.get("p95") and iqn_loss["p95"] > 10 * (iqn_loss.get("median") or 1):
            suggestions.append(
                f"{a['exp_id']}: IQN loss has high spikes (p95={iqn_loss['p95']:.4g} vs "
                f"median={iqn_loss.get('median', '?'):.4g}). Consider lower iqn_lr or "
                f"tighter iqn_grad_clip."
            )
        q_stats = m.get("q/mean_q", {})
        if q_stats.get("max") and abs(q_stats["max"]) > 100:
            suggestions.append(
                f"{a['exp_id']}: Q-values grew large (max={q_stats['max']:.2f}). "
                f"Consider lower iqn_lr, stricter backup_clip_range, or reward_normalize_scale."
            )

    overrides_tried: dict[str, list[tuple[str, Any, float | None]]] = {}
    for e in completed:
        ap = ANALYSIS_DIR / f"{e['exp_id']}.json"
        ft = None
        if ap.exists():
            ad = json.loads(ap.read_text())
            ft = ad.get("best_finish_time_s")
        for key, val in _flatten_dict(e.get("config_overrides", {})):
            overrides_tried.setdefault(key, []).append((e["exp_id"], val, ft))

    for key, trials in overrides_tried.items():
        with_ft = [(eid, val, ft) for eid, val, ft in trials if ft is not None]
        if len(with_ft) >= 2:
            with_ft.sort(key=lambda x: x[2])
            best = with_ft[0]
            worst = with_ft[-1]
            suggestions.append(
                f"Parameter '{key}': best result with value={best[1]} ({best[0]}, "
                f"{best[2]:.1f}s), worst with value={worst[1]} ({worst[0]}, {worst[2]:.1f}s)."
            )

    if not suggestions:
        suggestions.append("Not enough data for cross-experiment analysis yet.")

    print("=== Experiment Suggestions ===\n")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, path))
        else:
            items.append((path, v))
    return items


def _find_wandb_run(api, entity: str, project: str, run_id_with_suffix: str):
    """Find a W&B run by ID, falling back to a filter search on display_name.

    Direct ``api.run()`` requires the exact run ID.  If that fails (e.g. the
    project was wrong, the ID was sanitised, or the run was created with a
    generated ID), search across entity projects for a matching display name.
    """
    # 1. Direct lookup (fast path)
    try:
        return _retry(
            lambda: api.run(f"{entity}/{project}/{run_id_with_suffix}"),
            retries=2,
            base_delay=5.0,
            label=f"direct lookup {run_id_with_suffix}",
        )
    except Exception:
        pass

    # 2. Fallback: search by display_name across the expected project
    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": run_id_with_suffix},
            per_page=1,
        )
        for r in runs:
            _warn(f"Run '{run_id_with_suffix}' found via display_name search (actual id: {r.id})")
            return r
    except Exception:
        pass

    # 3. Fallback: search ALL projects under the entity
    try:
        all_projects = [p.name for p in api.projects(entity)]
    except Exception:
        all_projects = []
    for proj in all_projects:
        if proj == project:
            continue
        try:
            r = api.run(f"{entity}/{proj}/{run_id_with_suffix}")
            _warn(
                f"Run '{run_id_with_suffix}' found in project '{proj}' "
                f"(orchestrator expected '{project}'). Fix orchestrator_config.yaml!"
            )
            return r
        except Exception:
            continue

    raise LookupError(
        f"Could not find run '{run_id_with_suffix}' in {entity}/{project} "
        f"(also searched {len(all_projects)} other project(s))"
    )


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Pull current W&B metrics for a running experiment (used by orchestrator)."""
    _load_dotenv()

    exp_id = args.exp_id
    entries = {e.get("exp_id"): e for e in _read_registry()}
    entry = entries.get(exp_id)
    if not entry:
        print(json.dumps({"error": f"{exp_id} not found"}))
        sys.exit(1)

    wandb_run_id = entry.get("wandb_run_id")
    if not wandb_run_id:
        print(json.dumps({"error": f"{exp_id} has no wandb_run_id"}))
        sys.exit(1)

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print(json.dumps({"error": "WANDB_API_KEY not set"}))
        sys.exit(1)

    import wandb

    try:
        api = wandb.Api(timeout=60)
    except Exception as exc:
        print(json.dumps({"error": f"wandb.Api init failed: {exc}"}))
        sys.exit(1)

    entity = args.entity or "tmrl"
    project = args.project or "tmrl"

    snapshot: dict[str, Any] = {
        "exp_id": exp_id,
        "wandb_run_id": wandb_run_id,
        "hypothesis": entry.get("hypothesis", ""),
        "config_overrides": entry.get("config_overrides", {}),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    # --- Trainer data ---
    try:
        trainer = _find_wandb_run(api, entity, project, f"{wandb_run_id} TRAINER")
        snapshot["trainer_state"] = trainer.state

        h = _retry(
            lambda: trainer.history(samples=10_000),
            retries=2,
            base_delay=10.0,
            label="snapshot: fetch trainer history",
        )
        snapshot["total_steps"] = len(h)

        recent = h.tail(min(100, len(h)))

        def _recent_stats(col: str) -> dict[str, Any] | None:
            if col not in recent.columns:
                return None
            try:
                s = _safe_float_series(recent[col]).dropna()
            except Exception:
                return None
            if len(s) == 0:
                return None
            return {
                "last": float(s.iloc[-1]),
                "mean": float(s.mean()),
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "p95": float(s.quantile(0.95)),
            }

        key_cols = [
            "loss/iqn_loss",
            "q/mean_q",
            "q/max_q",
            "q/min_q",
            "exploration/epsilon",
            "metrics/return_train",
            "eval/return_deterministic",
            "eval/finish_time_test_s",
            "debug/td_abs_mean",
            "debug/grad_norm",
            "debug/grad_norm_pre_clip",
            "buffer/memory_len",
        ]
        snapshot["recent_metrics"] = {}
        for col in key_cols:
            st = _recent_stats(col)
            if st:
                snapshot["recent_metrics"][col] = st

        if "eval/finish_time_test_s" in h.columns:
            try:
                ft = _safe_float_series(h["eval/finish_time_test_s"]).dropna()
                ft = ft[ft > 0]
                if len(ft) > 0:
                    snapshot["best_finish_time_s"] = float(ft.min())
                    snapshot["last_finish_time_s"] = float(ft.iloc[-1])
            except Exception as exc:
                _warn(f"Snapshot: finish_time series error: {exc}")

        if len(h) >= 200:
            try:
                n = len(h)
                early_df = h.iloc[: n // 3]
                late_df = h.tail(min(100, n // 3))
                trends: dict[str, str] = {}
                for col in ("loss/iqn_loss", "q/max_q", "metrics/return_train"):
                    if col not in h.columns:
                        continue
                    e_s = _safe_float_series(early_df[col]).dropna()
                    l_s = _safe_float_series(late_df[col]).dropna()
                    if len(e_s) == 0 or len(l_s) == 0:
                        continue
                    e_mean, l_mean = float(e_s.mean()), float(l_s.mean())
                    if e_mean == 0:
                        continue
                    ratio = l_mean / e_mean
                    if col == "loss/iqn_loss":
                        trends[col] = (
                            "improving" if ratio < 0.8 else "degrading" if ratio > 1.3 else "stable"
                        )
                    elif col == "metrics/return_train":
                        trends[col] = (
                            "improving" if ratio > 1.3 else "degrading" if ratio < 0.7 else "stable"
                        )
                    else:
                        trends[col] = "stable" if 0.8 <= ratio <= 1.2 else "changing"
                if trends:
                    snapshot["trends"] = trends
            except Exception as exc:
                _warn(f"Snapshot: trend analysis error: {exc}")

    except Exception as exc:
        snapshot["error"] = f"Trainer fetch failed after retries: {exc}"
        snapshot["trainer_state"] = "unreachable"

    # --- Worker data ---
    try:
        worker = _find_wandb_run(api, entity, project, f"{wandb_run_id} WORKER")
        snapshot["worker_state"] = worker.state
        hw = _retry(
            lambda: worker.history(samples=10_000),
            retries=2,
            base_delay=10.0,
            label="snapshot: fetch worker history",
        )
        if "run/finish_time" in hw.columns:
            wft = _safe_float_series(hw["run/finish_time"]).dropna()
            wft = wft[wft > 0]
            if len(wft) > 0:
                w_best = float(wft.min())
                snapshot["worker_best_finish_time_s"] = w_best
                snapshot["worker_finish_count"] = len(wft)
                t_best = snapshot.get("best_finish_time_s")
                if t_best is None or w_best < t_best:
                    snapshot["best_finish_time_s"] = w_best
                if "last_finish_time_s" not in snapshot:
                    snapshot["last_finish_time_s"] = float(wft.iloc[-1])
    except Exception as exc:
        snapshot["worker_state"] = "unknown"
        _warn(f"Snapshot: worker fetch failed: {exc}")

    print(json.dumps(snapshot, indent=2, default=str))


def _compute_briefing() -> dict[str, Any]:
    """Build comprehensive context for experiment proposal agents.

    Returns a dict with leaderboard, parameter effects, search space coverage,
    failure patterns, and actionable insights.  Called by ``cmd_briefing``
    (CLI) and importable by the orchestrator.
    """
    entries = _read_registry()

    target_time = _load_target_time()

    briefing: dict[str, Any] = {
        "target_finish_time_s": target_time,
        "total_experiments": len(entries),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    by_status: dict[str, int] = {}
    for e in entries:
        s = e.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    briefing["status_counts"] = by_status

    # ------------------------------------------------------------------
    # Load all saved analyses
    # ------------------------------------------------------------------
    analyses: dict[str, dict[str, Any]] = {}
    for e in entries:
        ap = ANALYSIS_DIR / f"{e['exp_id']}.json"
        if ap.exists():
            with contextlib.suppress(Exception):
                analyses[e["exp_id"]] = json.loads(ap.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Leaderboard (best finish time first, DNFs last)
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for e in entries:
        a = analyses.get(e["exp_id"], {})
        ft = a.get("best_finish_time_s")
        sm = e.get("summary_metrics") or {}
        if ft is None:
            ft = sm.get("best_finish_time_s")
        worker_fc = a.get("worker", {}).get("finish_count") or a.get("worker", {}).get(
            "finish_rate"
        )
        loss_med = a.get("metrics", {}).get("loss/iqn_loss", {}).get("median")
        ret_last = a.get("metrics", {}).get("metrics/return_train", {}).get("last")
        trends = a.get("training_trends", {})
        loss_dir = (
            trends.get("loss/iqn_loss", {}).get("direction")
            if isinstance(trends.get("loss/iqn_loss"), dict)
            else None
        )
        ret_dir = (
            trends.get("metrics/return_train", {}).get("direction")
            if isinstance(trends.get("metrics/return_train"), dict)
            else None
        )
        rows.append(
            {
                "exp_id": e["exp_id"],
                "status": e.get("status"),
                "best_finish_time_s": ft if ft and ft > 0 else None,
                "overrides": e.get("config_overrides", {}),
                "hypothesis": e.get("hypothesis", ""),
                "stop_reason": e.get("stop_reason"),
                "worker_finish_count": worker_fc,
                "loss_median": loss_med,
                "return_last": ret_last,
                "loss_trend": loss_dir,
                "return_trend": ret_dir,
            }
        )

    with_ft = sorted(
        [r for r in rows if r["best_finish_time_s"]], key=lambda r: r["best_finish_time_s"]
    )
    without_ft = [r for r in rows if not r["best_finish_time_s"]]
    briefing["leaderboard"] = with_ft + without_ft

    if with_ft:
        briefing["best_experiment"] = with_ft[0]
        briefing["gap_to_target_s"] = round(with_ft[0]["best_finish_time_s"] - target_time, 2)

    # ------------------------------------------------------------------
    # Parameter effect analysis
    # ------------------------------------------------------------------
    param_effects: dict[str, list[dict[str, Any]]] = {}
    baseline_analysis = analyses.get("gtn-baseline", {})
    baseline_ft = baseline_analysis.get("best_finish_time_s")
    baseline_loss = baseline_analysis.get("metrics", {}).get("loss/iqn_loss", {}).get("median")
    baseline_ret = baseline_analysis.get("metrics", {}).get("metrics/return_train", {}).get("last")

    for e in entries:
        if e.get("status") in ("planned", "running"):
            continue
        a_opt = analyses.get(e["exp_id"])
        if not a_opt:
            continue
        a = a_opt
        ft = a.get("best_finish_time_s")
        loss_m = a.get("metrics", {}).get("loss/iqn_loss", {}).get("median")
        ret_l = a.get("metrics", {}).get("metrics/return_train", {}).get("last")
        fr = a.get("worker", {}).get("finish_rate")

        for dotted_key, val in _flatten_dict(e.get("config_overrides", {})):
            effect: dict[str, Any] = {
                "exp_id": e["exp_id"],
                "value": val,
                "best_finish_time_s": ft if ft and ft > 0 else None,
                "loss_median": loss_m,
                "return_last": ret_l,
                "finish_rate": fr,
                "status": e["status"],
            }
            if baseline_ft and ft and ft > 0:
                effect["ft_delta_vs_baseline"] = round(ft - baseline_ft, 2)
            if baseline_loss and loss_m:
                effect["loss_delta_vs_baseline"] = round(loss_m - baseline_loss, 2)
            if baseline_ret and ret_l:
                effect["return_delta_vs_baseline"] = round(ret_l - baseline_ret, 2)
            param_effects.setdefault(dotted_key, []).append(effect)

    briefing["parameter_effects"] = param_effects

    # ------------------------------------------------------------------
    # Search space coverage
    # ------------------------------------------------------------------
    search_space: dict[str, Any] = {}
    if SEARCH_SPACE_PATH.exists():
        with SEARCH_SPACE_PATH.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                search_space = loaded

    all_ss_params: dict[str, dict[str, Any]] = {}
    for category, params in search_space.items():
        if not isinstance(params, dict):
            continue
        for param_key, param_def in params.items():
            if not isinstance(param_def, dict):
                continue
            all_ss_params[param_key] = {"category": category, **param_def}

    tried_params: dict[str, list[tuple[str, Any]]] = {}
    for e in entries:
        for dotted_key, val in _flatten_dict(e.get("config_overrides", {})):
            tried_params.setdefault(dotted_key, []).append((e["exp_id"], val))

    untried: list[dict[str, Any]] = []
    tried_summary: list[dict[str, Any]] = []
    for param_key, param_def in all_ss_params.items():
        trials = tried_params.get(param_key, [])
        if not trials:
            untried.append(
                {
                    "param": param_key,
                    "category": param_def.get("category"),
                    "baseline": param_def.get("baseline"),
                    "range": param_def.get("range"),
                    "notes": param_def.get("notes", ""),
                }
            )
        else:
            tried_summary.append(
                {
                    "param": param_key,
                    "baseline": param_def.get("baseline"),
                    "range": param_def.get("range"),
                    "tried_values": [{"exp_id": eid, "value": v} for eid, v in trials],
                }
            )

    briefing["search_space_coverage"] = {
        "total_params": len(all_ss_params),
        "tried_count": len(tried_summary),
        "untried_count": len(untried),
        "tried": tried_summary,
        "untried": untried,
    }

    # ------------------------------------------------------------------
    # Failure patterns
    # ------------------------------------------------------------------
    failures = [e for e in entries if e.get("status") == "failed"]
    failure_reasons: dict[str, int] = {}
    for e in failures:
        reason = (e.get("stop_reason") or "unknown").lower()
        if "stuck" in reason or "hang" in reason:
            key = "trainer_stuck"
        elif "server died" in reason or "server" in reason:
            key = "server_crash"
        elif "process crash" in reason:
            key = "process_crash"
        else:
            key = (e.get("stop_reason") or "unknown")[:60]
        failure_reasons[key] = failure_reasons.get(key, 0) + 1

    briefing["failure_patterns"] = {
        "total_failures": len(failures),
        "reasons": failure_reasons,
        "failed_overrides": [
            {"exp_id": e["exp_id"], "overrides": e.get("config_overrides", {})}
            for e in failures
            if e.get("config_overrides")
        ],
    }

    # ------------------------------------------------------------------
    # Cross-experiment insights
    # ------------------------------------------------------------------
    insights: list[str] = []

    # Gradient saturation
    grad_sat = 0
    for a in analyses.values():
        g = a.get("metrics", {}).get("debug/grad_norm", {})
        if g.get("median") and g.get("max") and g["median"] >= g["max"] * 0.99:
            grad_sat += 1
    if grad_sat and grad_sat > len(analyses) * 0.5:
        insights.append(
            f"Gradient clipping saturating in {grad_sat}/{len(analyses)} experiments. "
            f"Consider reducing iqn_grad_clip or lowering learning rate."
        )

    # Loss growing during training
    loss_growing = 0
    for a in analyses.values():
        lo = a.get("metrics", {}).get("loss/iqn_loss", {})
        if lo.get("last") and lo.get("mean") and lo["last"] > lo["mean"] * 1.5:
            loss_growing += 1
    if loss_growing and loss_growing > len(analyses) * 0.5:
        insights.append(
            f"Loss growing during training in {loss_growing}/{len(analyses)} experiments. "
            f"Training may be diverging -- try lower lr, tighter clipping, or smaller batch."
        )

    # Direction recommendation
    if with_ft:
        gap = with_ft[0]["best_finish_time_s"] - target_time
        best_cfg = with_ft[0]["overrides"]
        if gap > 30:
            insights.append(
                f"Large gap to target ({gap:.1f}s). Focus on fundamental changes: "
                f"reward shaping, exploration schedule, or architecture."
            )
        elif gap > 10:
            insights.append(
                f"Moderate gap ({gap:.1f}s). Fine-tune the best config ({with_ft[0]['exp_id']}): "
                f"nearby lr values, epsilon schedule adjustments, reward weights."
            )
        else:
            insights.append(
                f"Close to target ({gap:.1f}s)! Careful refinement of {with_ft[0]['exp_id']}: "
                f"micro-adjust lr, tau, or extend training duration."
            )
        if best_cfg:
            insights.append(f"Best config overrides so far: {json.dumps(best_cfg)}")
    else:
        insights.append(
            "No experiment has finished the track yet. Prioritise: check reward signal is "
            "reachable, ensure adequate exploration, verify environment connectivity."
        )

    # Suggest combining best single-param results
    if len(param_effects) >= 2:
        best_per_param: list[tuple[str, Any, float]] = []
        for pk, effect_trials in param_effects.items():
            completed_trials = [
                t
                for t in effect_trials
                if t.get("best_finish_time_s") and t["status"] in ("completed", "stopped_early")
            ]
            if completed_trials:
                best_t = min(completed_trials, key=lambda t: t["best_finish_time_s"])
                best_per_param.append((pk, best_t["value"], best_t["best_finish_time_s"]))
        if len(best_per_param) >= 2:
            best_per_param.sort(key=lambda x: x[2])
            top2 = best_per_param[:2]
            insights.append(
                f"Consider combining best single-param results: "
                f"{top2[0][0]}={top2[0][1]} ({top2[0][2]:.1f}s) + "
                f"{top2[1][0]}={top2[1][1]} ({top2[1][2]:.1f}s)."
            )

    briefing["insights"] = insights
    return briefing


def cmd_briefing(args: argparse.Namespace) -> None:
    """Generate comprehensive context for experiment proposal agents."""
    briefing = _compute_briefing()

    if args.json:
        print(json.dumps(briefing, indent=2, default=str))
    else:
        _print_briefing_text(briefing)


def _print_briefing_text(b: dict[str, Any]) -> None:
    print("=" * 70)
    print("EXPERIMENT BRIEFING")
    print(
        f"Target: {b['target_finish_time_s']}s | "
        f"Experiments: {b['total_experiments']} | "
        f"Status: {b.get('status_counts', {})}"
    )
    print("=" * 70)

    print("\n--- LEADERBOARD ---")
    for i, r in enumerate(b.get("leaderboard", [])[:10], 1):
        ft = r.get("best_finish_time_s")
        ft_s = f"{ft:.2f}s" if ft else "DNF"
        extra = []
        if r.get("loss_median") is not None:
            extra.append(f"loss_med={r['loss_median']:.1f}")
        if r.get("return_last") is not None:
            extra.append(f"ret={r['return_last']:.0f}")
        if r.get("loss_trend"):
            extra.append(f"loss:{r['loss_trend']}")
        if r.get("return_trend"):
            extra.append(f"ret:{r['return_trend']}")
        ex = f" ({', '.join(extra)})" if extra else ""
        print(f"  {i}. {r['exp_id']}: {ft_s} [{r['status']}]{ex}")

    if b.get("best_experiment"):
        be = b["best_experiment"]
        print(
            f"\n  ** Best: {be['exp_id']} = {be['best_finish_time_s']:.2f}s "
            f"(gap to target: {b.get('gap_to_target_s', 0):.2f}s)"
        )

    cov = b.get("search_space_coverage", {})
    print(
        f"\n--- SEARCH SPACE COVERAGE "
        f"({cov.get('tried_count', 0)}/{cov.get('total_params', 0)} params tried) ---"
    )
    for p in cov.get("untried", [])[:10]:
        notes = (p.get("notes") or "")[:60]
        print(
            f"  UNTRIED: {p['param']}  baseline={p.get('baseline')}  "
            f"range={p.get('range')}  {notes}"
        )

    pe = b.get("parameter_effects", {})
    if pe:
        print("\n--- PARAMETER EFFECTS ---")
        for param, trials in pe.items():
            for t in trials:
                ft_s = f"{t['best_finish_time_s']:.2f}s" if t.get("best_finish_time_s") else "DNF"
                deltas = []
                if t.get("ft_delta_vs_baseline") is not None:
                    deltas.append(f"ft_delta={t['ft_delta_vs_baseline']:+.1f}s")
                if t.get("loss_delta_vs_baseline") is not None:
                    deltas.append(f"loss_delta={t['loss_delta_vs_baseline']:+.1f}")
                d_s = f" [{', '.join(deltas)}]" if deltas else ""
                print(f"  {param}={t['value']}: {ft_s} ({t['status']}){d_s}")

    fp = b.get("failure_patterns", {})
    if fp.get("total_failures"):
        print(f"\n--- FAILURES ({fp['total_failures']}) ---")
        for reason, count in fp.get("reasons", {}).items():
            print(f"  {reason}: {count}x")

    ins = b.get("insights", [])
    if ins:
        print("\n--- INSIGHTS & RECOMMENDATIONS ---")
        for i, txt in enumerate(ins, 1):
            print(f"  {i}. {txt}")
    print()


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Rank experiments by finish time and key metrics."""
    entries = _read_registry()
    analyses: dict[str, dict[str, Any]] = {}
    for e in entries:
        ap = ANALYSIS_DIR / f"{e['exp_id']}.json"
        if ap.exists():
            with contextlib.suppress(Exception):
                analyses[e["exp_id"]] = json.loads(ap.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for e in entries:
        a = analyses.get(e["exp_id"], {})
        ft = a.get("best_finish_time_s")
        sm = e.get("summary_metrics") or {}
        if ft is None:
            ft = sm.get("best_finish_time_s")
        rows.append(
            {
                "exp_id": e["exp_id"],
                "status": e.get("status"),
                "best_ft": ft if ft and ft > 0 else None,
                "loss_median": a.get("metrics", {}).get("loss/iqn_loss", {}).get("median"),
                "return_last": a.get("metrics", {}).get("metrics/return_train", {}).get("last"),
                "finish_rate": a.get("worker", {}).get("finish_rate"),
                "overrides": e.get("config_overrides", {}),
            }
        )

    with_ft = sorted([r for r in rows if r["best_ft"]], key=lambda r: r["best_ft"])
    without_ft = [r for r in rows if not r["best_ft"]]

    if args.json:
        print(json.dumps(with_ft + without_ft, indent=2, default=str))
        return

    print(
        f"{'#':<4} {'EXP_ID':<35} {'STATUS':<14} {'FINISH':<10} "
        f"{'LOSS_MED':<10} {'RETURN':<10} {'FIN_RATE':<10}"
    )
    print("-" * 93)
    for i, r in enumerate(with_ft + without_ft, 1):
        ft_s = f"{r['best_ft']:.2f}s" if r["best_ft"] else "DNF"
        lm = f"{r['loss_median']:.1f}" if r.get("loss_median") is not None else "-"
        rl = f"{r['return_last']:.0f}" if r.get("return_last") is not None else "-"
        fr = f"{r['finish_rate']:.1%}" if r.get("finish_rate") is not None else "-"
        print(f"{i:<4} {r['exp_id']:<35} {r['status']:<14} {ft_s:<10} {lm:<10} {rl:<10} {fr:<10}")


def _delete_exp_artifacts(exp_id: str, dry_run: bool) -> list[str]:
    """Remove per-experiment config, analysis, and log directory."""
    actions: list[str] = []
    cfg = CONFIGS_DIR / f"{exp_id}.yaml"
    if cfg.exists():
        actions.append(f"delete {cfg}")
        if not dry_run:
            cfg.unlink()
    aj = ANALYSIS_DIR / f"{exp_id}.json"
    if aj.exists():
        actions.append(f"delete {aj}")
        if not dry_run:
            aj.unlink()
    log_dir = EXPERIMENTS_LOGS_DIR / exp_id
    if log_dir.exists() and log_dir.is_dir():
        actions.append(f"rmtree {log_dir}")
        if not dry_run:
            shutil.rmtree(log_dir, ignore_errors=True)
    return actions


def _delete_all_experiment_artifacts(dry_run: bool) -> list[str]:
    """Remove every file under configs/, analysis/, and logs/ (experiment outputs only)."""
    actions: list[str] = []
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(CONFIGS_DIR.glob("*.yaml")):
        actions.append(f"delete {p}")
        if not dry_run:
            p.unlink()
    for p in sorted(ANALYSIS_DIR.glob("*.json")):
        actions.append(f"delete {p}")
        if not dry_run:
            p.unlink()
    for child in sorted(EXPERIMENTS_LOGS_DIR.iterdir(), key=lambda x: x.name):
        if child.is_dir():
            actions.append(f"rmtree {child}")
            if not dry_run:
                shutil.rmtree(child, ignore_errors=True)
        elif child.is_file():
            actions.append(f"delete {child}")
            if not dry_run:
                with contextlib.suppress(OSError):
                    child.unlink()
    return actions


def cmd_reset(args: argparse.Namespace) -> None:
    """Drop unfinished runs from the registry (and disk), or wipe everything for a fresh start."""
    dry = args.dry_run
    if not dry and not args.yes:
        print(
            "ERROR: Refusing to change files without --yes. "
            "Re-run with --dry-run to preview, or --yes to apply.",
            file=sys.stderr,
        )
        sys.exit(1)

    entries = _read_registry()
    scope = args.scope
    actions: list[str] = []

    if scope == "all":
        actions.append(
            f"Clear registry ({len(entries)} row(s)) and wipe "
            f"{CONFIGS_DIR.name}/, {ANALYSIS_DIR.name}/, {EXPERIMENTS_LOGS_DIR.name}/"
        )
        if not dry:
            _write_registry([])
        actions.extend(_delete_all_experiment_artifacts(dry))
        if args.clear_agent_context and AGENT_CONTEXT_PATH.exists():
            actions.append(f"delete {AGENT_CONTEXT_PATH}")
            if not dry:
                AGENT_CONTEXT_PATH.unlink()
        if args.clear_decisions:
            ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
            body = (
                f"# Experiment Decisions Log\n\n"
                f"*(Registry reset at {ts} — restored stub. "
                f"Restore full history from git if needed.)*\n\n"
                f"**Target:** See `experiments/orchestrator_config.yaml`"
                f" (`target_finish_time_s`).\n\n"
                f"---\n\n"
                f"*Append new decisions below this line.*\n"
            )
            actions.append(f"overwrite {DECISIONS_PATH} (stub)")
            if not dry:
                DECISIONS_PATH.write_text(body, encoding="utf-8")
    else:
        incomplete = {"failed", "planned", "running"}
        if args.include_stopped_early:
            incomplete.add("stopped_early")
        to_remove = [e for e in entries if e.get("status") in incomplete]
        remove_ids = {e["exp_id"] for e in to_remove}
        kept = [e for e in entries if e["exp_id"] not in remove_ids]

        if args.prune_orphans:
            kept_ids = {e["exp_id"] for e in kept}
            for yf in CONFIGS_DIR.glob("*.yaml"):
                if yf.stem not in kept_ids:
                    remove_ids.add(yf.stem)

        actions.append(
            f"Registry: keep {len(kept)} experiment(s), remove {len(remove_ids)} id(s): "
            f"{sorted(remove_ids)}"
        )
        if not dry:
            _write_registry(kept)
        for eid in sorted(remove_ids):
            actions.extend(_delete_exp_artifacts(eid, dry))

    print("=== reset {} {}===".format(scope, "(dry-run) " if dry else ""))
    for a in actions:
        print(f"  {a}")
    print(f"Done ({len(actions)} action(s)).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="experiment_manager",
        description="TMRL Experiment Manager for agentic hyperparameter tuning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="Register a new experiment")
    p_reg.add_argument("--exp-id", default="", help="Override auto-generated ID")
    p_reg.add_argument("--parent", default="baseline", help="Parent experiment ID")
    p_reg.add_argument("--hypothesis", default="", help="Why this experiment")
    p_reg.add_argument("--overrides", default="{}", help="JSON dict of config overrides")

    sub.add_parser("status", help="Show all experiments and their states")

    _def_entity, _def_project = _orch_defaults()

    p_analyze = sub.add_parser("analyze", help="Pull W&B metrics for an experiment")
    p_analyze.add_argument("--exp-id", required=True)
    p_analyze.add_argument("--entity", default=_def_entity)
    p_analyze.add_argument("--project", default=_def_project)

    p_compare = sub.add_parser("compare", help="Diff two experiments")
    p_compare.add_argument("--exp-a", required=True)
    p_compare.add_argument("--exp-b", required=True)
    p_compare.add_argument("--json", action="store_true", help="Output as JSON")

    sub.add_parser("suggest", help="Cross-experiment suggestions")

    p_snap = sub.add_parser("snapshot", help="Current metrics snapshot (JSON to stdout)")
    p_snap.add_argument("--exp-id", required=True)
    p_snap.add_argument("--entity", default=_def_entity)
    p_snap.add_argument("--project", default=_def_project)

    p_brief = sub.add_parser("briefing", help="Comprehensive context for proposal agents")
    p_brief.add_argument(
        "--json", action="store_true", help="Output as JSON (default: human-readable)"
    )

    p_lb = sub.add_parser("leaderboard", help="Rank experiments by finish time")
    p_lb.add_argument("--json", action="store_true", help="Output as JSON")

    p_reset = sub.add_parser(
        "reset",
        help="Remove unfinished experiments from registry+disk, or wipe all experiment state",
    )
    p_reset.add_argument(
        "scope",
        choices=["incomplete", "all"],
        help="incomplete: failed/planned/running (+optional stopped_early); "
        "all: empty registry and delete all configs, analysis JSON, and log dirs",
    )
    p_reset.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Required (except with --dry-run) to actually delete or rewrite files",
    )
    p_reset.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not modify files",
    )
    p_reset.add_argument(
        "--include-stopped-early",
        action="store_true",
        help="With incomplete: also remove stopped_early runs from registry and disk",
    )
    p_reset.add_argument(
        "--prune-orphans",
        action="store_true",
        help="With incomplete: delete configs/analysis/logs for YAML names not in kept registry",
    )
    p_reset.add_argument(
        "--clear-decisions",
        action="store_true",
        help="With all: overwrite decisions.md with a short stub (use git restore to undo)",
    )
    p_reset.add_argument(
        "--clear-agent-context",
        action="store_true",
        help="With all: delete experiments/_agent_context.json",
    )

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "suggest":
        cmd_suggest(args)
    elif args.command == "snapshot":
        cmd_snapshot(args)
    elif args.command == "briefing":
        cmd_briefing(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "reset":
        cmd_reset(args)


if __name__ == "__main__":
    main()
