"""W&B snapshot, compare, and suggest commands for experiment_manager."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any

from tmrl.tools._exp_config_utils import (
    ANALYSIS_DIR,
    _build_full_config,
    _deep_diff,
    _load_target_time,
    _retry,
    _safe_float_series,
    _warn,
)
from tmrl.tools._experiment_io import (
    load_dotenv as _load_dotenv,
)
from tmrl.tools._experiment_io import (
    read_registry as _read_registry,
)


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a nested dict into ``[(dotted.key, value), …]`` pairs.

    Args:
        d: The dict to flatten.
        prefix: Dot-separated key prefix accumulated during recursion.

    Returns:
        A list of ``(dotted_key, leaf_value)`` tuples.
    """
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


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two experiments by config diff and key metrics, printing a text or JSON report."""
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
    """Print cross-experiment suggestions derived from completed analysis files."""
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


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Pull current W&B metrics for a running experiment and print a JSON snapshot.

    Fetches the trainer and worker run histories, computes recent-metric
    statistics (last 100 trainer steps), extracts best/last finish times,
    and emits a JSON object to stdout.  Used by the orchestrator to decide
    whether to continue or stop a run.
    """
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
            """Return basic stats for *col* over the recent tail, or ``None`` if absent/empty."""
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
