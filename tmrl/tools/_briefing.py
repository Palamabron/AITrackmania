"""Experiment briefing: compute and display comprehensive tuning context."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
from typing import Any

import yaml

from tmrl.tools._exp_config_utils import (
    ANALYSIS_DIR,
    SEARCH_SPACE_PATH,
    _load_target_time,
)
from tmrl.tools._experiment_io import (
    read_registry as _read_registry,
)
from tmrl.tools._wandb_snapshot import _flatten_dict


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

    # Load all saved analyses
    analyses: dict[str, dict[str, Any]] = {}
    for e in entries:
        ap = ANALYSIS_DIR / f"{e['exp_id']}.json"
        if ap.exists():
            with contextlib.suppress(Exception):
                analyses[e["exp_id"]] = json.loads(ap.read_text(encoding="utf-8"))

    # Leaderboard (best finish time first, DNFs last)
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

    # Parameter effect analysis
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

    # Search space coverage
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

    # Failure patterns
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

    # Cross-experiment insights
    insights: list[str] = []

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
