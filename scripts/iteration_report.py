#!/usr/bin/env python3
"""Generate a comprehensive iteration report from local analysis data.

Run after every autoresearch iteration (or after fetch_analysis.py) to get a
standardized summary: leaderboard, per-experiment health cards, aggregate
patterns, and actionable next-step suggestions.

Usage:
    python scripts/iteration_report.py
    python scripts/iteration_report.py --format json
    python scripts/iteration_report.py --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"
BASELINE_PATH = EXPERIMENTS_DIR / "baseline.yaml"
TARGET_FINISH_S = 36.0


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _load_analyses() -> dict[str, dict]:
    result = {}
    for f in sorted(ANALYSIS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result[data.get("exp_id", f.stem)] = data
        except Exception:
            pass
    return result


def _load_registry() -> list[dict]:
    entries = []
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _flatten_overrides(overrides: dict, prefix: str = "") -> list[tuple[str, Any]]:
    items = []
    for k, v in overrides.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_overrides(v, full_key))
        else:
            items.append((full_key, v))
    return items


# ── Report sections ─────────────────────────────────────────────────


def section_leaderboard(analyses: dict[str, dict], top_n: int = 0) -> str:
    ranked = []
    for exp_id, data in analyses.items():
        ft = data.get("best_finish_time_s")
        if ft and ft > 0:
            ranked.append((exp_id, ft, data))
    ranked.sort(key=lambda x: x[1])

    dnf = [
        (eid, d)
        for eid, d in analyses.items()
        if not d.get("best_finish_time_s") or d["best_finish_time_s"] <= 0
    ]

    lines = ["## Leaderboard", ""]
    lines.append(
        f"Target: {TARGET_FINISH_S}s | Experiments with finishes: {len(ranked)}/{len(analyses)}"
    )
    lines.append("")
    lines.append(
        f"{'#':<4} {'Experiment':<42} {'Best':>8} {'WrkBest':>8} {'Wrk%':>6} "
        f"{'LossMed':>8} {'RetTrn':>8} {'Trend':>10}"
    )
    lines.append(
        f"{'─' * 4} {'─' * 42} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 10}"
    )

    show = ranked[:top_n] if top_n > 0 else ranked
    for i, (eid, ft, data) in enumerate(show, 1):
        wb = _get(data, "worker", "best_finish_time_s")
        wb_s = f"{wb:.1f}s" if wb else "-"
        wr = _get(data, "worker", "finish_rate", default=0)
        wr_s = f"{wr * 100:.0f}%"
        lm = _get(data, "metrics", "loss/iqn_loss", "median")
        lm_s = f"{lm:.1f}" if lm is not None else "-"
        rt = _get(data, "metrics", "metrics/return_train", "last")
        rt_s = f"{rt:.0f}" if rt is not None else "-"
        trend = _get(data, "training_trends", "metrics/return_train", "direction", default="?")
        lines.append(
            f"{i:<4} {eid:<42} {ft:>7.1f}s {wb_s:>8} {wr_s:>6} {lm_s:>8} {rt_s:>8} {trend:>10}"
        )

    if dnf:
        lines.append("")
        lines.append(f"DNF ({len(dnf)} experiments):")
        for eid, data in sorted(dnf, key=lambda x: x[0]):
            wc = _get(data, "worker", "finish_count", default=0)
            wb = _get(data, "worker", "best_finish_time_s")
            note = ""
            if wc > 0 and wb:
                note = f" (worker: {wc} finishes, best {wb:.1f}s)"
            lines.append(f"  - {eid}{note}")

    if ranked:
        lines.append("")
        gap = ranked[0][1] - TARGET_FINISH_S
        lines.append(
            f"Gap to target: {gap:+.1f}s (best={ranked[0][1]:.2f}s, target={TARGET_FINISH_S}s)"
        )

    return "\n".join(lines)


def section_health_cards(analyses: dict[str, dict]) -> str:
    lines = ["## Per-Experiment Health Cards", ""]

    for exp_id in sorted(analyses.keys()):
        data = analyses[exp_id]
        ft = data.get("best_finish_time_s")
        ft_s = f"{ft:.2f}s" if ft and ft > 0 else "DNF"

        loss_med = _get(data, "metrics", "loss/iqn_loss", "median")
        loss_last = _get(data, "metrics", "loss/iqn_loss", "last")
        max_q = _get(data, "metrics", "q/max_q", "last")
        mean_q = _get(data, "metrics", "q/mean_q", "last")
        eps = _get(data, "metrics", "exploration/epsilon", "last")
        grad_norm = _get(data, "metrics", "debug/grad_norm", "median")
        pre_clip = _get(data, "metrics", "debug/grad_norm_pre_clip", "mean")
        ret_train = _get(data, "metrics", "metrics/return_train", "last")
        wr = _get(data, "worker", "finish_rate", default=0)
        wc = _get(data, "worker", "finish_count", default=0)

        # Health flags
        flags = []
        if loss_last and loss_med and loss_last > loss_med * 2:
            flags.append("LOSS_SPIKE")
        if max_q and max_q > 100:
            flags.append("Q_EXPLODING")
        if pre_clip and grad_norm and grad_norm > 0:
            clip_ratio = pre_clip / grad_norm
            if clip_ratio > 50:
                flags.append(f"SEVERE_CLIP({clip_ratio:.0f}x)")
            elif clip_ratio > 10:
                flags.append(f"HEAVY_CLIP({clip_ratio:.0f}x)")
        if ft and ft > 0 and ft > TARGET_FINISH_S * 2.5:
            flags.append("FAR_FROM_TARGET")

        flag_str = " | ".join(flags) if flags else "OK"

        lines.append(f"### {exp_id}")
        lines.append(
            f"  Finish: {ft_s} | Worker: {wc} finishes ({wr * 100:.0f}%) | Flags: {flag_str}"
        )
        lm_s = f"{loss_med:.1f}" if loss_med else "0"
        ll_s = f"{loss_last:.1f}" if loss_last else "0"
        mq_s = f"{max_q:.1f}" if max_q else "0"
        mnq_s = f"{mean_q:.1f}" if mean_q else "0"
        eps_s = f"{eps:.3f}" if eps else "0"
        lines.append(f"  Loss: med={lm_s}, last={ll_s} | Q: max={mq_s}, mean={mnq_s} | eps={eps_s}")
        if pre_clip:
            if grad_norm and grad_norm > 0:
                lines.append(
                    f"  Grad: norm={grad_norm:.2f}, pre_clip_mean={pre_clip:.1f}, "
                    f"clip_ratio={pre_clip / grad_norm:.0f}x"
                )
            else:
                lines.append(f"  Grad: pre_clip_mean={pre_clip:.1f}")
        rt_s = f"{ret_train:.0f}" if ret_train else "0"
        lines.append(f"  Return(train): {rt_s}")

        trends = data.get("training_trends", {})
        if trends:
            t_parts = [f"{k.split('/')[-1]}={v.get('direction', '?')}" for k, v in trends.items()]
            lines.append(f"  Trends: {', '.join(t_parts)}")
        lines.append("")

    return "\n".join(lines)


def section_patterns(analyses: dict[str, dict], registry: list[dict]) -> str:
    lines = ["## Aggregate Patterns", ""]

    reg_map = {e["exp_id"]: e for e in registry}

    # Parameter impact analysis
    param_results: dict[str, list[tuple[Any, float | None]]] = defaultdict(list)
    for exp_id, data in analyses.items():
        entry = reg_map.get(exp_id, {})
        overrides = entry.get("config_overrides", {})
        ft = data.get("best_finish_time_s")
        for key, val in _flatten_overrides(overrides):
            param_results[key].append((val, ft))

    if param_results:
        lines.append("### Parameter Impact")
        lines.append("")
        for param, results in sorted(param_results.items()):
            if len(results) < 1:
                continue
            finished = [(v, t) for v, t in results if t and t > 0]
            dnf = [(v, t) for v, t in results if not t or t <= 0]
            parts = []
            for val, ft in sorted(finished, key=lambda x: x[1]):
                parts.append(f"  {val} -> {ft:.1f}s")
            for val, _ in dnf:
                parts.append(f"  {val} -> DNF")
            if parts:
                lines.append(f"  {param}:")
                for p in parts:
                    lines.append(f"    {p}")
        lines.append("")

    # Gradient clipping severity
    clip_data = []
    for exp_id, data in analyses.items():
        pre_clip = _get(data, "metrics", "debug/grad_norm_pre_clip", "mean")
        grad_norm = _get(data, "metrics", "debug/grad_norm", "median")
        if pre_clip and grad_norm and grad_norm > 0:
            clip_data.append((exp_id, pre_clip / grad_norm, pre_clip))
    if clip_data:
        clip_data.sort(key=lambda x: x[1], reverse=True)
        lines.append("### Gradient Clipping Severity (pre_clip / clip_limit)")
        lines.append("")
        for eid, ratio, pre in clip_data[:10]:
            lines.append(f"  {eid:<42} {ratio:>6.0f}x  (pre_clip_mean={pre:.1f})")
        lines.append("")

    # Loss vs finish time correlation
    loss_ft_pairs = []
    for exp_id, data in analyses.items():
        ft = data.get("best_finish_time_s")
        lm = _get(data, "metrics", "loss/iqn_loss", "median")
        if ft and ft > 0 and lm is not None:
            loss_ft_pairs.append((exp_id, lm, ft))
    if loss_ft_pairs:
        lines.append("### Loss vs Finish Time")
        lines.append("")
        loss_ft_pairs.sort(key=lambda x: x[2])
        for eid, lm, ft in loss_ft_pairs:
            lines.append(f"  {eid:<42} loss_med={lm:>6.1f}  finish={ft:>7.1f}s")
        lines.append("")

    # Termination patterns
    term_counts: Counter = Counter()
    for data in analyses.values():
        tc = _get(data, "worker", "termination_counts", default={})
        for reason, count in tc.items():
            if isinstance(count, (int, float)):
                term_counts[reason] += int(count)
    if term_counts:
        total = sum(term_counts.values())
        lines.append("### Termination Reasons (aggregate)")
        lines.append("")
        for reason, count in term_counts.most_common():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {reason:<30} {count:>5} ({pct:.1f}%)")
        lines.append("")

    return "\n".join(lines)


def section_suggestions(analyses: dict[str, dict], registry: list[dict]) -> str:
    lines = ["## Suggestions for Next Iteration", ""]

    ranked = sorted(
        [(eid, d.get("best_finish_time_s", 0) or 0, d) for eid, d in analyses.items()],
        key=lambda x: x[1] if x[1] > 0 else float("inf"),
    )
    best_finished = [(eid, ft, d) for eid, ft, d in ranked if ft > 0]

    if not best_finished:
        lines.append(
            "- **Critical:** No experiments have finished the track. "
            "Consider reducing track difficulty or increasing training time."
        )
        return "\n".join(lines)

    best_id, best_ft, _best_data = best_finished[0]

    if best_ft > TARGET_FINISH_S * 2:
        lines.append(
            f"- **Gap is large** ({best_ft:.1f}s vs {TARGET_FINISH_S}s target). "
            f"Focus on getting consistent finishes before optimizing time."
        )
    elif best_ft > TARGET_FINISH_S * 1.3:
        lines.append(
            f"- **Approaching target** ({best_ft:.1f}s vs {TARGET_FINISH_S}s). "
            f"Fine-tune around the best config ({best_id})."
        )

    # Check if any pattern is clear
    all_severe_clip = all(
        _get(d, "metrics", "debug/grad_norm_pre_clip", "mean", default=0)
        / max(_get(d, "metrics", "debug/grad_norm", "median", default=1), 0.01)
        > 20
        for d in analyses.values()
        if _get(d, "metrics", "debug/grad_norm_pre_clip", "mean") is not None
    )
    if all_severe_clip:
        lines.append(
            "- **Structural gradient issue:** ALL experiments show severe pre-clip/clip "
            "ratios (>20x). This is likely a model architecture or loss landscape issue, "
            "not fixable by clip tuning alone. Consider: architecture changes, "
            "learning rate warmup, gradient penalty, or spectral normalization."
        )

    # Check return vs baseline for best experiments
    for eid, ft, data in best_finished[:3]:
        vs = data.get("vs_baseline", {})
        ret_delta = _get(vs, "metrics/return_train", "pct", default=0)
        if ret_delta > 0:
            lines.append(
                f"- **{eid}** shows +{ret_delta:.0f}% return over baseline "
                f"(finish={ft:.1f}s). Build on this config."
            )

    # Check what hyperparameters the best experiments share
    reg_map = {e["exp_id"]: e for e in registry}
    best_overrides: dict[str, list] = defaultdict(list)
    for eid, ft, _ in best_finished[:3]:
        entry = reg_map.get(eid, {})
        for key, val in _flatten_overrides(entry.get("config_overrides", {})):
            best_overrides[key].append((val, ft))
    if best_overrides:
        lines.append("")
        lines.append("Config patterns in top-3 finishers:")
        for key, vals in sorted(best_overrides.items()):
            val_str = ", ".join(f"{v}({ft:.0f}s)" for v, ft in vals)
            lines.append(f"  - {key}: {val_str}")

    # Suggest untried directions
    lines.append("")
    lines.append("Potentially untried directions:")
    tried_params = set()
    for entry in registry:
        for key, _ in _flatten_overrides(entry.get("config_overrides", {})):
            tried_params.add(key)

    suggestions = {
        "training.batch_size": "Try larger batch sizes (1024+) if memory allows",
        "algorithm.iqn_epsilon_decay_steps": "Faster epsilon decay for quicker exploitation",
        "environment.end_of_track_reward": "Higher finish bonus to incentivize completion",
        "environment.reward.speed_reward_weight": "Tune speed reward to balance progress vs safety",
        "algorithm.iqn_munchausen_enabled": "Munchausen RL for implicit policy regularization",
        "model.residual_mlp_hidden_dim": "Model capacity changes",
        "algorithm.reward_normalize_scale": "Reward scaling to tame gradients",
    }
    for param, desc in suggestions.items():
        if param not in tried_params:
            lines.append(f"  - {param}: {desc}")

    return "\n".join(lines)


def build_report(analyses: dict[str, dict], registry: list[dict], top_n: int = 0) -> str:
    parts = [
        "# Autoresearch Iteration Report",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        f"Total experiments: {len(analyses)}",
        "",
        section_leaderboard(analyses, top_n),
        "",
        section_patterns(analyses, registry),
        "",
        section_suggestions(analyses, registry),
        "",
        section_health_cards(analyses),
    ]
    return "\n".join(parts)


def build_json_report(analyses: dict[str, dict], registry: list[dict]) -> dict:
    ranked = sorted(
        [(eid, d.get("best_finish_time_s", 0) or 0) for eid, d in analyses.items()],
        key=lambda x: x[1] if x[1] > 0 else float("inf"),
    )

    experiments = []
    for eid, data in sorted(analyses.items()):
        ft = data.get("best_finish_time_s")
        experiments.append(
            {
                "exp_id": eid,
                "best_finish_time_s": ft if ft and ft > 0 else None,
                "worker_finish_count": _get(data, "worker", "finish_count", default=0),
                "worker_finish_rate": _get(data, "worker", "finish_rate", default=0),
                "worker_best_finish_time_s": _get(data, "worker", "best_finish_time_s"),
                "loss_median": _get(data, "metrics", "loss/iqn_loss", "median"),
                "loss_last": _get(data, "metrics", "loss/iqn_loss", "last"),
                "max_q_last": _get(data, "metrics", "q/max_q", "last"),
                "epsilon_last": _get(data, "metrics", "exploration/epsilon", "last"),
                "return_train_last": _get(data, "metrics", "metrics/return_train", "last"),
                "grad_norm_pre_clip_mean": _get(
                    data, "metrics", "debug/grad_norm_pre_clip", "mean"
                ),
                "training_trends": data.get("training_trends"),
            }
        )

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "target_finish_s": TARGET_FINISH_S,
        "total_experiments": len(analyses),
        "experiments_with_finish": sum(1 for _, ft in ranked if ft > 0),
        "best_overall": ranked[0][0] if ranked and ranked[0][1] > 0 else None,
        "best_time_s": ranked[0][1] if ranked and ranked[0][1] > 0 else None,
        "leaderboard": [
            {"rank": i + 1, "exp_id": eid, "best_time_s": ft}
            for i, (eid, ft) in enumerate(ranked)
            if ft > 0
        ],
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate autoresearch iteration report")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--top", type=int, default=0, help="Show only top N in leaderboard")
    parser.add_argument("--out", default="", help="Write to file instead of stdout")
    args = parser.parse_args()

    analyses = _load_analyses()
    registry = _load_registry()

    if not analyses:
        print("No analysis files found. Run: python scripts/fetch_analysis.py", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        report = build_json_report(analyses, registry)
        output = json.dumps(report, indent=2, default=str)
    else:
        output = build_report(analyses, registry, args.top)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
