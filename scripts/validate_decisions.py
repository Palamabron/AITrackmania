#!/usr/bin/env python3
"""Validate decisions.md claims against analysis JSON ground truth.

Designed to run every autoresearch iteration to catch stale or incorrect
claims before they compound into bad decisions.

Checks:
  1. Finish time claims vs actual best_finish_time_s
  2. "never finished" / "no finishes" claims vs worker and eval data
  3. Loss-based divergence claims vs actual loss statistics
  4. Gradient clipping claims vs actual grad_norm_pre_clip data
  5. Leaderboard correctness (who's actually the best?)
  6. Missing experiments (in registry but no analysis)

Usage:
    python scripts/validate_decisions.py
    python scripts/validate_decisions.py --json-out validation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DECISIONS_PATH = EXPERIMENTS_DIR / "decisions.md"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"


@dataclass
class Finding:
    exp_id: str
    severity: str  # ERROR, WARNING, INFO
    category: str
    claim: str
    ground_truth: str
    suggestion: str = ""


def _load_analysis() -> dict[str, dict]:
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


def _parse_sections(text: str) -> list[dict]:
    """Extract individual checkpoint decision sections."""
    sections = []
    pattern = re.compile(
        r"###\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+UTC)\s+--\s+([\w-]+)\s*\n(.*?)(?=\n###|\n---|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        ts, exp_id, body = m.group(1), m.group(2), m.group(3).strip()
        action_m = re.search(r"\*\*Action:\*\*\s*(\w+)", body)
        reason_m = re.search(r"\*\*Reason:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)", body, re.DOTALL)
        sections.append(
            {
                "timestamp": ts,
                "exp_id": exp_id,
                "action": action_m.group(1) if action_m else None,
                "reason": reason_m.group(1).strip() if reason_m else body,
                "body": body,
            }
        )
    return sections


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# Validation checks


def check_finish_claims(section: dict, analysis: dict) -> list[Finding]:
    """Check if finish time numbers in the reason match ground truth."""
    findings = []
    reason = section["reason"]
    exp_id = section["exp_id"]

    best_ft = analysis.get("best_finish_time_s")
    worker_best = _get(analysis, "worker", "best_finish_time_s")
    worker_count = _get(analysis, "worker", "finish_count", default=0)

    # "never finished" / "has not finished" claims
    no_finish_claim = bool(
        re.search(r"has\s+not\s+(?:yet\s+)?finish", reason, re.I)
        or re.search(r"hasn't\s+finish", reason, re.I)
        or re.search(r"no\s+(?:track\s+)?(?:finishes|completions)", reason, re.I)
        or re.search(r"not\s+(?:yet\s+)?(?:finished|completed)\s+(?:the\s+)?track", reason, re.I)
    )

    if no_finish_claim:
        if best_ft and best_ft > 0:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="ERROR",
                    category="finish_status",
                    claim="Claimed no finish",
                    ground_truth=f"best_finish={best_ft:.2f}s (eval), worker_count={worker_count}",
                    suggestion="Agent had eval finishes; update decision to reflect this",
                )
            )
        elif worker_best and worker_best > 0:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="WARNING",
                    category="finish_status",
                    claim="Claimed no finish",
                    ground_truth=(
                        f"Worker best={worker_best:.2f}s, count={worker_count} (eval=0, lag)"
                    ),
                    suggestion="Worker was finishing; eval may lag behind worker. Note both.",
                )
            )

    # Explicit finish time numbers in text
    ft_nums = re.findall(r"best_finish_time_s[=:\s]+(\d+\.?\d*)", reason)
    ft_nums += re.findall(r"best\s+(?:finish\s+)?time[^.]*?(\d{2,4}\.?\d*)\s*s", reason, re.I)
    for num_str in ft_nums:
        claimed = float(num_str)
        if best_ft and best_ft > 0 and abs(claimed - best_ft) > 5.0:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="WARNING",
                    category="finish_time_value",
                    claim=f"Mentioned {claimed:.1f}s",
                    ground_truth=f"Final best_finish_time_s={best_ft:.2f}s",
                    suggestion="Time was a checkpoint snapshot; final value differs.",
                )
            )

    return findings


def check_loss_claims(section: dict, analysis: dict) -> list[Finding]:
    """Check loss/divergence claims against actual loss stats."""
    findings = []
    reason = section["reason"]
    exp_id = section["exp_id"]

    loss_stats = _get(analysis, "metrics", "loss/iqn_loss", default={})
    loss_median = loss_stats.get("median")
    loss_mean = loss_stats.get("mean")

    if (
        re.search(r"loss\s+(?:is\s+)?consistently\s+above\s+50", reason, re.I)
        and loss_median is not None
        and loss_median < 50
    ):
        findings.append(
            Finding(
                exp_id=exp_id,
                severity="WARNING",
                category="loss_assessment",
                claim="Loss consistently above 50",
                ground_truth=f"Full-run loss median={loss_median:.1f}, mean={loss_mean:.1f}",
                suggestion="Loss median is below 50 for the full run. "
                "Checkpoint may have been at a peak.",
            )
        )

    if re.search(r"(?:indicating|signs?\s+of)\s+divergence", reason, re.I):
        best_ft = analysis.get("best_finish_time_s")
        if best_ft and best_ft > 0:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="WARNING",
                    category="divergence_claim",
                    claim="Claimed divergence",
                    ground_truth=f"Experiment achieved best_finish={best_ft:.2f}s",
                    suggestion="Divergence was overstated; experiment produced finishes.",
                )
            )

    return findings


def check_gradient_claims(section: dict, analysis: dict) -> list[Finding]:
    """Check gradient clipping claims against actual data."""
    findings = []
    reason = section["reason"]
    exp_id = section["exp_id"]

    pre_clip = _get(analysis, "metrics", "debug/grad_norm_pre_clip", default={})
    grad_norm = _get(analysis, "metrics", "debug/grad_norm", default={})

    pre_clip_mean = pre_clip.get("mean")
    grad_norm_median = grad_norm.get("median")

    # Check "saturating at clip limit" claims: see if grad norm is indeed at clip
    if re.search(r"saturat\w+\s+at\s+(?:the\s+)?clip\s+limit", reason, re.I) and (
        grad_norm_median is not None and pre_clip_mean is not None
    ):
        ratio = pre_clip_mean / grad_norm_median if grad_norm_median > 0 else 0
        if ratio < 2:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="INFO",
                    category="gradient_saturation",
                    claim="Gradients saturating at clip limit",
                    ground_truth=f"pre_clip/clip ratio={ratio:.1f}x (mild clipping)",
                    suggestion="Clipping is mild; saturation claim overstated.",
                )
            )

    # Check pre-clip norm claims (e.g., "mean 121.57, max 509.68")
    pre_clip_claims = re.findall(
        r"pre[_-]clip\s+(?:norms?\s+)?(?:\()?mean\s+(\d+\.?\d*)", reason, re.I
    )
    for claimed_str in pre_clip_claims:
        claimed = float(claimed_str)
        if pre_clip_mean is not None and abs(claimed - pre_clip_mean) / max(pre_clip_mean, 1) > 0.5:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="INFO",
                    category="grad_pre_clip_value",
                    claim=f"pre_clip mean={claimed:.1f}",
                    ground_truth=f"Full-run pre_clip mean={pre_clip_mean:.1f}",
                    suggestion="Checkpoint value vs full-run differ significantly.",
                )
            )

    return findings


def check_leaderboard(analyses: dict[str, dict]) -> list[Finding]:
    """Check the overall leaderboard for correctness."""
    findings = []

    ranked = []
    for exp_id, data in analyses.items():
        ft = data.get("best_finish_time_s")
        if ft and ft > 0:
            ranked.append((exp_id, ft))
    ranked.sort(key=lambda x: x[1])

    if not ranked:
        findings.append(
            Finding(
                exp_id="GLOBAL",
                severity="WARNING",
                category="leaderboard",
                claim="No experiments with finishes",
                ground_truth="All experiments are DNF",
                suggestion="Review experimental approach; no configuration finished the track.",
            )
        )
        return findings

    best_id, best_time = ranked[0]
    findings.append(
        Finding(
            exp_id="GLOBAL",
            severity="INFO",
            category="leaderboard",
            claim=f"Current leader: {best_id}",
            ground_truth=f"Best time: {best_time:.2f}s",
            suggestion="",
        )
    )

    # Check if decisions.md mentions a different leader
    if DECISIONS_PATH.exists():
        text = DECISIONS_PATH.read_text(encoding="utf-8")
        leader_claim = re.search(r"leader\s+so\s+far", text, re.I)
        if leader_claim:
            context = text[max(0, leader_claim.start() - 100) : leader_claim.end() + 50]
            if best_id not in context:
                findings.append(
                    Finding(
                        exp_id="GLOBAL",
                        severity="WARNING",
                        category="leaderboard_mismatch",
                        claim="decisions.md claims a different leader",
                        ground_truth=f"Actual leader is {best_id} at {best_time:.2f}s",
                        suggestion="Update leaderboard claim in decisions.md",
                    )
                )

    # Check for experiments where worker finishes but eval doesn't
    for exp_id, data in analyses.items():
        eval_ft = data.get("best_finish_time_s")
        worker_best = _get(data, "worker", "best_finish_time_s")
        worker_count = _get(data, "worker", "finish_count", default=0)

        if (not eval_ft or eval_ft <= 0) and worker_best and worker_best > 0 and worker_count >= 5:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="WARNING",
                    category="eval_worker_gap",
                    claim="Eval shows no finish",
                    ground_truth=f"Worker has {worker_count} finishes, best={worker_best:.2f}s",
                    suggestion="Deterministic eval may be lagging. "
                    "Consider running longer or checking eval frequency.",
                )
            )

    return findings


def check_missing_analyses(registry: list[dict], analyses: dict[str, dict]) -> list[Finding]:
    """Flag experiments in registry without analysis files."""
    findings = []
    for entry in registry:
        exp_id = entry["exp_id"]
        if exp_id not in analyses:
            findings.append(
                Finding(
                    exp_id=exp_id,
                    severity="WARNING",
                    category="missing_analysis",
                    claim="Experiment in registry",
                    ground_truth="No analysis JSON found",
                    suggestion="Run: python scripts/fetch_analysis.py --exp-id " + exp_id,
                )
            )
    return findings


def check_stop_patterns(sections: list[dict], analyses: dict[str, dict]) -> list[Finding]:
    """Identify problematic stop patterns across experiments."""
    findings = []

    early_stops_with_worker_finishes = []
    for section in sections:
        if section["action"] != "stop":
            continue
        exp_id = section["exp_id"]
        data = analyses.get(exp_id)
        if not data:
            continue
        worker_count = _get(data, "worker", "finish_count", default=0)
        worker_best = _get(data, "worker", "best_finish_time_s")
        if worker_count > 5 and worker_best and worker_best > 0:
            early_stops_with_worker_finishes.append((exp_id, worker_count, worker_best))

    if early_stops_with_worker_finishes:
        details = "; ".join(
            f"{e[0]}(count={e[1]},best={e[2]:.1f}s)" for e in early_stops_with_worker_finishes
        )
        findings.append(
            Finding(
                exp_id="PATTERN",
                severity="WARNING",
                category="premature_stops",
                claim=(
                    f"{len(early_stops_with_worker_finishes)} experiments "
                    "stopped despite worker finishes"
                ),
                ground_truth=details,
                suggestion="Eval may lag worker. Consider using worker finish count as "
                "a keep-running signal.",
            )
        )

    # Check for gradient-obsessed stops
    grad_stops = []
    for section in sections:
        if section["action"] != "stop":
            continue
        if re.search(r"gradient\s+norm", section["reason"], re.I):
            grad_stops.append(section["exp_id"])

    if len(grad_stops) > 3:
        findings.append(
            Finding(
                exp_id="PATTERN",
                severity="WARNING",
                category="gradient_obsession",
                claim=f"{len(grad_stops)} experiments stopped due to gradient norms",
                ground_truth=f"Experiments: {', '.join(grad_stops[:5])}...",
                suggestion="Pre-clip gradient norms being high is structural to this model. "
                "Focus on Q-value stability and finish rate instead.",
            )
        )

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate decisions.md against ground truth")
    parser.add_argument("--json-out", default="", help="Path to write JSON report")
    args = parser.parse_args()

    if not DECISIONS_PATH.exists():
        print(f"ERROR: {DECISIONS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    text = DECISIONS_PATH.read_text(encoding="utf-8")
    analyses = _load_analysis()
    registry = _load_registry()
    sections = _parse_sections(text)

    all_findings: list[Finding] = []

    # Per-section checks
    for section in sections:
        exp_id = section["exp_id"]
        data = analyses.get(exp_id)
        if not data:
            continue
        all_findings.extend(check_finish_claims(section, data))
        all_findings.extend(check_loss_claims(section, data))
        all_findings.extend(check_gradient_claims(section, data))

    # Global checks
    all_findings.extend(check_leaderboard(analyses))
    all_findings.extend(check_missing_analyses(registry, analyses))
    all_findings.extend(check_stop_patterns(sections, analyses))

    errors = [f for f in all_findings if f.severity == "ERROR"]
    warnings = [f for f in all_findings if f.severity == "WARNING"]
    infos = [f for f in all_findings if f.severity == "INFO"]

    print(f"\n{'=' * 80}")
    print("DECISIONS.MD VALIDATION REPORT")
    print(f"{'=' * 80}")
    print(f"Sections analyzed: {len(sections)}")
    print(f"Experiments with analysis: {len(analyses)}")
    print(f"Errors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Info:     {len(infos)}")

    if errors:
        print(f"\n{'-' * 40}")
        print("ERRORS (factually incorrect)")
        print(f"{'-' * 40}")
        for f in errors:
            print(f"\n  [{f.severity}] {f.exp_id} / {f.category}")
            print(f"    Claim:   {f.claim}")
            print(f"    Truth:   {f.ground_truth}")
            if f.suggestion:
                print(f"    Fix:     {f.suggestion}")

    if warnings:
        print(f"\n{'-' * 40}")
        print("WARNINGS (misleading or stale)")
        print(f"{'-' * 40}")
        for f in warnings:
            print(f"\n  [{f.severity}] {f.exp_id} / {f.category}")
            print(f"    Claim:   {f.claim}")
            print(f"    Truth:   {f.ground_truth}")
            if f.suggestion:
                print(f"    Fix:     {f.suggestion}")

    if infos:
        print(f"\n{'-' * 40}")
        print("INFO (notes)")
        print(f"{'-' * 40}")
        for f in infos:
            print(f"\n  [{f.severity}] {f.exp_id} / {f.category}")
            print(f"    {f.claim}: {f.ground_truth}")

    # Leaderboard
    ranked = sorted(
        [(eid, d.get("best_finish_time_s", 0) or 0) for eid, d in analyses.items()],
        key=lambda x: x[1] if x[1] > 0 else float("inf"),
    )
    print(f"\n{'-' * 40}")
    print("CORRECTED LEADERBOARD")
    print(f"{'-' * 40}")
    print(f"  {'#':<4} {'Experiment':<45} {'Best':>8} {'Worker%':>8} {'Loss Med':>9}")
    print(f"  {'-' * 4} {'-' * 45} {'-' * 8} {'-' * 8} {'-' * 9}")
    rank = 1
    for eid, ft in ranked:
        data = analyses[eid]
        ft_str = f"{ft:.1f}s" if ft > 0 else "DNF"
        wr = _get(data, "worker", "finish_rate", default=0)
        wr_str = f"{wr * 100:.0f}%" if wr else "0%"
        lm = _get(data, "metrics", "loss/iqn_loss", "median")
        lm_str = f"{lm:.1f}" if lm is not None else "-"
        r_str = str(rank) if ft > 0 else "-"
        if ft > 0:
            rank += 1
        print(f"  {r_str:<4} {eid:<45} {ft_str:>8} {wr_str:>8} {lm_str:>9}")

    if args.json_out:
        report = {
            "sections_analyzed": len(sections),
            "experiments_with_analysis": len(analyses),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(infos),
            "findings": [
                {
                    "exp_id": f.exp_id,
                    "severity": f.severity,
                    "category": f.category,
                    "claim": f.claim,
                    "ground_truth": f.ground_truth,
                    "suggestion": f.suggestion,
                }
                for f in all_findings
            ],
            "leaderboard": [
                {
                    "rank": i + 1 if ft > 0 else None,
                    "exp_id": eid,
                    "best_time_s": ft if ft > 0 else None,
                }
                for i, (eid, ft) in enumerate(ranked)
                if ft > 0
            ],
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
