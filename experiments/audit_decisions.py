#!/usr/bin/env python3
"""Audit decisions.md against analysis JSON ground truth.

Scans decisions.md for claims about best_finish_time_s, finish status, and loss
values, then cross-checks them against the authoritative analysis/*.json files.
Reports discrepancies so agents can correct stale or incorrect conclusions.

Usage:
    python experiments/audit_decisions.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DECISIONS_PATH = EXPERIMENTS_DIR / "decisions.md"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"


@dataclass
class AuditFinding:
    exp_id: str
    severity: str  # ERROR, WARNING, INFO
    field: str
    claim: str
    ground_truth: str
    context: str = ""


@dataclass
class ExperimentGroundTruth:
    exp_id: str
    best_finish_time_s: float | None = None
    worker_best_finish_time_s: float | None = None
    worker_finish_count: int = 0
    worker_finish_rate: float = 0.0
    eval_finish_time_max: float = 0.0
    loss_median: float | None = None
    loss_last: float | None = None
    loss_mean: float | None = None
    status: str = "unknown"


def load_ground_truth() -> dict[str, ExperimentGroundTruth]:
    result = {}

    registry_map: dict[str, dict] = {}
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                e = json.loads(line)
                registry_map[e["exp_id"]] = e

    for f in sorted(ANALYSIS_DIR.glob("*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        exp_id = a.get("exp_id", f.stem)

        gt = ExperimentGroundTruth(exp_id=exp_id)

        best_ft = a.get("best_finish_time_s")
        if best_ft and best_ft > 0:
            gt.best_finish_time_s = best_ft

        worker = a.get("worker", {})
        w_best = worker.get("best_finish_time_s")
        if w_best and w_best > 0:
            gt.worker_best_finish_time_s = w_best
        gt.worker_finish_count = worker.get("finish_count", 0)
        gt.worker_finish_rate = worker.get("finish_rate", 0.0)

        metrics = a.get("metrics", {})
        ft_metric = metrics.get("eval/finish_time_test_s", {})
        gt.eval_finish_time_max = ft_metric.get("max", 0.0)

        loss = metrics.get("loss/iqn_loss", {})
        gt.loss_median = loss.get("median")
        gt.loss_last = loss.get("last")
        gt.loss_mean = loss.get("mean")

        if exp_id in registry_map:
            gt.status = registry_map[exp_id].get("status", "unknown")

        result[exp_id] = gt

    return result


def parse_decisions_sections(text: str) -> list[dict]:
    """Parse decisions.md into sections by experiment checkpoint."""
    sections = []
    pattern = re.compile(
        r"###\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+UTC)\s+--\s+([\w-]+)\s*\n(.*?)(?=\n###|\n---|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        timestamp, exp_id, body = m.group(1), m.group(2), m.group(3).strip()

        action_match = re.search(r"\*\*Action:\*\*\s*(\w+)", body)
        reason_match = re.search(r"\*\*Reason:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)", body, re.DOTALL)

        sections.append(
            {
                "timestamp": timestamp,
                "exp_id": exp_id,
                "action": action_match.group(1) if action_match else None,
                "reason": reason_match.group(1).strip() if reason_match else body,
                "body": body,
            }
        )
    return sections


def audit_section(section: dict, gt: ExperimentGroundTruth) -> list[AuditFinding]:
    findings = []
    reason = section["reason"]
    exp_id = section["exp_id"]

    ft_claims = re.findall(r"best_finish_time_s[=:\s]+(\d+\.?\d*)", reason)
    for claimed_ft_str in ft_claims:
        claimed_ft = float(claimed_ft_str)
        if (
            gt.best_finish_time_s
            and gt.best_finish_time_s > 0
            and abs(claimed_ft - gt.best_finish_time_s) > 1.0
        ):
            findings.append(
                AuditFinding(
                    exp_id=exp_id,
                    severity="ERROR",
                    field="best_finish_time_s",
                    claim=f"{claimed_ft:.2f}s",
                    ground_truth=f"{gt.best_finish_time_s:.2f}s",
                    context=(
                        f"Checkpoint claimed {claimed_ft:.2f}s but "
                        f"final best is {gt.best_finish_time_s:.2f}s"
                    ),
                )
            )

    best_time_claims = re.findall(
        r"best\s+(?:finish\s+)?time[^.]*?(\d{2,4}\.?\d*)\s*s", reason, re.IGNORECASE
    )
    for claimed_str in best_time_claims:
        claimed = float(claimed_str)
        if (
            gt.best_finish_time_s
            and gt.best_finish_time_s > 0
            and abs(claimed - gt.best_finish_time_s) > 5.0
            and claimed > gt.best_finish_time_s
        ):
            findings.append(
                AuditFinding(
                    exp_id=exp_id,
                    severity="WARNING",
                    field="best_time_reference",
                    claim=f"{claimed:.1f}s",
                    ground_truth=f"{gt.best_finish_time_s:.2f}s",
                    context=(
                        f"Mentioned best time {claimed:.1f}s but "
                        f"final best is {gt.best_finish_time_s:.2f}s"
                    ),
                )
            )

    never_finished = (
        re.search(r"has\s+not\s+(?:yet\s+)?finish", reason, re.IGNORECASE)
        or re.search(r"hasn't\s+finish", reason, re.IGNORECASE)
        or re.search(r"agent\s+has\s+not\s+(?:yet\s+)?finish", reason, re.IGNORECASE)
        or re.search(r"no\s+(?:track\s+)?(?:finishes|completions)", reason, re.IGNORECASE)
    )
    if never_finished:
        has_worker_finish = gt.worker_finish_count > 0 or (
            gt.worker_best_finish_time_s and gt.worker_best_finish_time_s > 0
        )
        has_eval_finish = gt.best_finish_time_s and gt.best_finish_time_s > 0
        if has_eval_finish:
            findings.append(
                AuditFinding(
                    exp_id=exp_id,
                    severity="ERROR",
                    field="finish_status",
                    claim="claimed no finish",
                    ground_truth=(
                        f"best_finish={gt.best_finish_time_s:.2f}s, "
                        f"worker_count={gt.worker_finish_count}"
                    ),
                    context="Agent claimed no finish but experiment DID finish",
                )
            )
        elif has_worker_finish:
            findings.append(
                AuditFinding(
                    exp_id=exp_id,
                    severity="WARNING",
                    field="finish_status",
                    claim="claimed no finish",
                    ground_truth=(
                        f"worker_best={gt.worker_best_finish_time_s:.2f}s, "
                        f"worker_count={gt.worker_finish_count}"
                    ),
                    context="Agent claimed no finish but worker WAS finishing (eval lag)",
                )
            )

    loss_above_50 = re.search(r"loss\s+(?:is\s+)?consistently\s+above\s+50", reason, re.IGNORECASE)
    if loss_above_50 and gt.loss_median is not None and gt.loss_median < 50:
        findings.append(
            AuditFinding(
                exp_id=exp_id,
                severity="WARNING",
                field="loss_assessment",
                claim="loss consistently above 50",
                ground_truth=f"loss median={gt.loss_median:.1f}, mean={gt.loss_mean:.1f}",
                context="Loss median is actually below 50 for the full run",
            )
        )

    divergence = re.search(r"(?:indicating|signs? of)\s+divergence", reason, re.IGNORECASE)
    if divergence and gt.best_finish_time_s and gt.best_finish_time_s > 0:
        findings.append(
            AuditFinding(
                exp_id=exp_id,
                severity="WARNING",
                field="divergence_claim",
                claim="claimed divergence",
                ground_truth=f"experiment achieved best_finish={gt.best_finish_time_s:.2f}s",
                context="Divergence claimed but experiment produced finishes",
            )
        )

    return findings


def main():
    if not DECISIONS_PATH.exists():
        print(f"ERROR: {DECISIONS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    text = DECISIONS_PATH.read_text(encoding="utf-8")
    ground_truth = load_ground_truth()
    sections = parse_decisions_sections(text)

    all_findings: list[AuditFinding] = []
    for section in sections:
        gt = ground_truth.get(section["exp_id"])
        if gt:
            findings = audit_section(section, gt)
            all_findings.extend(findings)

    errors = [f for f in all_findings if f.severity == "ERROR"]
    warnings = [f for f in all_findings if f.severity == "WARNING"]

    print(f"\n{'=' * 80}")
    print("DECISIONS.MD AUDIT REPORT")
    print(f"{'=' * 80}")
    print(f"Sections analyzed: {len(sections)}")
    print(f"Errors found:      {len(errors)}")
    print(f"Warnings found:    {len(warnings)}")
    print()

    if errors:
        print("--- ERRORS (factually wrong at final analysis) ---\n")
        for f in errors:
            print(f"  [{f.severity}] {f.exp_id} / {f.field}")
            print(f"    Claim:        {f.claim}")
            print(f"    Ground truth: {f.ground_truth}")
            print(f"    Context:      {f.context}")
            print()

    if warnings:
        print("--- WARNINGS (misleading or stale data) ---\n")
        for f in warnings:
            print(f"  [{f.severity}] {f.exp_id} / {f.field}")
            print(f"    Claim:        {f.claim}")
            print(f"    Ground truth: {f.ground_truth}")
            print(f"    Context:      {f.context}")
            print()

    print("--- CORRECTED EXPERIMENT SUMMARY ---\n")
    sorted_gt = sorted(
        ground_truth.values(),
        key=lambda g: (
            g.best_finish_time_s
            if g.best_finish_time_s and g.best_finish_time_s > 0
            else float("inf")
        ),
    )
    print(
        f"  {'Experiment':<45} {'Best Time':>10} {'Worker Best':>12} {'Finish%':>8} {'Loss Med':>9}"
    )
    print(f"  {'-' * 45} {'-' * 10} {'-' * 12} {'-' * 8} {'-' * 9}")
    for gt in sorted_gt:
        ft = f"{gt.best_finish_time_s:.2f}s" if gt.best_finish_time_s else "DNF"
        wft = f"{gt.worker_best_finish_time_s:.2f}s" if gt.worker_best_finish_time_s else "-"
        fr = f"{gt.worker_finish_rate * 100:.1f}%" if gt.worker_finish_rate else "0%"
        lm = f"{gt.loss_median:.1f}" if gt.loss_median is not None else "-"
        print(f"  {gt.exp_id:<45} {ft:>10} {wft:>12} {fr:>8} {lm:>9}")

    audit_output = {
        "sections_analyzed": len(sections),
        "errors": len(errors),
        "warnings": len(warnings),
        "findings": [
            {
                "exp_id": f.exp_id,
                "severity": f.severity,
                "field": f.field,
                "claim": f.claim,
                "ground_truth": f.ground_truth,
                "context": f.context,
            }
            for f in all_findings
        ],
    }
    output_path = EXPERIMENTS_DIR / "audit_report.json"
    output_path.write_text(json.dumps(audit_output, indent=2), encoding="utf-8")
    print(f"\nAudit report saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
