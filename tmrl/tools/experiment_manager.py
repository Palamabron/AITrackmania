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
import shutil
import sys
from typing import Any

from tmrl.tools._briefing import cmd_briefing
from tmrl.tools._exp_config_utils import (
    AGENT_CONTEXT_PATH,
    ANALYSIS_DIR,
    CONFIGS_DIR,
    DECISIONS_PATH,
    EXPERIMENTS_LOGS_DIR,
    _orch_defaults,
)
from tmrl.tools._experiment_io import (
    read_registry as _read_registry,
)
from tmrl.tools._experiment_io import (
    write_registry as _write_registry,
)
from tmrl.tools._wandb_analyze import cmd_analyze, cmd_register, cmd_status
from tmrl.tools._wandb_snapshot import cmd_compare, cmd_snapshot, cmd_suggest


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
