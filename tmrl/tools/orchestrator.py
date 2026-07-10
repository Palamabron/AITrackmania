"""Autonomous experiment orchestrator.

Runs a loop: launch experiment -> monitor via W&B -> invoke Cursor agent
for stop/continue decisions -> teardown -> propose next experiment -> repeat.

Usage:
    uv run python -m tmrl.tools.orchestrator
    uv run python -m tmrl.tools.orchestrator --exp-id EXP001   # start from specific experiment

On startup, runs ``experiment_manager reset incomplete --yes`` before the loop.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time

from tmrl.tools._experiment_io import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
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
from tmrl.tools._gemini_client import _make_decision
from tmrl.tools._orchestrator_utils import (
    _capture_git_hash,
    _clean_stale_checkpoint,
    _create_experiment_branch,
    _detect_uv_env,
    _free_distributed_ports,
    _get_base_branch,
    _get_next_planned_experiment,
    _load_config,
    _log,
    _rollback_to_branch,
)
from tmrl.tools._process_manager import ProcessManager
from tmrl.tools._snapshot_analyze import (
    _append_decision_log,
    _reset_incomplete,
    _run_analyze,
    _run_snapshot,
    _validate_wandb_project,
)


def run_experiment_loop(start_exp_id: str | None = None) -> None:
    """Run the autonomous experiment loop.

    Loads orchestrator config, then repeatedly: picks the next planned
    experiment from the registry, launches the three TMRL subprocesses via
    :class:`ProcessManager`, runs smoke checks, monitors training via W&B
    snapshots, applies tier-based budget extensions, invokes the Gemini agent
    for stop/continue decisions, tears down processes, and writes the final
    status back to the registry.  Stops after ``max_consecutive_failures``
    back-to-back failures or when no more planned experiments remain and the
    agent cannot propose new ones.

    Args:
        start_exp_id: Resume from this specific experiment ID.  When given,
            the ``reset incomplete`` pre-flight step is skipped and the loop
            starts by looking up this ID in the registry rather than taking
            the first ``'planned'`` entry.
    """
    _load_dotenv()
    cfg = _load_config()

    target_time = cfg.get("target_finish_time_s", 36.0)
    smoke_check_min = cfg.get("smoke_check_minutes", 10)
    check_interval_min = cfg.get("check_interval_minutes", 60)
    base_max_hours = cfg.get("base_max_hours", 4)
    duration_tiers = cfg.get(
        "duration_tiers",
        [
            {"threshold_s": 60.0, "max_hours": 8},
            {"threshold_s": 45.0, "max_hours": 16},
            {"threshold_s": 40.0, "max_hours": 24},
        ],
    )
    duration_tiers.sort(key=lambda t: t["threshold_s"], reverse=True)
    max_failures = cfg.get("max_consecutive_failures", 3)
    entity = cfg.get("wandb_entity", "dsc-pjatk-warsaw")
    project = cfg.get("wandb_project", "tmrl")
    server_port = cfg.get("server_port", 55555)
    uv_env = cfg.get("uv_env", "") or _detect_uv_env()
    _log(
        f"Config: target={target_time}s, base_max={base_max_hours}h, "
        f"tiers={duration_tiers}, uv_env={uv_env}"
    )

    _validate_wandb_project(entity, project)

    if start_exp_id:
        _log(f"Explicit --exp-id={start_exp_id}, skipping reset incomplete.")
    else:
        _reset_incomplete(uv_env)
    _free_distributed_ports(server_port)

    consecutive_failures = 0

    while True:
        pm: ProcessManager | None = None
        exp_id: str = ""
        base_branch: str | None = None
        code_patches: list[dict] | None = None
        try:
            if start_exp_id:
                exp_entry = None
                for e in _read_registry():
                    if e.get("exp_id") == start_exp_id:
                        exp_entry = e
                        break
                start_exp_id = None
            else:
                exp_entry = _get_next_planned_experiment()

            if not exp_entry:
                _log("No planned experiments. Invoking agent to propose next...")
                all_entries = _read_registry()
                agent_result = _make_decision(
                    "propose",
                    {
                        "registry": all_entries,
                        "target_finish_time_s": target_time,
                    },
                )

                if agent_result.get("action") == "no_proposal":
                    _log(f"Agent could not propose: {agent_result.get('reason')}")
                    _log("Orchestrator stopping. Register experiments manually to continue.")
                    break

                _log("Agent proposed next experiment, re-checking registry...")
                exp_entry = _get_next_planned_experiment()
                if not exp_entry:
                    _log("No new planned experiment after agent proposal. Stopping.")
                    break

            exp_id = exp_entry["exp_id"]
            overrides = exp_entry.get("config_overrides", {})
            code_patches = exp_entry.get("code_patches") or None
            _log(f"{'=' * 60}")
            _log(f"Starting experiment: {exp_id}")
            _log(f"  Hypothesis: {exp_entry.get('hypothesis', 'N/A')}")
            _log(f"  Overrides: {json.dumps(overrides, indent=2)}")
            if code_patches:
                _log(f"  Code patches: {len(code_patches)} file(s)")
            _log(f"{'=' * 60}")

            _log("Cleaning stale checkpoints/weights for this exp_id...")
            _clean_stale_checkpoint(exp_id)

            git = _capture_git_hash()
            if git:
                commit = str(git.get("commit", "?"))[:10]
                branch = str(git.get("branch", "?"))
                dirty = " [dirty]" if git.get("dirty") else ""
                _log(f"  Git: {commit} ({branch}){dirty}")

            exp_commit: str | None = None
            if code_patches:
                base_branch = _get_base_branch()
                _log(f"  Creating experiment branch from {base_branch}...")
                exp_commit = _create_experiment_branch(exp_id, code_patches)
                if exp_commit is None:
                    _log("  Code patch application failed. Skipping experiment.")
                    _update_registry_entry(
                        exp_id,
                        {
                            "status": "failed",
                            "stop_reason": "code_patch_validation_failed",
                        },
                    )
                    _append_decision_log(exp_id, "failed", "Code patch validation failed")
                    base_branch = None
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        _log(f"Hit {max_failures} consecutive failures. Stopping.")
                        break
                    continue

            _update_registry_entry(
                exp_id,
                {
                    "status": "running",
                    "wandb_run_id": exp_id,
                    "git": git,
                    **(
                        {"git_branch": f"exp/{exp_id}", "git_base_commit": exp_commit}
                        if exp_commit
                        else {}
                    ),
                },
            )

            pm = ProcessManager(exp_id, overrides, uv_env, server_port)
            try:
                pm.start()
            except Exception as exc:
                _log(f"Failed to start processes: {exc}")
                _update_registry_entry(exp_id, {"status": "failed", "stop_reason": str(exc)})
                _append_decision_log(exp_id, "failed", f"Process start failed: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    _log(f"Hit {max_failures} consecutive failures. Stopping orchestrator.")
                    break
                continue

            _log(f"Waiting {smoke_check_min} min for smoke check...")
            time.sleep(smoke_check_min * 60)

            if not pm.all_alive():
                status = pm.status_summary()
                _log(f"SMOKE CHECK FAILED: {status}")
                pm.stop()
                _update_registry_entry(
                    exp_id,
                    {
                        "status": "failed",
                        "stop_reason": f"Smoke check failed: {status}",
                        "stopped_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    },
                )
                _append_decision_log(exp_id, "failed", f"Smoke check: processes died: {status}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    _log(f"Hit {max_failures} consecutive failures. Stopping orchestrator.")
                    break
                continue

            trainer_ok = pm.trainer_is_active()
            samples_ok = not pm.worker_is_sending_samples() or pm.trainer_is_receiving_samples()
            if pm.worker_is_sending_samples() and not pm.trainer_is_receiving_samples():
                _log(
                    "WARNING: Worker sends rollouts but trainer still has 0 samples "
                    "(worker→server→trainer pipeline broken)."
                )
            if not trainer_ok or not samples_ok:
                _log("WARNING: Trainer not healthy yet. Waiting 3 more min...")
                time.sleep(180)
                trainer_ok = pm.trainer_is_active()
                samples_ok = not pm.worker_is_sending_samples() or pm.trainer_is_receiving_samples()

            max_trainer_retries = 2
            trainer_retries = 0
            while (not trainer_ok or not samples_ok) and trainer_retries < max_trainer_retries:
                trainer_retries += 1

                _log(
                    f"Pipeline stuck. Full restart"
                    f" (attempt {trainer_retries}/{max_trainer_retries})..."
                )
                alive = pm.restart_role("trainer")
                if not alive:
                    _log("Full pipeline restart failed.")
                    break
                _log(f"Waiting {smoke_check_min} min after pipeline restart...")
                time.sleep(smoke_check_min * 60)
                trainer_ok = pm.trainer_is_active()
                samples_ok = not pm.worker_is_sending_samples() or pm.trainer_is_receiving_samples()
                if (not trainer_ok or not samples_ok) and trainer_retries < max_trainer_retries:
                    _log("Trainer still not receiving samples after restart, will retry...")

            if not trainer_ok or not samples_ok:
                reason = (
                    "Trainer not receiving worker samples (check ports 55555-55558, "
                    "no duplicate trainer process)"
                    if pm.worker_is_sending_samples() and not pm.trainer_is_receiving_samples()
                    else "Trainer stuck during initialization after retries"
                )
                _log(f"SMOKE CHECK FAILED: {reason}")
                pm.stop()
                _update_registry_entry(
                    exp_id,
                    {
                        "status": "failed",
                        "stop_reason": reason,
                        "stopped_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    },
                )
                _append_decision_log(exp_id, "failed", reason)
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    _log(f"Hit {max_failures} consecutive failures. Stopping orchestrator.")
                    break
                continue

            _log("Smoke check passed. All processes alive and trainer active.")
            consecutive_failures = 0

            exp_start = time.time()
            experiment_done = False
            final_status = "completed"
            stop_reason = "max_duration_reached"
            current_max_hours = base_max_hours
            current_tier_idx = -1
            consecutive_snapshot_failures = 0

            while not experiment_done:
                _log(f"Sleeping {check_interval_min} min until next check...")
                time.sleep(check_interval_min * 60)

                elapsed_h = (time.time() - exp_start) / 3600
                _log(
                    f"Check at {elapsed_h:.1f}h elapsed"
                    f" (max={current_max_hours}h, tier={current_tier_idx})"
                )

                if not pm.all_alive():
                    status = pm.status_summary()
                    _log(f"Process(es) died during training: {status}")

                    if (
                        not pm.worker_is_alive()
                        and pm.processes.get("server")
                        and pm.processes["server"].poll() is None
                    ):
                        _log("Worker died mid-training. Attempting restart...")
                        if pm.restart_role("worker"):
                            _log("Worker restarted successfully. Continuing experiment.")
                        else:
                            _log("Worker restart failed.")
                            final_status = "failed"
                            stop_reason = f"Worker crash mid-training (restart failed): {status}"
                            experiment_done = True
                            break
                    elif pm.processes.get("trainer") and pm.processes["trainer"].poll() is not None:
                        _log("Trainer died mid-training. Attempting restart...")
                        if pm.restart_role("trainer"):
                            _log("Trainer restarted successfully. Continuing experiment.")
                        else:
                            _log("Trainer restart failed.")
                            final_status = "failed"
                            stop_reason = f"Trainer crash mid-training (restart failed): {status}"
                            experiment_done = True
                            break
                    else:
                        final_status = "failed"
                        stop_reason = f"Process crash mid-training: {status}"
                        experiment_done = True
                        break

                if elapsed_h >= current_max_hours:
                    _log(f"Max duration ({current_max_hours}h) reached.")
                    experiment_done = True
                    break

                snapshot = _run_snapshot(exp_id, entity, project)
                if not snapshot:
                    consecutive_snapshot_failures += 1
                    _log(
                        f"Could not get snapshot "
                        f"({consecutive_snapshot_failures} consecutive failures), "
                        f"will retry next interval."
                    )
                    if consecutive_snapshot_failures >= 5:
                        _log(
                            "WARNING: 5 consecutive snapshot failures. "
                            "Continuing experiment but decisions are blind."
                        )
                    continue
                consecutive_snapshot_failures = 0

                best_ft = snapshot.get("best_finish_time_s")
                if best_ft is not None and best_ft > 0 and best_ft <= target_time:
                    _log(f"TARGET REACHED: {best_ft:.2f}s <= {target_time}s")
                    final_status = "completed"
                    stop_reason = f"Target reached: {best_ft:.2f}s"
                    experiment_done = True
                    break

                if best_ft is not None and best_ft > 0:
                    for i, tier in enumerate(duration_tiers):
                        if i <= current_tier_idx:
                            continue
                        if best_ft <= tier["threshold_s"]:
                            old_max = current_max_hours
                            current_max_hours = tier["max_hours"]
                            current_tier_idx = i
                            _log(
                                f"TIER UP: {best_ft:.2f}s <= {tier['threshold_s']}s. "
                                f"Extended from {old_max}h to {current_max_hours}h."
                            )
                            _append_decision_log(
                                exp_id,
                                "extended",
                                f"Reached {best_ft:.2f}s (<= {tier['threshold_s']}s), "
                                f"extended from {old_max}h to {current_max_hours}h",
                            )

                agent_context = {
                    "snapshot": snapshot,
                    "target_finish_time_s": target_time,
                    "current_max_hours": current_max_hours,
                    "current_tier": duration_tiers[current_tier_idx]
                    if current_tier_idx >= 0
                    else None,
                    "elapsed_hours": elapsed_h,
                    "exp_entry": exp_entry,
                    "registry": _read_registry(),
                }
                try:
                    decision = _make_decision("decide", agent_context)
                except Exception as exc:
                    _log(f"Decision error ({exc!r}), defaulting to continue")
                    decision = {"action": "continue", "reason": f"Decision error: {exc}"}

                action = decision.get("action", "continue")
                reason = decision.get("reason", "no reason")

                _log(f"Agent decision: {action} -- {reason}")
                _append_decision_log(exp_id, action, reason)

                if action == "stop":
                    final_status = "stopped_early"
                    stop_reason = reason
                    experiment_done = True

            pm.stop()
            pm = None

            if base_branch and code_patches:
                _rollback_to_branch(base_branch)
                base_branch = None

            now = datetime.datetime.now(datetime.UTC).isoformat()
            _update_registry_entry(
                exp_id,
                {
                    "status": final_status,
                    "stop_reason": stop_reason,
                    "stopped_at": now,
                },
            )
            _log(f"Experiment {exp_id} finished: {final_status} -- {stop_reason}")

            _log("Running post-experiment analysis...")
            _run_analyze(exp_id, entity, project)

            _log("Running post-experiment validation...")
            try:
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/validate_decisions.py",
                        "--json-out",
                        str(EXPERIMENTS_DIR / "validation_report.json"),
                    ],
                    timeout=60,
                    cwd=str(REPO_ROOT),
                    check=False,
                )
            except Exception as exc:
                _log(f"Validation script error: {exc}")

            if final_status == "failed":
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    _log(f"Hit {max_failures} consecutive failures. Stopping.")
                    break
            else:
                consecutive_failures = 0

        except KeyboardInterrupt:
            _log("KeyboardInterrupt received. Cleaning up...")
            if pm is not None:
                pm.stop()
            if base_branch and code_patches:
                _rollback_to_branch(base_branch)
            if exp_id:
                _update_registry_entry(
                    exp_id,
                    {
                        "status": "failed",
                        "stop_reason": "KeyboardInterrupt",
                        "stopped_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    },
                )
            break
        except Exception as exc:
            _log(f"UNEXPECTED ERROR in main loop: {exc!r}")
            if pm is not None:
                try:
                    pm.stop()
                except Exception as stop_exc:
                    _log(f"Error stopping processes during cleanup: {stop_exc}")
            if base_branch and code_patches:
                _rollback_to_branch(base_branch)
                base_branch = None
            if exp_id:
                try:
                    _update_registry_entry(
                        exp_id,
                        {
                            "status": "failed",
                            "stop_reason": f"Unexpected error: {exc}",
                            "stopped_at": datetime.datetime.now(datetime.UTC).isoformat(),
                        },
                    )
                    _append_decision_log(exp_id, "failed", f"Unexpected error: {exc}")
                except Exception:
                    pass
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                _log(f"Hit {max_failures} consecutive failures after unexpected error. Stopping.")
                break
            _log("Attempting to continue with next experiment...")
            continue

    _log("Orchestrator loop ended.")


def main() -> None:
    """Parse CLI arguments and enter the orchestrator loop."""
    import argparse

    parser = argparse.ArgumentParser(description="TMRL Autonomous Experiment Orchestrator")
    parser.add_argument("--exp-id", default=None, help="Start from a specific experiment ID")
    args = parser.parse_args()
    run_experiment_loop(start_exp_id=args.exp_id)


if __name__ == "__main__":
    main()
