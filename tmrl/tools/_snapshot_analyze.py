"""Snapshot, analyze, and logging helpers for the orchestrator."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from tmrl.tools._experiment_io import EXPERIMENTS_DIR, REPO_ROOT
from tmrl.tools._orchestrator_utils import _log


def _extract_json_from_output(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from mixed stdout (logs + JSON)."""
    brace_depth = 0
    json_start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and json_start is not None:
                try:
                    parsed: dict[str, Any] = json.loads(text[json_start : i + 1])
                    return parsed
                except json.JSONDecodeError:
                    json_start = None
    return None


def _subprocess_env(uv_env: str) -> dict[str, str]:
    env = dict(os.environ)
    if uv_env:
        env["UV_PROJECT_ENVIRONMENT"] = uv_env
    env.pop("VIRTUAL_ENV", None)
    return env


def _reset_incomplete(uv_env: str) -> None:
    """Drop unfinished experiments from registry before starting the loop."""
    _log("Resetting incomplete experiments...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "tmrl.tools.experiment_manager",
            "reset",
            "incomplete",
            "--yes",
        ],
        cwd=str(REPO_ROOT),
        env=_subprocess_env(uv_env),
        check=False,
    )
    if result.returncode != 0:
        _log(f"WARNING: reset incomplete failed (rc={result.returncode})")
    else:
        _log("Reset incomplete completed.")


def _run_snapshot(
    exp_id: str, entity: str, project: str, *, retries: int = 3
) -> dict[str, Any] | None:
    """Call experiment_manager snapshot and return parsed JSON.

    Retries up to *retries* times with exponential backoff on transient failures.
    """
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "tmrl.tools.experiment_manager",
                    "snapshot",
                    "--exp-id",
                    exp_id,
                    "--entity",
                    entity,
                    "--project",
                    project,
                ],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(REPO_ROOT),
            )
            if result.returncode == 0:
                parsed = _extract_json_from_output(result.stdout)
                if parsed:
                    if parsed.get("error"):
                        _log(f"Snapshot returned error field: {parsed['error']}")
                    return parsed
                last_err = f"no valid JSON in output ({len(result.stdout)} chars)"
            else:
                last_err = f"rc={result.returncode}: {result.stderr[:300]}"
        except subprocess.TimeoutExpired:
            last_err = "subprocess timed out (180s)"
        except Exception as exc:
            last_err = str(exc)

        if attempt < retries:
            delay = 10 * (2 ** (attempt - 1))
            _log(
                f"Snapshot attempt {attempt}/{retries} failed ({last_err}), retrying in {delay}s..."
            )
            time.sleep(delay)

    _log(f"Snapshot failed after {retries} attempts: {last_err}")
    return None


def _run_analyze(exp_id: str, entity: str, project: str, *, retries: int = 2) -> None:
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "tmrl.tools.experiment_manager",
                    "analyze",
                    "--exp-id",
                    exp_id,
                    "--entity",
                    entity,
                    "--project",
                    project,
                ],
                timeout=300,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                _log(f"Analyze completed for {exp_id}")
                return
            _log(f"Analyze failed (rc={result.returncode}): {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            _log(f"Analyze timed out for {exp_id} (300s)")
        except Exception as exc:
            _log(f"Analyze error: {exc}")

        if attempt < retries:
            delay = 15 * attempt
            _log(f"Retrying analyze in {delay}s (attempt {attempt}/{retries})...")
            time.sleep(delay)

    _log(f"Analyze failed after {retries} attempts for {exp_id}")


def _append_decision_log(exp_id: str, action: str, reason: str) -> None:
    decisions_path = EXPERIMENTS_DIR / "decisions.md"
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n### {ts} -- {exp_id}\n\n**Action:** {action}\n**Reason:** {reason}\n"
    try:
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("a", encoding="utf-8") as f:
            f.write(entry)
    except OSError as exc:
        _log(f"WARNING: Could not write to decisions log: {exc}")


def _validate_wandb_project(orch_entity: str, orch_project: str) -> None:
    """Warn loudly if orchestrator W&B project diverges from TMRL's local.yaml."""
    tmrl_local = Path.home() / "TmrlData" / "config" / "local.yaml"
    if not tmrl_local.exists():
        return
    try:
        with tmrl_local.open(encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        tmrl_project = local_cfg.get("wandb", {}).get("project", "")
        tmrl_entity = local_cfg.get("wandb", {}).get("entity", "")
        if tmrl_project and tmrl_project != orch_project:
            _log(
                f"FATAL: W&B project mismatch! "
                f"orchestrator_config.yaml says '{orch_project}' but "
                f"TmrlData/config/local.yaml says '{tmrl_project}'. "
                f"Snapshot queries will fail. Fix orchestrator_config.yaml."
            )
            raise SystemExit(1)
        if tmrl_entity and tmrl_entity != orch_entity:
            _log(
                f"WARNING: W&B entity mismatch: orchestrator='{orch_entity}', "
                f"TmrlData/config/local.yaml='{tmrl_entity}'"
            )
    except SystemExit:
        raise
    except Exception as exc:
        _log(f"WARNING: Could not validate W&B config against local.yaml: {exc}")
