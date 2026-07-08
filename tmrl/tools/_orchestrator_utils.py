"""Orchestrator utility functions and module-level constants."""

from __future__ import annotations

import contextlib
import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from tmrl.tools._experiment_io import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
)
from tmrl.tools._experiment_io import (
    read_registry as _read_registry,
)

ORCHESTRATOR_CONFIG_PATH = EXPERIMENTS_DIR / "orchestrator_config.yaml"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"
LOGS_DIR = EXPERIMENTS_DIR / "logs"


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[orchestrator {ts}] {msg}", flush=True)


def _load_config() -> dict[str, Any]:
    try:
        with ORCHESTRATOR_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                return cfg
            _log("WARNING: orchestrator_config.yaml did not parse as dict, using defaults")
            return {}
    except Exception as exc:
        _log(f"WARNING: Could not load orchestrator config ({exc}), using defaults")
        return {}


def _get_next_planned_experiment() -> dict[str, Any] | None:
    try:
        for e in _read_registry():
            if e.get("status") == "planned":
                return e
    except Exception as exc:
        _log(f"WARNING: Error reading registry for next experiment: {exc}")
    return None


def _detect_uv_env() -> str:
    """Auto-detect the UV venv directory, matching Makefile logic."""
    if sys.platform == "win32":
        if (REPO_ROOT / ".venv-windows").exists():
            return ".venv-windows"
        if (REPO_ROOT / ".venv").exists():
            return ".venv"
        return ".venv-windows"
    else:
        if (REPO_ROOT / ".venv-linux").exists():
            return ".venv-linux"
        if (REPO_ROOT / ".venv").exists():
            return ".venv"
        return ".venv"


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (avoids zombie child processes on Windows)."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                timeout=15,
                capture_output=True,
            )
        except Exception as exc:
            _log(f"  taskkill warning: {exc}")
    else:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _free_distributed_ports(server_port: int = 55555) -> None:
    """Free server + tlspyo local ports (server, trainer, worker)."""
    for port in range(server_port, server_port + 4):
        _kill_port(port)


def _kill_port(port: int) -> None:
    """Kill any process holding the given TCP port (same as Makefile kill-server)."""
    kill_script = REPO_ROOT / "scripts" / "platform" / "kill_tcp_port.ps1"
    if sys.platform == "win32" and kill_script.exists():
        _log(f"Freeing port {port} (kill_tcp_port.ps1)...")
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(kill_script),
                    "-Port",
                    str(port),
                ],
                timeout=15,
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            _log(f"  kill_port warning: {exc}")
    else:
        _log(f"Freeing port {port} (lsof)...")
        try:
            subprocess.run(
                [
                    "sh",
                    "-c",
                    f"pids=$(lsof -ti:{port} 2>/dev/null); "
                    f'[ -n "$pids" ] && kill -9 $pids 2>/dev/null || true',
                ],
                timeout=10,
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            _log(f"  kill_port warning: {exc}")
    time.sleep(1)


def _clean_stale_checkpoint(exp_id: str) -> None:
    """Delete checkpoint and weight files from a prior run with the same name.

    Checkpoint/weight paths are derived from ``run.name`` under ~/TmrlData.
    If the orchestrator sets ``run.name = exp_id``, a previous crash or rerun
    can leave stale ``.tcpt`` / ``.tmod`` files that cause the trainer to resume
    instead of starting fresh.
    """
    tmrl_folder = Path.home() / "TmrlData"
    ckpt_dir = tmrl_folder / "checkpoints"
    weights_dir = tmrl_folder / "weights"
    patterns = [
        (ckpt_dir, f"{exp_id}_t.tcpt"),
        (ckpt_dir, f"{exp_id}_rew_*_t.tcpt"),
        (weights_dir, f"{exp_id}.tmod"),
        (weights_dir, f"{exp_id}_t.tmod"),
        (weights_dir, f"{exp_id}_*"),
    ]
    for folder, pat in patterns:
        if not folder.exists():
            continue
        for f in folder.glob(pat):
            try:
                f.unlink()
                _log(f"  Deleted stale file: {f}")
            except OSError as exc:
                _log(f"  WARNING: could not delete {f}: {exc}")


def _capture_git_hash() -> dict[str, str | bool]:
    """Return {commit, branch, dirty} for the repo, or empty dict on failure."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            timeout=10,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            timeout=10,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            text=True,
            timeout=10,
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": bool(status)}
    except Exception:
        return {}


def _get_base_branch() -> str:
    """Return the current branch name (used to restore after experiment)."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            timeout=10,
        ).strip()
        return branch or "main"
    except Exception:
        return "main"


def _validate_code_changes(files: list[str]) -> tuple[bool, str]:
    """Validate changed Python files via py_compile + import smoke test.

    Returns:
        (ok, error_message) — ok is True if all checks pass.
    """
    import py_compile

    for fpath in files:
        if not fpath.endswith(".py"):
            continue
        abs_path = REPO_ROOT / fpath if not os.path.isabs(fpath) else Path(fpath)
        if not abs_path.exists():
            return False, f"File does not exist: {abs_path}"

        try:
            py_compile.compile(str(abs_path), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, f"Syntax error in {fpath}: {exc}"

        rel = abs_path.relative_to(REPO_ROOT)
        module_path = str(rel).replace(os.sep, ".").replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module_path}"],
                cwd=str(REPO_ROOT),
                timeout=30,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[:500]
                return False, f"Import failed for {module_path}: {stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Import timed out for {module_path}"
        except Exception as exc:
            return False, f"Import check error for {module_path}: {exc}"

    return True, ""


def _create_experiment_branch(exp_id: str, code_patches: list[dict]) -> str | None:
    """Create ``exp/<exp_id>`` branch, apply file writes, validate, and commit.

    Each patch dict must have keys: ``file`` (relative path) and ``content`` (full file text).
    Optionally ``description`` for the commit message.

    Returns:
        The commit hash if successful, or None if no patches or validation failed.
    """
    if not code_patches:
        return None

    branch_name = f"exp/{exp_id}"
    try:
        subprocess.check_call(
            ["git", "checkout", "-b", branch_name],
            cwd=str(REPO_ROOT),
            timeout=15,
        )
    except subprocess.CalledProcessError as exc:
        _log(f"  Failed to create branch {branch_name}: {exc}")
        return None

    changed_files: list[str] = []
    descriptions: list[str] = []
    for patch in code_patches:
        fpath = patch.get("file", "")
        content = patch.get("content", "")
        desc = patch.get("description", fpath)
        if not fpath:
            continue
        abs_path = REPO_ROOT / fpath
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        changed_files.append(fpath)
        descriptions.append(desc)

    if not changed_files:
        _log("  No files written by code_patches. Rolling back branch.")
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        return None

    ok, err = _validate_code_changes(changed_files)
    if not ok:
        _log(f"  Code validation FAILED: {err}")
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        return None

    try:
        subprocess.check_call(
            ["git", "add", *changed_files],
            cwd=str(REPO_ROOT),
            timeout=15,
        )
        commit_msg = f"exp({exp_id}): {'; '.join(descriptions[:3])}"
        subprocess.check_call(
            ["git", "commit", "-m", commit_msg],
            cwd=str(REPO_ROOT),
            timeout=15,
        )
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            timeout=10,
        ).strip()
        _log(f"  Committed code patches on branch {branch_name}: {commit_hash[:10]}")
        return commit_hash
    except subprocess.CalledProcessError as exc:
        _log(f"  Git commit failed: {exc}")
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(REPO_ROOT),
            timeout=15,
            check=False,
        )
        return None


def _rollback_to_branch(branch: str) -> None:
    """Check out the given branch (called in finally block after experiment)."""
    try:
        subprocess.check_call(
            ["git", "checkout", branch],
            cwd=str(REPO_ROOT),
            timeout=15,
        )
        _log(f"  Rolled back to branch: {branch}")
    except Exception as exc:
        _log(f"  WARNING: rollback to branch {branch} failed: {exc}")
