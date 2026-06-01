"""Autonomous experiment orchestrator.

Runs a loop: launch experiment -> monitor via W&B -> invoke Cursor agent
for stop/continue decisions -> teardown -> propose next experiment -> repeat.

Usage:
    uv run python -m tmrl.tools.orchestrator
    uv run python -m tmrl.tools.orchestrator --exp-id EXP001   # start from specific experiment

On startup, runs ``experiment_manager reset incomplete --yes`` before the loop.
"""

from __future__ import annotations

import contextlib
import datetime
import json
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
    load_dotenv as _load_dotenv,
)
from tmrl.tools._experiment_io import (
    read_registry as _read_registry,
)
from tmrl.tools._experiment_io import (
    update_registry_entry as _update_registry_entry,
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


# ---------------------------------------------------------------------------
# Git branch management for code-patching experiments
# ---------------------------------------------------------------------------


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


class ProcessManager:
    """Manages the three TMRL subprocesses: server, trainer, worker."""

    def __init__(
        self, exp_id: str, config_overrides: dict[str, Any], uv_env: str, server_port: int = 55555
    ):
        self.exp_id = exp_id
        self.config_overrides = config_overrides
        self.uv_env = uv_env
        self.server_port = server_port
        self.processes: dict[str, subprocess.Popen] = {}
        self._log_dir = LOGS_DIR / exp_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_files: dict[str, Any] = {}

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.uv_env:
            env["UV_PROJECT_ENVIRONMENT"] = self.uv_env
        # Remove stale VIRTUAL_ENV from parent (WSL .venv vs Windows .venv-windows)
        env.pop("VIRTUAL_ENV", None)
        env["TMRL_EXPERIMENT_ID"] = self.exp_id
        overrides: dict[str, Any] = dict(self.config_overrides)
        overrides.setdefault("run", {})["name"] = self.exp_id
        # Force reset_training so no stale checkpoint triggers wandb resume
        overrides.setdefault("run", {})["reset_training"] = True
        env["TMRL_CONFIG_OVERRIDES"] = json.dumps(overrides, default=str)
        # Prevent wandb.init() from hanging indefinitely
        env["WANDB__SERVICE_WAIT"] = "60"
        env["WANDB_INIT_TIMEOUT"] = "60"
        # On Windows, the default W&B service start method can hang in subprocesses
        env.setdefault("WANDB_START_METHOD", "thread")
        # Avoid UnicodeEncodeError from loguru on cp1250 consoles
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _wait_for_port(self, timeout: int = 60) -> bool:
        """Wait until self.server_port is listening (server is ready)."""
        import socket

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.processes["server"].poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", self.server_port), timeout=2):
                    return True
            except (ConnectionRefusedError, OSError, TimeoutError):
                time.sleep(2)
        return False

    def start(self) -> None:
        _free_distributed_ports(self.server_port)

        # Clear old logs for this experiment so we don't mix runs
        for old_log in self._log_dir.glob("*.log"):
            old_log.write_text("", encoding="utf-8")

        env = self._build_env()
        _log(f"Using UV_PROJECT_ENVIRONMENT={self.uv_env}")

        # --- Start server and wait for port ---
        self._start_role("server", ["uv", "run", "python", "-m", "tmrl", "--server"], env)
        _log(f"Waiting for server to listen on port {self.server_port}...")
        if not self._wait_for_port(timeout=90):
            if self.processes["server"].poll() is not None:
                raise RuntimeError(
                    f"Server died (rc={self.processes['server'].returncode}). "
                    f"Check {self._log_dir / 'server_stderr.log'}"
                )
            raise RuntimeError(
                f"Server not listening on port {self.server_port} after 90s. "
                f"Check {self._log_dir / 'server_stderr.log'}"
            )
        _log(f"Server is listening on port {self.server_port}.")

        # --- Start trainer and verify alive ---
        self._start_role("trainer", ["uv", "run", "python", "-m", "tmrl", "--trainer"], env)
        time.sleep(15)
        if self.processes["trainer"].poll() is not None:
            raise RuntimeError(
                f"Trainer died immediately (rc={self.processes['trainer'].returncode}). "
                f"Check {self._log_dir / 'trainer_stderr.log'}"
            )
        _log("Trainer is alive.")

        # --- Start worker ---
        self._start_role("worker", ["uv", "run", "python", "-m", "tmrl", "--worker"], env)
        time.sleep(5)
        if self.processes["worker"].poll() is not None:
            raise RuntimeError(
                f"Worker died immediately (rc={self.processes['worker'].returncode}). "
                f"Check {self._log_dir / 'worker_stderr.log'}"
            )
        _log("Worker is alive. All processes started.")

    def _start_role(self, role: str, args: list[str], env: dict[str, str]) -> None:
        stdout_f = (self._log_dir / f"{role}_stdout.log").open("a", encoding="utf-8")
        stderr_f = (self._log_dir / f"{role}_stderr.log").open("a", encoding="utf-8")
        self._log_files[f"{role}_stdout"] = stdout_f
        self._log_files[f"{role}_stderr"] = stderr_f

        _log(f"Starting {role} for {self.exp_id}...")
        kwargs: dict[str, Any] = {
            "stdout": stdout_f,
            "stderr": stderr_f,
            "env": env,
            "cwd": str(REPO_ROOT),
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        self.processes[role] = subprocess.Popen(args, **kwargs)
        _log(f"  {role} PID={self.processes[role].pid}")

    def all_alive(self) -> bool:
        return all(p.poll() is None for p in self.processes.values())

    def _kill_role(self, role: str) -> None:
        """Kill a single subprocess and close its log handles."""
        proc = self.processes.get(role)
        if proc and proc.poll() is None:
            _log(f"  Killing {role} (PID={proc.pid})...")
            _kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
            except OSError:
                pass

        for suffix in ("stdout", "stderr"):
            key = f"{role}_{suffix}"
            fh = self._log_files.get(key)
            if fh:
                with contextlib.suppress(Exception):
                    fh.close()

        for logf in (f"{role}_stdout.log", f"{role}_stderr.log"):
            p = self._log_dir / logf
            if p.exists():
                p.write_text("", encoding="utf-8")

        local_port_offsets = {"server": 1, "trainer": 2, "worker": 3}
        if role in local_port_offsets:
            _kill_port(self.server_port + local_port_offsets[role])

    def restart_role(self, role: str) -> bool:
        """Kill and restart a single subprocess (e.g. 'trainer').

        When restarting the trainer, the server is also restarted because
        tlspyo's Relay keeps a stale "trainers" slot for the dead connection,
        causing ``Cannot add more clients to group trainers`` for the new
        trainer.  The worker is also restarted because its tlspyo connection
        to the old server cannot reliably auto-reconnect.
        """
        if role == "trainer":
            _log("  Restarting server+trainer+worker (full pipeline restart)...")
            self._kill_role("worker")
            self._kill_role("trainer")
            self._kill_role("server")
            _kill_port(self.server_port)
            for offset in range(1, 4):
                _kill_port(self.server_port + offset)
            time.sleep(5)

            env = self._build_env()
            self._start_role("server", ["uv", "run", "python", "-m", "tmrl", "--server"], env)
            _log(f"  Waiting for server to listen on port {self.server_port}...")
            if not self._wait_for_port(timeout=90):
                _log("  Server did not come back up after restart.")
                return False
            _log(f"  Server is listening on port {self.server_port}.")

            self._start_role("trainer", ["uv", "run", "python", "-m", "tmrl", "--trainer"], env)
            time.sleep(15)
            if self.processes["trainer"].poll() is not None:
                _log("  Trainer died immediately after restart.")
                return False

            self._start_role("worker", ["uv", "run", "python", "-m", "tmrl", "--worker"], env)
            time.sleep(5)
            if self.processes["worker"].poll() is not None:
                _log("  Worker died immediately after restart.")
                return False
            _log("  All three processes restarted successfully.")
            return True

        self._kill_role(role)
        time.sleep(5)
        env = self._build_env()
        self._start_role(role, ["uv", "run", "python", "-m", "tmrl", f"--{role}"], env)
        time.sleep(15)
        return self.processes[role].poll() is None

    def _read_role_log(self, role: str, stream: str = "stdout") -> str:
        path = self._log_dir / f"{role}_{stream}.log"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def worker_is_sending_samples(self) -> bool:
        return "Sent" in self._read_role_log(
            "worker"
        ) and "sample(s) to server" in self._read_role_log("worker")

    def trainer_is_receiving_samples(self) -> bool:
        text = self._read_role_log("trainer")
        return (
            "Received" in text and "samples from server" in text
        ) or "retrieve_buffer: got" in text

    def trainer_is_active(self, min_stdout_lines: int = 15) -> bool:
        """Check if trainer is actually training (not just alive but stuck on init)."""
        stdout_log = self._log_dir / "trainer_stdout.log"
        stderr_log = self._log_dir / "trainer_stderr.log"
        if not stdout_log.exists():
            return False
        stdout_text = self._read_role_log("trainer")
        stdout_lines = stdout_text.strip().splitlines()
        if "training_step" in stdout_text or " Resuming training" in stdout_text:
            return True
        if self.trainer_is_receiving_samples():
            return True
        if len(stdout_lines) >= min_stdout_lines:
            # Log spam from "Still waiting for samples" must not count as healthy training.
            return not (
                "Still waiting for samples" in stdout_text
                and not self.trainer_is_receiving_samples()
            )
        stderr_text = ""
        if stderr_log.exists():
            stderr_text = stderr_log.read_text(encoding="utf-8", errors="replace")
        return "wandb: Tracking run" in stderr_text

    def worker_is_alive(self) -> bool:
        proc = self.processes.get("worker")
        return proc is not None and proc.poll() is None

    def status_summary(self) -> dict[str, str]:
        return {
            role: "running" if p.poll() is None else f"exited({p.returncode})"
            for role, p in self.processes.items()
        }

    def stop(self) -> None:
        _log(f"Stopping processes for {self.exp_id}...")
        for role in ["worker", "trainer", "server"]:
            proc = self.processes.get(role)
            if proc and proc.poll() is None:
                _log(f"  Terminating {role} (PID={proc.pid})...")
                _kill_process_tree(proc.pid)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _log(f"  Force killing {role}")
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=5)
            time.sleep(2)

        for f in self._log_files.values():
            with contextlib.suppress(Exception):
                f.close()
        _log(f"All processes stopped for {self.exp_id}")


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


def _decide(context: dict[str, Any]) -> dict[str, Any]:
    """Decide whether to continue or stop the current experiment.

    Analyzes training dynamics: loss stability, Q-value health,
    reward trends, and convergence signals.
    """
    snapshot = context.get("snapshot", {})
    recent = snapshot.get("recent_metrics", {})
    target_time = context.get("target_finish_time_s", 36.0)
    elapsed_h = context.get("elapsed_hours", 0)
    best_ft = snapshot.get("best_finish_time_s")

    # --- Target check ---
    if best_ft is not None and best_ft > 0 and best_ft <= target_time:
        return {"action": "stop", "reason": f"Target reached: {best_ft:.2f}s"}

    # --- Worker finishing = learning is happening, keep running ---
    worker_finish_count = snapshot.get("worker_finish_count", 0)
    worker_best = snapshot.get("worker_best_finish_time_s")
    if worker_finish_count >= 5 and worker_best and worker_best > 0:
        return {
            "action": "continue",
            "reason": f"Worker is finishing tracks ({worker_finish_count} finishes, "
            f"best {worker_best:.1f}s). Learning is progressing.",
        }

    # --- Catastrophic failure checks ---
    loss = recent.get("loss/iqn_loss", {})
    loss_last = loss.get("last")
    loss_p95 = loss.get("p95")
    loss_median = loss.get("median")

    if loss_last and loss_last > 100:
        return {"action": "stop", "reason": f"Loss diverged catastrophically: {loss_last:.4g}"}

    q_max = recent.get("q/max_q", {})
    q_max_last = q_max.get("last")
    if q_max_last and abs(q_max_last) > 500:
        return {"action": "stop", "reason": f"Q-values exploded: {q_max_last:.2f}"}

    q_min = recent.get("q/min_q", {})
    q_min_last = q_min.get("last")
    if q_min_last is not None and q_min_last < -50:
        return {"action": "stop", "reason": f"Q-values collapsed (min_q={q_min_last:.2f})"}

    # NOTE: Gradient saturation (pre-clip >> clip) is structural for this
    # architecture and must NOT be used as a stop criterion.  All 18 past
    # experiments showed pre-clip/clip ratios >20x; stopping on this wasted
    # 11 experiments.  Only NaN gradients warrant a stop.
    grad = recent.get("debug/grad_norm", {})
    grad_last = grad.get("last")
    if grad_last is not None and (grad_last != grad_last):  # NaN check
        return {"action": "stop", "reason": "Gradient norm is NaN."}

    # --- Loss spike detection ---
    if loss_p95 and loss_median and loss_median > 0:
        spike_ratio = loss_p95 / loss_median
        if spike_ratio > 5 and elapsed_h >= 2:
            return {
                "action": "stop",
                "reason": f"Loss highly unstable (p95/median={spike_ratio:.1f}). "
                f"Consider lower lr or tighter grad_clip.",
            }

    # --- Stagnation after extended time ---
    ret_train = recent.get("metrics/return_train", {})
    if elapsed_h >= 3 and ret_train:
        ret_last = ret_train.get("last", 0)
        ret_p95 = ret_train.get("p95", 0)
        if ret_last > 0 and ret_p95 > 0 and ret_last < ret_p95 * 0.3:
            pass  # Return dropped significantly but might be exploration; don't stop

    # --- Memory buffer health ---
    buffer_len = recent.get("buffer/memory_len", {}).get("last")
    if buffer_len is not None and buffer_len < 100 and elapsed_h > 0.5:
        return {
            "action": "stop",
            "reason": f"Buffer nearly empty ({buffer_len}) after {elapsed_h:.1f}h. "
            f"Worker/server connection issue.",
        }

    # --- All clear ---
    reasons = []
    if loss_last and loss_median:
        reasons.append(f"loss={loss_last:.2f}(med={loss_median:.2f})")
    if q_max_last:
        reasons.append(f"Q_max={q_max_last:.1f}")
    if best_ft and best_ft > 0:
        reasons.append(f"best_finish={best_ft:.1f}s")
    elif best_ft is None or best_ft == 0:
        reasons.append("no finish yet")
    if ret_train.get("last"):
        reasons.append(f"return={ret_train['last']:.1f}")

    summary = ", ".join(reasons) if reasons else "metrics unavailable"
    return {"action": "continue", "reason": f"Training healthy: {summary}"}


def _propose(context: dict[str, Any]) -> dict[str, Any]:
    """Propose the next experiment based on completed results.

    Reads the search space and past experiments to suggest what to try next.
    """
    registry = context.get("registry", [])
    completed = [e for e in registry if e.get("status") in ("completed", "stopped_early")]

    # Check what we've already tried
    tried_params: set[str] = set()
    for e in registry:
        overrides = e.get("config_overrides", {})
        for section in overrides.values():
            if isinstance(section, dict):
                tried_params.update(section.keys())
            else:
                tried_params.add(str(section))

    # Load analyses for completed experiments
    analyses: list[dict[str, Any]] = []
    for e in completed:
        ap = EXPERIMENTS_DIR / "analysis" / f"{e['exp_id']}.json"
        if ap.exists():
            with contextlib.suppress(Exception):
                analyses.append(json.loads(ap.read_text(encoding="utf-8")))

    best_ft = float("inf")
    best_parent = "gtn-baseline"
    best_return = -float("inf")
    for a in analyses:
        ft = a.get("best_finish_time_s")
        if ft and ft > 0 and ft < best_ft:
            best_ft = ft
            best_parent = a.get("exp_id", "gtn-baseline")
        ret = a.get("metrics", {}).get("metrics/return_train", {}).get("last", 0)
        if ret > best_return:
            best_return = ret

    # Proposal logic based on what hasn't been tried
    proposals = []

    if "iqn_lr" not in tried_params:
        proposals.append(
            {
                "exp_id": "higher-lr-5e5",
                "hypothesis": "Increase learning rate to 5e-5 for faster convergence.",
                "overrides": {"algorithm": {"iqn_lr": 5e-5}},
            }
        )

    if "batch_size" not in tried_params:
        proposals.append(
            {
                "exp_id": "batch-512",
                "hypothesis": "Double batch size to 512 for lower gradient variance.",
                "overrides": {"training": {"batch_size": 512}},
            }
        )

    if "iqn_epsilon_decay_steps" not in tried_params:
        proposals.append(
            {
                "exp_id": "fast-exploit-800k",
                "hypothesis": "Reduce epsilon decay to 800k steps for faster exploitation.",
                "overrides": {"algorithm": {"iqn_epsilon_decay_steps": 800000}},
            }
        )

    if "gamma" not in tried_params:
        proposals.append(
            {
                "exp_id": "shorter-horizon-gamma99",
                "hypothesis": (
                    "Lower gamma to 0.99 for more stable Q-values and faster credit assignment."
                ),
                "overrides": {"algorithm": {"gamma": 0.99}},
            }
        )

    if "end_of_track_reward" not in tried_params:
        proposals.append(
            {
                "exp_id": "big-finish-bonus-16",
                "hypothesis": (
                    "Double finish reward to 16.0 to strongly incentivize track completion."
                ),
                "overrides": {"environment": {"end_of_track_reward": 16.0}},
            }
        )

    if "n_steps" not in tried_params:
        proposals.append(
            {
                "exp_id": "nsteps-5-longer-returns",
                "hypothesis": (
                    "Increase n_steps to 5 for better multi-step returns (longer TD targets)."
                ),
                "overrides": {"algorithm": {"n_steps": 5}},
            }
        )

    # If best experiment had high loss, propose lower lr
    for a in analyses:
        loss_p95 = a.get("metrics", {}).get("loss/iqn_loss", {}).get("p95", 0)
        if loss_p95 > 30:
            proposals.append(
                {
                    "exp_id": f"lower-lr-from-{a.get('exp_id', 'unknown')}",
                    "hypothesis": (
                        f"Loss was high (p95={loss_p95:.1f}) in {a.get('exp_id')}. Try lr=2e-5."
                    ),
                    "overrides": {"algorithm": {"iqn_lr": 2e-5}},
                    "parent": a.get("exp_id", "gtn-baseline"),
                }
            )
            break

    if not proposals:
        return {
            "action": "no_proposal",
            "reason": "All standard variations have been tried. Manual review needed.",
        }

    # Pick the first untried proposal
    existing_ids = {e["exp_id"] for e in registry}
    for prop in proposals:
        if prop["exp_id"] not in existing_ids:
            parent = prop.get("parent", best_parent)
            try:
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "tmrl.tools.experiment_manager",
                        "register",
                        "--exp-id",
                        str(prop["exp_id"]),
                        "--parent",
                        str(parent),
                        "--hypothesis",
                        str(prop["hypothesis"]),
                        "--overrides",
                        json.dumps(prop["overrides"]),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(REPO_ROOT),
                )
                if result.returncode == 0:
                    _log(f"Proposed & registered: {prop['exp_id']}")
                    return {"action": "proposed", "exp_id": prop["exp_id"]}
                _log(f"Registration failed: {result.stderr[:200]}")
            except Exception as exc:
                _log(f"Registration error: {exc}")

    return {
        "action": "no_proposal",
        "reason": "Could not register new experiment.",
    }


VALID_OVERRIDE_SECTIONS = {
    "algorithm",
    "training",
    "model",
    "environment",
    "player_runs",
    "run",
    "wandb",
    "distributed",
}

# Map common param names to their correct section
PARAM_TO_SECTION: dict[str, str] = {
    "iqn_lr": "algorithm",
    "gamma": "algorithm",
    "iqn_grad_clip": "algorithm",
    "iqn_epsilon_decay_steps": "algorithm",
    "iqn_epsilon_start": "algorithm",
    "iqn_epsilon_end": "algorithm",
    "n_steps": "algorithm",
    "iqn_soft_target_tau": "algorithm",
    "backup_clip_range": "algorithm",
    "iqn_huber_kappa": "algorithm",
    "iqn_dueling": "algorithm",
    "iqn_double_dqn": "algorithm",
    "iqn_sort_quantiles": "algorithm",
    "reward_normalize_scale": "algorithm",
    "batch_size": "training",
    "training_steps_per_round": "training",
    "rounds_per_epoch": "training",
    "environment_steps_before_training": "training",
    "max_training_steps_per_environment_step": "training",
    "update_model_interval": "training",
    "update_buffer_interval": "training",
    "residual_mlp_hidden_dim": "model",
    "residual_mlp_num_blocks": "model",
    "gnn_layers": "model",
    "gnn_hidden": "model",
    "binary_brake": "model",
    "end_of_track_reward": "environment",
    "crash_penalty": "environment",
    "speed_reward_weight": "environment",
    "constant_penalty": "environment",
    "demo_sampling_weight": "player_runs",
    "demo_max_batch_fraction": "player_runs",
    "buffers_maxlen": "run",
    "rw_max_samples_per_episode": "run",
}


def _fix_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Fix malformed overrides from Gemini (wrong section keys, dot notation)."""
    fixed: dict[str, Any] = {}

    for section, params in overrides.items():
        if not isinstance(params, dict):
            continue

        if section in VALID_OVERRIDE_SECTIONS:
            # Valid section -- but check for dot-notation keys inside
            clean_params: dict[str, Any] = {}
            for key, val in params.items():
                if "." in key:
                    # e.g. "algorithm.iqn_grad_clip" -> extract param name
                    real_key = key.split(".")[-1]
                    real_section = PARAM_TO_SECTION.get(real_key, section)
                    fixed.setdefault(real_section, {})[real_key] = val
                else:
                    clean_params[key] = val
            if clean_params:
                fixed.setdefault(section, {}).update(clean_params)
        else:
            # Invalid section (e.g. "optimization", "rl_algorithm")
            # Try to remap each param to its correct section
            for key, val in params.items():
                real_key = key.split(".")[-1] if "." in key else key
                real_section = PARAM_TO_SECTION.get(real_key)  # type: ignore[assignment]
                if real_section:
                    fixed.setdefault(real_section, {})[real_key] = val
                else:
                    # Unknown param -- put in algorithm as best guess
                    fixed.setdefault("algorithm", {})[real_key] = val

    return fixed


def _call_gemini(prompt: str, *, retries: int = 3) -> str | None:
    """Call Gemini API and return the text response.

    Retries transient failures with exponential backoff.
    """
    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _log("WARNING: GEMINI_API_KEY not set, falling back to heuristic")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        _log(f"Gemini SDK not installed: {exc}")
        return None

    client = genai.Client(api_key=api_key)
    last_err: str = ""
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.9,
                ),
            )
            return response.text
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                delay = 10 * (2 ** (attempt - 1))
                _log(
                    f"Gemini attempt {attempt}/{retries} failed ({exc!r}), retrying in {delay}s..."
                )
                time.sleep(delay)

    _log(f"Gemini API failed after {retries} attempts: {last_err}")
    return None


def _build_decide_prompt(context: dict[str, Any]) -> str:
    snapshot = context.get("snapshot", {})
    target = context.get("target_finish_time_s", 36.0)
    elapsed = context.get("elapsed_hours", 0)
    max_hours = context.get("current_max_hours", 4)
    exp_entry = context.get("exp_entry", {})
    recent = snapshot.get("recent_metrics", {})

    return f"""You are an ML experiment analyst for a TrackMania reinforcement learning agent.

TARGET: Finish the track in {target}s or less.
Primary metric: eval/finish_time_test_s > 0 (lower is better).
Note: eval logs 0.0 when the agent did NOT finish that eval episode;
use min of positive values only.

CURRENT EXPERIMENT: {exp_entry.get("exp_id", "unknown")}
HYPOTHESIS: {exp_entry.get("hypothesis", "N/A")}
ELAPSED: {elapsed:.1f}h / {max_hours}h max
OVERRIDES: {json.dumps(exp_entry.get("config_overrides", {}), indent=2)}

RECENT METRICS (last ~100 trainer steps):
{json.dumps(recent, indent=2, default=str)}

SNAPSHOT SUMMARY (trust these before claiming "never finished"):
- best_finish_time_s: {snapshot.get("best_finish_time_s", "None (no positive finish yet)")}
- last_finish_time_s: {snapshot.get("last_finish_time_s", "N/A")}
- worker_best_finish_time_s: {snapshot.get("worker_best_finish_time_s", "N/A")}
- worker_finish_count: {snapshot.get("worker_finish_count", 0)} (episodes with run/finish_time > 0)
- trainer_state: {snapshot.get("trainer_state", "unknown")}
- worker_state: {snapshot.get("worker_state", "unknown")}

IMPORTANT - IQN loss scale for this project:
- loss/iqn_loss in the 30-90 range is COMMON while Q-values are stable (max_q roughly 15-50).
- Do NOT stop solely because recent loss last/p95 exceeds 50.
- Only treat loss as diverged if last > 100, or Q-values explode (|max_q| > 200),
  or loss trends sharply upward with collapsing returns.

NOTE ON GRADIENTS: Pre-clip gradient norms are structurally 50-150x the clip limit for this
architecture. This is NORMAL and must NOT be used as a stop criterion. Only stop on gradients
if grad_norm itself is NaN.

SIGNS TO STOP EARLY:
- Q-values exploding (max_q > 200 or min_q < -50)
- Loss last > 100 or NaN
- No positive best_finish_time_s AND worker_finish_count == 0 after 2+ hours
- Buffer empty (connection issue)

SIGNS TO CONTINUE:
- worker_finish_count >= 5 — STRONG signal to continue, even if other metrics look bad
- best_finish_time_s > 0 (even if >> target) — learning to finish
- worker_finish_count increasing
- Returns increasing (even slowly)
- Q-values in reasonable range (roughly 0-50 for max_q in this setup)
- Epsilon still decaying (still exploring)

Respond with ONLY a JSON object (no markdown, no explanation):
{{"action": "continue" or "stop", "reason": "brief explanation"}}"""


def _build_propose_prompt(context: dict[str, Any]) -> str:
    registry = context.get("registry", [])
    target = context.get("target_finish_time_s", 36.0)

    # Load search space
    search_space_text = ""
    sp_path = EXPERIMENTS_DIR / "search_space.yaml"
    if sp_path.exists():
        search_space_text = sp_path.read_text(encoding="utf-8")[:3000]

    # Load decisions log
    decisions_text = ""
    dec_path = EXPERIMENTS_DIR / "decisions.md"
    if dec_path.exists():
        decisions_text = dec_path.read_text(encoding="utf-8")[-3000:]

    # Load validation report (produced by scripts/validate_decisions.py)
    validation_text = ""
    val_path = EXPERIMENTS_DIR / "validation_report.json"
    if val_path.exists():
        try:
            val = json.loads(val_path.read_text(encoding="utf-8"))
            parts = [
                f"Errors: {val.get('error_count', 0)}, Warnings: {val.get('warning_count', 0)}"
            ]
            for lb in val.get("leaderboard", [])[:5]:
                parts.append(f"  #{lb['rank']} {lb['exp_id']}: {lb['best_time_s']:.1f}s")
            for f in val.get("findings", []):
                if f.get("severity") == "WARNING" and f.get("category") in (
                    "gradient_obsession",
                    "premature_stops",
                    "leaderboard_mismatch",
                ):
                    parts.append(f"  [{f['category']}] {f['claim']}")
            validation_text = "VALIDATION REPORT:\n" + "\n".join(parts)
        except Exception:
            pass

    # Build rich experiment summary with analysis data
    reg_summary = []
    for e in registry:
        parts = [
            f"- {e['exp_id']}: status={e.get('status')}",
            f"  overrides={json.dumps(e.get('config_overrides', {}))}",
        ]
        sm = e.get("summary_metrics") or {}
        ft = sm.get("best_finish_time_s")
        if ft and ft > 0:
            parts.append(f"  best_finish={ft:.2f}s")

        ap = EXPERIMENTS_DIR / "analysis" / f"{e['exp_id']}.json"
        if ap.exists():
            try:
                a = json.loads(ap.read_text(encoding="utf-8"))
                lo = a.get("metrics", {}).get("loss/iqn_loss", {})
                if lo.get("median"):
                    parts.append(f"  loss_median={lo['median']:.1f}")
                ret = a.get("metrics", {}).get("metrics/return_train", {})
                if ret.get("last"):
                    parts.append(f"  return_last={ret['last']:.0f}")
                w = a.get("worker", {})
                if w.get("finish_rate"):
                    parts.append(f"  worker_finish_rate={w['finish_rate']:.1%}")
                if w.get("finish_count"):
                    parts.append(f"  worker_finishes={w['finish_count']}")
                trends = a.get("training_trends", {})
                if trends:
                    dirs = {
                        k: v.get("direction", "?") if isinstance(v, dict) else v
                        for k, v in trends.items()
                    }
                    parts.append(f"  trends={dirs}")
            except Exception:
                pass

        if e.get("stop_reason"):
            parts.append(f"  stop_reason={e['stop_reason']}")

        reg_summary.append("\n".join(parts))

    # Parameter effects summary
    param_effects_text = ""
    completed = [e for e in registry if e.get("status") in ("completed", "stopped_early")]
    if completed:
        effects: dict[str, list[str]] = {}
        for e in completed:
            ap = EXPERIMENTS_DIR / "analysis" / f"{e['exp_id']}.json"
            ana: dict[str, Any] = {}
            if ap.exists():
                with contextlib.suppress(Exception):
                    ana = json.loads(ap.read_text(encoding="utf-8"))
            ft = ana.get("best_finish_time_s")
            ft_s = f"{ft:.1f}s" if ft and ft > 0 else "DNF"
            lo = ana.get("metrics", {}).get("loss/iqn_loss", {}).get("median")
            lo_s = f"loss={lo:.1f}" if lo else ""
            for dk, val in _flatten_dict_orch(e.get("config_overrides", {})):
                effects.setdefault(dk, []).append(f"{val}({e['exp_id']}:{ft_s} {lo_s})")
        lines = []
        for pk, trials in effects.items():
            lines.append(f"  {pk}: {', '.join(trials)}")
        if lines:
            param_effects_text = "PARAMETER EFFECTS (param: value(exp:result) ...):\n" + "\n".join(
                lines
            )

    return f"""You are an ML experiment designer for a TrackMania RL agent (IQN algorithm).

TARGET: Finish the track in {target} seconds.

PAST EXPERIMENTS (with metrics):
{chr(10).join(reg_summary)}

{param_effects_text}

DECISIONS LOG (recent):
{decisions_text[-2000:]}

{validation_text}

SEARCH SPACE (available parameters to tune):
{search_space_text[:2000]}

CRITICAL FINDINGS FROM PAST EXPERIMENTS:
- Gradient clipping saturation is STRUCTURAL. Do NOT propose experiments that try to "fix"
  gradient clipping (changing iqn_grad_clip, adam_eps, weight_decay for gradient purposes,
  or iqn_soft_target_tau for gradient purposes). These have been tried 11 times and all failed.
- The best config is: batch_size=512, iqn_lr=3e-5, iqn_grad_clip=1.0. Always use
  stable-learning-with-strict-clip (61.65s) as parent. Adding n_steps=7 (long-horizon-planning-v2)
  improved finish rate (25%) but not best time (78.18s).
- UNTRIED directions that should be explored: iqn_epsilon_decay_steps (currently 2M, try 800k),
  end_of_track_reward (currently 8, try 16+), speed_reward_weight, constant_penalty,
  reward_normalize_scale, Munchausen RL, model capacity changes, longer training time.
- Do NOT propose experiments similar to: adam-eps-for-stability, softer-target-network,
  stable-clip-regularized-tau, weight-decay variants, or any "fix gradient clipping" idea.

STRATEGY:
1. Look at which experiments had the best finish times and return values.
2. Check the parameter effects -- which params improved performance?
3. Consider combining parameters that individually helped.
4. Check which search space params haven't been tried yet.
5. Avoid configs similar to experiments that failed or were stopped early for bad metrics.
6. If the best experiment's training trends show "improving", consider extending its approach.
7. ALWAYS use "stable-learning-with-strict-clip" (best: 61.65s) as parent
   unless you have a specific reason not to.

CRITICAL - OVERRIDE FORMAT RULES:
The "overrides" dict must use EXACTLY these top-level section keys:
- "algorithm": iqn_lr, gamma, iqn_grad_clip, iqn_epsilon_decay_steps,
  n_steps, iqn_soft_target_tau, backup_clip_range, etc.
- "training": batch_size, training_steps_per_round,
  rounds_per_epoch, environment_steps_before_training, etc.
- "model" for: residual_mlp_hidden_dim, residual_mlp_num_blocks, gnn_layers, gnn_hidden, etc.
- "environment" for: end_of_track_reward, reward (nested: crash_penalty, speed_reward_weight, etc.)
- "player_runs" for: demo_sampling_weight, demo_max_batch_fraction, etc.
- "run" for: buffers_maxlen, rw_max_samples_per_episode, etc.

WRONG: {{"optimization": {{"algorithm.iqn_grad_clip": 5.0}}}}
CORRECT: {{"algorithm": {{"iqn_grad_clip": 5.0}}}}

WRONG: {{"rl_algorithm": {{"algorithm.gamma": 0.99}}}}
CORRECT: {{"algorithm": {{"gamma": 0.99}}}}

Respond with ONLY a JSON object (no markdown, no code fences):
{{"exp_id": "kebab-case-name", "parent": "gtn-baseline",
"hypothesis": "why this should help",
"overrides": {{"section_name": {{"param_name": value}}}}}}"""


def _flatten_dict_orch(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict_orch(v, path))
        else:
            items.append((path, v))
    return items


def _is_gradient_stop(reason: str) -> bool:
    """True if the stop reason is about gradient norms/clipping (a known false alarm)."""
    lower = reason.lower()
    gradient_keywords = [
        "gradient norm",
        "grad_norm",
        "gradient clip",
        "grad clip",
        "pre-clip",
        "pre_clip",
        "clipping",
        "saturating",
        "truncated",
        "truncation",
    ]
    return any(kw in lower for kw in gradient_keywords)


def _make_decision(mode: str, context: dict[str, Any]) -> dict[str, Any]:
    """Make a decision using Gemini AI, with heuristic fallback.

    Hard overrides (applied AFTER Gemini):
    1. Worker finishing tracks (count >= 5) => always continue.
    2. Gradient-based stop reasons => override to continue (structural, not a bug).
    """
    if mode == "decide":
        heuristic = _decide(context)

        prompt = _build_decide_prompt(context)
        response = _call_gemini(prompt)

        gemini_decision: dict[str, Any] | None = None
        if response:
            try:
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    text = text.rsplit("```", 1)[0]
                try:
                    parsed: dict[str, Any] = json.loads(text.strip())
                except json.JSONDecodeError:
                    parsed = _extract_json_from_output(text) or {}
                if "action" in parsed:
                    _log(f"Gemini decision: {parsed}")
                    gemini_decision = parsed
                else:
                    _log(f"Gemini response missing 'action' key: {text[:200]}")
            except Exception as exc:
                _log(f"Error parsing Gemini decision: {exc}, response: {response[:200]}")

        decision = gemini_decision or heuristic

        if decision.get("action") == "stop":
            reason = decision.get("reason", "")

            # Override 1: worker is finishing tracks => keep running
            if heuristic.get("action") == "continue" and "Worker is finishing" in heuristic.get(
                "reason", ""
            ):
                _log(
                    f"OVERRIDE: Gemini said stop ({reason!r}) but worker is "
                    f"finishing tracks. Forcing continue."
                )
                return heuristic

            # Override 2: gradient-based stop reason => structural, not a problem
            if _is_gradient_stop(reason):
                _log(
                    f"OVERRIDE: Ignoring gradient-based stop ({reason!r}). "
                    f"Pre-clip >> clip is structural for this architecture."
                )
                return {
                    "action": "continue",
                    "reason": (
                        "Gradient stop overridden (structural). "
                        f"Heuristic: {heuristic.get('reason', 'N/A')}"
                    ),
                }

        return decision

    elif mode == "propose":
        prompt = _build_propose_prompt(context)
        response = _call_gemini(prompt)

        if response:
            try:
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    text = text.rsplit("```", 1)[0]

                # Try to extract JSON even if wrapped in extra text
                parsed_json = _extract_json_from_output(text) if "{" in text else None
                proposal: dict[str, Any] = parsed_json or json.loads(text.strip())

                if "exp_id" in proposal and "overrides" in proposal:
                    exp_id = str(proposal["exp_id"])
                    parent = str(proposal.get("parent", "gtn-baseline"))
                    hypothesis = str(proposal.get("hypothesis", "AI-proposed experiment"))
                    raw_overrides = proposal["overrides"]
                    if not isinstance(raw_overrides, dict):
                        _log(f"Gemini overrides not a dict: {type(raw_overrides)}")
                        raise ValueError("overrides must be a dict")
                    overrides = _fix_overrides(raw_overrides)
                    if overrides != raw_overrides:
                        _log(f"Fixed overrides: {raw_overrides} -> {overrides}")

                    try:
                        result = subprocess.run(
                            [
                                "uv",
                                "run",
                                "python",
                                "-m",
                                "tmrl.tools.experiment_manager",
                                "register",
                                "--exp-id",
                                exp_id,
                                "--parent",
                                parent,
                                "--hypothesis",
                                hypothesis,
                                "--overrides",
                                json.dumps(overrides),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            cwd=str(REPO_ROOT),
                        )
                        if result.returncode == 0:
                            _log(f"Gemini proposed & registered: {exp_id}")
                            return {"action": "proposed", "exp_id": exp_id}
                        _log(f"Registration failed: {result.stderr[:200]}")
                    except subprocess.TimeoutExpired:
                        _log("Registration subprocess timed out")
                    except Exception as exc:
                        _log(f"Registration error: {exc}")
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                _log(f"Gemini propose parse error: {exc}, response: {response[:300]}")
            except Exception as exc:
                _log(f"Unexpected error processing Gemini proposal: {exc}")

        _log("Falling back to heuristic proposal...")
        return _propose(context)

    return {"action": "continue", "reason": "Unknown mode"}


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


def run_experiment_loop(start_exp_id: str | None = None) -> None:
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

            # --- Code patch: create experiment branch ---
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

            # --- Smoke check ---
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

            # --- Monitoring loop ---
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

            # --- Teardown ---
            pm.stop()
            pm = None

            # --- Rollback code-patch branch ---
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
    import argparse

    parser = argparse.ArgumentParser(description="TMRL Autonomous Experiment Orchestrator")
    parser.add_argument("--exp-id", default=None, help="Start from a specific experiment ID")
    args = parser.parse_args()
    run_experiment_loop(start_exp_id=args.exp_id)


if __name__ == "__main__":
    main()
