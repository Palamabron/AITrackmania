"""ProcessManager: manages the three TMRL subprocesses (server, trainer, worker)."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from typing import Any

from tmrl.tools._experiment_io import REPO_ROOT
from tmrl.tools._orchestrator_utils import (
    LOGS_DIR,
    _free_distributed_ports,
    _kill_port,
    _kill_process_tree,
    _log,
)


class ProcessManager:
    """Manages the three TMRL subprocesses: server, trainer, worker."""

    def __init__(
        self, exp_id: str, config_overrides: dict[str, Any], uv_env: str, server_port: int = 55555
    ) -> None:
        """Initialise the manager without launching any processes.

        Args:
            exp_id: Unique experiment identifier, used for log file naming and
                injected into the subprocess environment as
                ``TMRL_EXPERIMENT_ID``.
            config_overrides: Hydra config overrides merged into
                ``TMRL_CONFIG_OVERRIDES`` for all three subprocesses.
            uv_env: UV venv directory name (e.g. ``'.venv'`` or
                ``'.venv-linux'``), forwarded as ``UV_PROJECT_ENVIRONMENT``.
            server_port: TCP port the relay server listens on; worker and
                trainer use ``server_port + 1..3`` as their local ports.
        """
        self.exp_id = exp_id
        self.config_overrides = config_overrides
        self.uv_env = uv_env
        self.server_port = server_port
        self.processes: dict[str, subprocess.Popen] = {}
        self._log_dir = LOGS_DIR / exp_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_files: dict[str, Any] = {}

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment dict with all TMRL/W&B knobs applied.

        Returns:
            A copy of the current process environment with the following
            modifications applied: UV venv path, experiment ID, config
            overrides JSON, W&B service timeout and start-method settings,
            and UTF-8 stdout encoding.
        """
        import os

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
        """Launch server, trainer, and worker in dependency order.

        Starts the relay server first and waits until its TCP port is
        accepting connections, then starts the trainer (waits 15 s to detect
        immediate crashes), then the worker (waits 5 s).

        Raises:
            RuntimeError: If any of the three processes dies immediately after
                launch or the server does not begin accepting connections
                within 90 seconds.
        """
        _free_distributed_ports(self.server_port)

        # Clear old logs for this experiment so we don't mix runs
        for old_log in self._log_dir.glob("*.log"):
            old_log.write_text("", encoding="utf-8")

        env = self._build_env()
        _log(f"Using UV_PROJECT_ENVIRONMENT={self.uv_env}")

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

        self._start_role("trainer", ["uv", "run", "python", "-m", "tmrl", "--trainer"], env)
        time.sleep(15)
        if self.processes["trainer"].poll() is not None:
            raise RuntimeError(
                f"Trainer died immediately (rc={self.processes['trainer'].returncode}). "
                f"Check {self._log_dir / 'trainer_stderr.log'}"
            )
        _log("Trainer is alive.")

        self._start_role("worker", ["uv", "run", "python", "-m", "tmrl", "--worker"], env)
        time.sleep(5)
        if self.processes["worker"].poll() is not None:
            raise RuntimeError(
                f"Worker died immediately (rc={self.processes['worker'].returncode}). "
                f"Check {self._log_dir / 'worker_stderr.log'}"
            )
        _log("Worker is alive. All processes started.")

    def _start_role(self, role: str, args: list[str], env: dict[str, str]) -> None:
        """Spawn a single subprocess and register its log file handles.

        Opens (or appends to) ``{role}_stdout.log`` and ``{role}_stderr.log``
        under the experiment log directory, then starts the process.

        Args:
            role: One of ``'server'``, ``'trainer'``, or ``'worker'``.
            args: Command-line argument list passed directly to
                :class:`subprocess.Popen`.
            env: Full environment dict for the new process.
        """
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
        """Return ``True`` if every tracked subprocess is still running."""
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
        """Read the full text of a role's stdout or stderr log file.

        Args:
            role: One of ``'server'``, ``'trainer'``, or ``'worker'``.
            stream: ``'stdout'`` or ``'stderr'``.

        Returns:
            File contents as a string, or an empty string if the file does
            not yet exist.
        """
        path = self._log_dir / f"{role}_{stream}.log"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def worker_is_sending_samples(self) -> bool:
        """Return ``True`` if the worker log contains a sample-send confirmation."""
        return "Sent" in self._read_role_log(
            "worker"
        ) and "sample(s) to server" in self._read_role_log("worker")

    def trainer_is_receiving_samples(self) -> bool:
        """Return ``True`` if the trainer log shows it has received samples from the server."""
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
        """Return ``True`` if the worker subprocess exists and has not exited."""
        proc = self.processes.get("worker")
        return proc is not None and proc.poll() is None

    def status_summary(self) -> dict[str, str]:
        """Return a role → status string mapping for all tracked processes.

        Returns:
            Dict mapping each role name to ``'running'`` or
            ``'exited(<returncode>)'``.
        """
        return {
            role: "running" if p.poll() is None else f"exited({p.returncode})"
            for role, p in self.processes.items()
        }

    def stop(self) -> None:
        """Terminate all subprocesses gracefully and close their log file handles.

        Processes are stopped in reverse dependency order (worker → trainer →
        server) so that the worker and trainer do not generate spurious
        reconnect errors after the server disappears.  Each process is given
        up to 10 seconds to exit before being force-killed.
        """
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
