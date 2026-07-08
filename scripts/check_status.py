#!/usr/bin/env python3
"""
check_status.py — snapshot of running TMRL trainer/worker experiments.

Usage:
    python check_status.py [--config] [--wandb] [--all]

Flags:
    --config   Print fully-merged active config (runs `python -m tmrl --print-config`)
    --wandb    Query WandB API for recent runs in the tmrl project
    --all      Enable all optional sections
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────────────


def load_dotenv(path: Path = Path(__file__).parent / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


ENV = load_dotenv()
for k, v in ENV.items():
    if v:
        os.environ.setdefault(k, v)

# ── Helpers ───────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def h(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET}  {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


def age(path: Path) -> str:
    mtime = path.stat().st_mtime
    delta = datetime.now(UTC).timestamp() - mtime
    if delta < 60:
        return f"{delta:.0f}s ago"
    if delta < 3600:
        return f"{delta / 60:.1f}m ago"
    if delta < 86400:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


# ── Process detection ────────────────────────────────────────────────────────

ROLES = {
    "server": ["--server"],
    "trainer": ["--trainer"],
    "worker": ["--worker"],
}


def find_tmrl_procs() -> dict[str, list[dict]]:
    """Return {role: [{"pid": ..., "cmd": ...}, ...]}"""
    found: dict[str, list[dict]] = {r: [] for r in ROLES}
    try:
        out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return found
    for line in out.splitlines():
        if "python" not in line and "tmrl" not in line:
            continue
        if "check_status" in line:
            continue
        for role, flags in ROLES.items():
            if any(f in line for f in flags):
                parts = line.split(None, 10)
                pid = parts[1] if len(parts) > 1 else "?"
                cmd = parts[-1] if len(parts) > 1 else line
                found[role].append({"pid": pid, "cmd": cmd[:80]})
    return found


# ── Port check ───────────────────────────────────────────────────────────────

SERVER_PORT = 55555


def port_open(host: str = "127.0.0.1", port: int = SERVER_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# ── TmrlData artifacts ───────────────────────────────────────────────────────

TMRL_DATA = Path.home() / "TmrlData"


def show_artifacts() -> None:
    h("TmrlData artifacts")

    def section(label: str, directory: Path, pattern: str = "*") -> None:
        files = sorted(directory.glob(pattern)) if directory.exists() else []
        if not files:
            warn(f"{label}: (none)")
            return
        ok(f"{label} ({len(files)} file{'s' if len(files) != 1 else ''})")
        for f in files[-5:]:  # show up to 5 most-recent
            try:
                a = age(f)
            except Exception:
                a = "?"
            info(f"{f.name:<45} {DIM}{a}{RESET}")
        if len(files) > 5:
            info(f"… and {len(files) - 5} more")

    section("Checkpoints", TMRL_DATA / "checkpoints", "*.tcpt")
    section("Weights    ", TMRL_DATA / "weights", "*.tmod")
    section("Dataset    ", TMRL_DATA / "dataset", "*")
    section("Reward pkl ", TMRL_DATA / "reward", "*.pkl")
    section("Track pkl  ", TMRL_DATA / "track", "*.pkl")

    # Repro artifacts (written alongside first checkpoint)
    repro_dir = TMRL_DATA / "checkpoints"
    for fname in ["repro_merged_config.yaml", "repro_provenance.json"]:
        fpath = repro_dir / fname
        if fpath.exists():
            ok(f"Repro artifact: {fname}  {DIM}({age(fpath)}){RESET}")


# ── .env / secrets summary ───────────────────────────────────────────────────


def show_env_summary() -> None:
    h(".env / environment")
    wandb_key = ENV.get("WANDB_API_KEY", "")
    if wandb_key:
        masked = wandb_key[:12] + "…" + wandb_key[-4:]
        ok(f"WANDB_API_KEY  {DIM}{masked}{RESET}")
    else:
        warn("WANDB_API_KEY  not set")

    tmrl_pw = ENV.get("TMRL_PASSWORD", "")
    if tmrl_pw:
        ok(f"TMRL_PASSWORD  {DIM}{'*' * min(len(tmrl_pw), 8)}{RESET}")
    else:
        warn("TMRL_PASSWORD  (empty — open/no-auth server)")

    for extra in ("TMRL_HYDRA_OVERRIDES", "TMRL_CONFIG_OVERRIDES", "TMRL_OUTPUT_FILES"):
        val = os.environ.get(extra, "")
        if val:
            ok(f"{extra}  {DIM}{val[:60]}{RESET}")


# ── Process / port summary ───────────────────────────────────────────────────


def show_processes() -> None:
    h("Running TMRL processes")
    procs = find_tmrl_procs()
    any_running = False
    for role, entries in procs.items():
        if entries:
            any_running = True
            for e in entries:
                ok(f"{role:<8}  PID {e['pid']}   {DIM}{e['cmd']}{RESET}")
        else:
            err(f"{role:<8}  not running")

    print()
    if port_open():
        ok(f"Port {SERVER_PORT}  open (server accepting connections)")
    else:
        warn(f"Port {SERVER_PORT}  closed (server not reachable)")

    return any_running


# ── Active config ─────────────────────────────────────────────────────────────


def show_config() -> None:
    h("Active config  (python -m tmrl --print-config)")
    venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    try:
        result = subprocess.run(
            [python, "-m", "tmrl", "--print-config"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent,
        )
        output = result.stdout or result.stderr
        if result.returncode != 0:
            warn("Command returned non-zero exit code")
        for line in output.splitlines()[:80]:  # cap at 80 lines
            print(f"  {line}")
        if output.count("\n") > 80:
            print(f"  {DIM}… (truncated){RESET}")
    except subprocess.TimeoutExpired:
        err("Timed out waiting for --print-config")
    except FileNotFoundError:
        err(f"Python not found at {python}")


# ── WandB runs ────────────────────────────────────────────────────────────────


def show_wandb_runs(project: str = "tmrl", entity: str = "tmrl", limit: int = 5) -> None:
    h(f"WandB runs  ({entity}/{project}, last {limit})")
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        err("WANDB_API_KEY not set — skipping")
        return
    try:
        import wandb  # type: ignore
    except ImportError:
        err("wandb not installed — run: pip install wandb")
        return

    try:
        api = wandb.Api(api_key=api_key)
        runs = api.runs(f"{entity}/{project}", per_page=limit)
        for run in runs:
            state_color = (
                GREEN if run.state == "running" else (YELLOW if run.state == "finished" else DIM)
            )
            print(f"  {state_color}{run.state:<10}{RESET} {run.name:<35} {DIM}{run.id}{RESET}")
            # Show key metrics if available
            summary = run.summary
            metrics = []
            for key in ("loss", "reward", "episode_reward", "avg_reward", "step"):
                if key in summary:
                    metrics.append(f"{key}={summary[key]:.4g}")
            if metrics:
                info("  " + "  ".join(metrics))
    except Exception as exc:
        err(f"WandB API error: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", action="store_true", help="Print merged active config")
    parser.add_argument("--wandb", action="store_true", help="Query WandB API for recent runs")
    parser.add_argument("--all", action="store_true", help="Enable all optional sections")
    args = parser.parse_args()

    print(
        f"\n{BOLD}TMRL experiment status  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}"
    )

    show_env_summary()
    show_processes()
    show_artifacts()

    if args.config or args.all:
        show_config()

    if args.wandb or args.all:
        show_wandb_runs()

    print()


if __name__ == "__main__":
    main()
