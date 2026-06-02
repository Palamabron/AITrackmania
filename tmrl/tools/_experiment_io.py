"""Shared I/O helpers for the experiment tooling (orchestrator + experiment_manager).

Provides dotenv loading, JSONL registry read/write, and common path constants
so that ``orchestrator.py`` and ``experiment_manager.py`` stay DRY.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"


def _warn(msg: str) -> None:
    print(f"[experiment_io WARNING] {msg}", file=sys.stderr, flush=True)


def load_dotenv() -> None:
    """Parse ``REPO_ROOT/.env`` into ``os.environ`` (skip existing keys)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _warn(f"Could not read .env: {exc}")
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


def read_registry() -> list[dict[str, Any]]:
    """Read all experiment entries from the JSONL registry.

    Skips corrupt lines instead of crashing, logging a warning for each.
    """
    if not REGISTRY_PATH.exists():
        return []
    try:
        raw = REGISTRY_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _warn(f"Could not read registry: {exc}")
        return []
    if not raw.strip():
        return []

    entries: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _warn(f"Corrupt registry line {lineno}, skipping: {exc}")
    return entries


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via temp-file + rename.

    Falls back to direct write if the rename fails (e.g. cross-device).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            with suppress(OSError):
                os.close(fd)
            raise
        tmp_path = Path(tmp)
        tmp_path.replace(path)
    except OSError:
        path.write_text(content, encoding="utf-8")


def write_registry(entries: list[dict[str, Any]]) -> None:
    """Overwrite the registry with *entries* (atomic write)."""
    lines = [json.dumps(e, default=str) + "\n" for e in entries]
    _atomic_write(REGISTRY_PATH, "".join(lines))


def append_registry(entry: dict[str, Any]) -> None:
    """Append a single entry to the registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with REGISTRY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        _warn(f"Could not append to registry: {exc}")
        raise


def update_registry_entry(exp_id: str, updates: dict[str, Any]) -> None:
    """Update fields of the registry entry matching *exp_id*."""
    entries = read_registry()
    updated = False
    for e in entries:
        if e.get("exp_id") == exp_id:
            e.update(updates)
            updated = True
            break
    if not updated:
        _warn(f"update_registry_entry: {exp_id!r} not found in registry")
        return
    write_registry(entries)
