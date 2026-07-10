"""Experiment registry utilities and path constants for experiment_manager."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import yaml

from tmrl.tools._experiment_io import EXPERIMENTS_DIR
from tmrl.tools._experiment_io import read_registry as _read_registry

CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"
EXPERIMENTS_LOGS_DIR = EXPERIMENTS_DIR / "logs"
AGENT_CONTEXT_PATH = EXPERIMENTS_DIR / "_agent_context.json"
BASELINE_PATH = EXPERIMENTS_DIR / "baseline.yaml"
DECISIONS_PATH = EXPERIMENTS_DIR / "decisions.md"
SEARCH_SPACE_PATH = EXPERIMENTS_DIR / "search_space.yaml"
ORCHESTRATOR_CONFIG_PATH = EXPERIMENTS_DIR / "orchestrator_config.yaml"


def _warn(msg: str) -> None:
    """Print a warning to stderr with a consistent ``[experiment_manager WARNING]`` prefix."""
    print(f"[experiment_manager WARNING] {msg}", file=sys.stderr, flush=True)


def _retry(fn, *, retries: int = 3, base_delay: float = 5.0, label: str = ""):
    """Call *fn()* with exponential-backoff retries on any exception.

    Returns the result on success, or re-raises on final failure.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                _warn(
                    f"{label or 'retry'}: attempt {attempt}/{retries} failed "
                    f"({exc!r}), retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _safe_float_series(series):
    """Convert a pandas series to float, coercing non-numeric values to NaN."""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").astype(float)


def _load_target_time() -> float:
    """Read target_finish_time_s from orchestrator_config.yaml."""
    if ORCHESTRATOR_CONFIG_PATH.exists():
        with ORCHESTRATOR_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("target_finish_time_s", 36.0))
    return 36.0


def _orch_defaults() -> tuple[str, str]:
    """Read default entity/project from orchestrator_config.yaml."""
    if ORCHESTRATOR_CONFIG_PATH.exists():
        with ORCHESTRATOR_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("wandb_entity", "tmrl"), cfg.get("wandb_project", "tmrl")
    return "tmrl", "tmrl"


def _next_exp_id() -> str:
    """Return the next auto-incremented experiment ID (``EXP001``, ``EXP002``, …).

    Custom-named experiments (e.g. ``gtn-baseline``) are ignored; only IDs of
    the form ``EXP###`` contribute to the counter.
    """
    entries = _read_registry()
    if not entries:
        return "EXP001"
    max_num = 0
    for e in entries:
        eid = e.get("exp_id", "")
        # Support both EXP### and custom names -- only auto-increment numeric
        if eid.startswith("EXP") and eid[3:].isdigit():
            max_num = max(max_num, int(eid[3:]))
    return f"EXP{max_num + 1:03d}"


def _load_baseline() -> dict[str, Any]:
    """Load ``experiments/baseline.yaml`` and return its contents as a dict."""
    with BASELINE_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*, returning a new dict.

    Nested dicts are merged rather than replaced; all other value types use
    the overlay value.
    """
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_diff(base: dict, other: dict, prefix: str = "") -> dict[str, tuple[Any, Any]]:
    """Return {dotted.key: (base_val, other_val)} for all leaves that differ."""
    diffs: dict[str, tuple[Any, Any]] = {}
    all_keys = set(base) | set(other)
    for k in sorted(all_keys):
        path = f"{prefix}.{k}" if prefix else k
        bv = base.get(k)
        ov = other.get(k)
        if isinstance(bv, dict) and isinstance(ov, dict):
            diffs.update(_deep_diff(bv, ov, path))
        elif bv != ov:
            diffs[path] = (bv, ov)
    return diffs


def _build_full_config(exp_id: str) -> dict[str, Any]:
    """Reconstruct the full config for an experiment by walking the parent chain."""
    entries = {e["exp_id"]: e for e in _read_registry()}
    overrides_chain: list[dict] = []
    current = exp_id
    while current and current != "baseline":
        entry = entries.get(current)
        if not entry:
            break
        overrides_chain.append(entry.get("config_overrides", {}))
        current = entry.get("parent_exp_id", "baseline")
    base = _load_baseline()
    for ovr in reversed(overrides_chain):
        base = _deep_merge(base, ovr)
    return base


def _build_config_overrides_json(overrides: dict[str, Any]) -> str:
    """Serialise *overrides* to the JSON string expected by ``TMRL_CONFIG_OVERRIDES``.

    Non-serialisable values are coerced via ``str()``.
    """
    return json.dumps(overrides, default=str)
