"""Write reproducibility files beside the trainer checkpoint (paper / audit trail).

On a *fresh* training run (no existing checkpoint), after the first save we write:

- ``repro_merged_config.yaml`` — Hydra + ``local.yaml`` merge, secrets redacted
- ``repro_validated_config.yaml`` — validated :class:`~tmrl.config.schema.main.MainConfig` as YAML
- ``repro_provenance.json`` — git revision (if any), Python/platform, argv, paths

Set ``TMRL_SKIP_REPRO_ARTIFACTS=1`` to disable. If Weights & Biases is active, files are also
registered with ``wandb.save`` so they appear in the run folder.

.. versionadded:: 0.6.x
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from tmrl.config.loader import MAIN_CONFIG, merged_config_snapshot_redacted


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _git_provenance(repo_root: Path | None = None) -> dict[str, Any] | None:
    """Return git metadata or None if not a git checkout / git missing."""
    try:
        cwd = str(repo_root) if repo_root is not None else os.getcwd()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        if commit.returncode != 0:
            return None
        sha = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        return {
            "commit": sha,
            "branch": branch.stdout.strip() if branch.returncode == 0 else "",
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _validated_config_redacted() -> dict[str, Any]:
    data = MAIN_CONFIG.model_dump(mode="json")
    wandb = data.get("wandb")
    if isinstance(wandb, dict) and wandb.get("api_key"):
        wandb = dict(wandb)
        wandb["api_key"] = "<redacted>"
        data = dict(data)
        data["wandb"] = wandb
    dist = data.get("distributed")
    if isinstance(dist, dict) and dist.get("password"):
        dist = dict(dist)
        dist["password"] = "<redacted>"
        data = dict(data)
        data["distributed"] = dist
    return data


def _try_find_git_root() -> Path | None:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / ".git").exists():
            return p
    return None


def write_run_repro_bundle(checkpoint_path: str) -> list[Path]:
    """Write reproducibility artifacts next to ``checkpoint_path`` (same directory).

    Returns paths written (empty if skipped).
    """
    if _truthy_env("TMRL_SKIP_REPRO_ARTIFACTS"):
        return []

    ckpt = Path(checkpoint_path)
    if ckpt.name.endswith("_remove_on_exit"):
        return []

    out_dir = ckpt.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("Could not create repro artifact dir {}: {}", out_dir, e)
        return []

    merged_path = out_dir / "repro_merged_config.yaml"
    validated_path = out_dir / "repro_validated_config.yaml"
    provenance_path = out_dir / "repro_provenance.json"

    if merged_path.is_file() and not _truthy_env("TMRL_REPRO_OVERWRITE"):
        logger.debug(
            "Repro artifacts already exist under {}; skipping "
            "(set TMRL_REPRO_OVERWRITE=1 to replace).",
            out_dir,
        )
        return []

    try:
        merged = merged_config_snapshot_redacted()
        validated = _validated_config_redacted()
        git_root = _try_find_git_root()
        provenance: dict[str, Any] = {
            "written_at_utc": datetime.now(UTC).isoformat(),
            "schema_version": MAIN_CONFIG.schema_version,
            "checkpoint_path": str(ckpt.resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "git": _git_provenance(git_root),
        }

        with open(merged_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                merged,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        with open(validated_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                validated,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

        written = [merged_path, validated_path, provenance_path]
        logger.info(
            "Wrote reproducibility bundle next to checkpoint: {}",
            ", ".join(p.name for p in written),
        )

        try:
            import wandb

            if wandb.run is not None:
                for p in written:
                    wandb.save(str(p.resolve()))
        except Exception as e:
            logger.debug("wandb.save for repro artifacts skipped: {}", e)

        return written
    except Exception as e:
        logger.warning("Failed to write repro artifacts to {}: {}", out_dir, e)
        return []
