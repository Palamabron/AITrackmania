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

from loguru import logger

from tmrl.config.loader import (
    MAIN_CONFIG,
    format_merged_config_yaml_readable,
    merged_config_snapshot_redacted,
)


def _truthy_env(name: str) -> bool:
    """Return ``True`` when the named environment variable holds a truthy string value.

    Treats ``"1"``, ``"true"``, ``"yes"``, and ``"on"`` (case-insensitive) as truthy;
    everything else (including an unset variable) is falsy.

    Args:
        name: Name of the environment variable to read.

    Returns:
        ``True`` if the variable is set to a recognized truthy string, ``False`` otherwise.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _git_provenance(repo_root: Path | None = None) -> dict[str, Any] | None:
    """Collect git metadata for the provenance record.

    Runs ``git rev-parse HEAD``, ``git rev-parse --abbrev-ref HEAD``, and
    ``git status --porcelain`` in *repo_root* (or the current working directory).
    All subprocess calls have a 5-second timeout and are non-fatal.

    Args:
        repo_root: Optional path to the git repository root.  Defaults to the
            current working directory when ``None``.

    Returns:
        A dict with keys ``"commit"`` (SHA), ``"branch"``, and ``"dirty"``
        (``True`` if there are uncommitted changes, ``None`` if the check failed),
        or ``None`` if git is unavailable or the directory is not a git checkout.
    """
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
    """Return the validated ``MAIN_CONFIG`` as a JSON-friendly dict with secrets redacted.

    Returns:
        A copy of ``MAIN_CONFIG.model_dump()`` with ``wandb.api_key`` and
        ``distributed.password`` replaced by ``"<redacted>"``.
    """
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
    """Walk parent directories from this file to find the nearest ``.git`` directory.

    Returns:
        Absolute path of the git repository root, or ``None`` if this file is not
        inside a git checkout.
    """
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / ".git").exists():
            return p
    return None


def write_run_repro_bundle(checkpoint_path: str) -> list[Path]:
    """Write reproducibility artifacts beside the checkpoint on the first trainer save.

    Writes three files into the same directory as *checkpoint_path*:

    - ``repro_merged_config.yaml`` — post-merge config (Hydra + local.yaml), secrets redacted.
    - ``repro_validated_config.yaml`` — Pydantic-validated config as YAML, secrets redacted.
    - ``repro_provenance.json`` — git revision, Python version, platform, argv, paths.

    Skipped (returns empty list) when:

    - ``TMRL_SKIP_REPRO_ARTIFACTS=1`` is set.
    - The checkpoint name ends with ``_remove_on_exit`` (ephemeral checkpoints).
    - The artifact directory cannot be created.
    - The bundle files already exist and ``TMRL_REPRO_OVERWRITE=1`` is not set.

    If a W&B run is active, each written file is also registered via ``wandb.save``.

    Args:
        checkpoint_path: Filesystem path to the checkpoint file.  Artifacts land
            in the same parent directory.

    Returns:
        List of :class:`~pathlib.Path` objects for each file written, or an empty
        list if the bundle was skipped or any error occurred.
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
            f.write(format_merged_config_yaml_readable(merged))
        with open(validated_path, "w", encoding="utf-8") as f:
            f.write(format_merged_config_yaml_readable(validated))
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
