"""Utilities for recording/importing player runs into replay memory."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger
import numpy as np

from tmrl.util import dump, load

if TYPE_CHECKING:
    from tmrl.networking import Buffer

PLAYER_RUN_FORMAT = "tmrl_player_run_v1"


def default_player_runs_dir() -> Path:
    """Return the default folder for player-run files."""
    import tmrl.config.config_constants as cfg

    return cfg.TMRL_FOLDER / "player_runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_payload(obj: Any, source: Path) -> dict[str, Any]:
    if isinstance(obj, dict):
        payload = dict(obj)
        payload.setdefault("format", PLAYER_RUN_FORMAT)
        payload.setdefault("run_id", source.stem)
        payload.setdefault("recorded_at", _utc_now_iso())
        payload.setdefault("metadata", {})
        payload.setdefault("samples", [])
        return payload
    if isinstance(obj, list):
        return {
            "format": PLAYER_RUN_FORMAT,
            "run_id": source.stem,
            "recorded_at": _utc_now_iso(),
            "metadata": {},
            "samples": obj,
        }
    raise ValueError(f"Unsupported run payload in '{source}'. Expected dict or list.")


def validate_samples(samples: list[Any], source: str = "player-run") -> None:
    """Validate transition tuples before replay insertion."""
    for idx, sample in enumerate(samples):
        if not isinstance(sample, (tuple, list)) or len(sample) != 6:
            raise ValueError(
                f"Invalid sample at index {idx} in {source}: expected 6-tuple "
                "(act, obs, rew, terminated, truncated, info)."
            )
        terminated = sample[3]
        truncated = sample[4]
        info = sample[5]
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise ValueError(
                f"Invalid terminal flags at index {idx} in {source}: "
                "terminated/truncated must be bool."
            )
        if not isinstance(info, dict):
            raise ValueError(f"Invalid info at index {idx} in {source}: info must be a dict.")


def save_player_run(
    samples: list[Any],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a single run as a pickle payload with metadata."""
    validate_samples(samples, source="recorded-episode")
    target_dir = Path(output_dir) if output_dir else default_player_runs_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    run_uid = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    payload = {
        "format": PLAYER_RUN_FORMAT,
        "run_id": run_uid,
        "recorded_at": _utc_now_iso(),
        "metadata": metadata or {},
        "samples": samples,
    }
    path = target_dir / f"{run_uid}.pkl"
    dump(payload, path)
    return path


def load_player_run(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate one player-run file."""
    source = Path(path)
    payload = _normalize_payload(load(source), source)
    if payload.get("format") != PLAYER_RUN_FORMAT:
        logger.warning(
            "Unknown player-run format '{}' in '{}'; trying compatibility mode.",
            payload.get("format"),
            source,
        )
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"Invalid samples field in '{source}', expected a list.")
    validate_samples(samples, source=str(source))
    return payload


def _append_samples_to_buffer(samples: list[Any], *, run_id: str | None = None) -> Buffer:
    from tmrl.networking import Buffer

    buffer = Buffer()
    for sample in samples:
        act, obs, rew, terminated, truncated, info = sample
        info_mod = dict(info) if isinstance(info, dict) else {"raw_info": info}
        info_mod.setdefault("is_demo", True)
        info_mod.setdefault("demo_source", "player_runs")
        if run_id is not None:
            info_mod.setdefault("demo_run_id", run_id)
        buffer.append_sample((act, obs, rew, terminated, truncated, info_mod))
    return buffer


def _trim_memory_data(memory: Any, max_samples: int) -> int:
    if not getattr(memory, "data", None):
        return 0
    current_len = len(memory.data[0])
    if current_len <= max_samples:
        return 0
    trim = current_len - max_samples
    for i in range(len(memory.data)):
        memory.data[i] = memory.data[i][trim:]
    return trim


def import_player_runs_to_dataset(
    run_paths: list[str | os.PathLike[str]],
    *,
    memory_factory: Any,
    dataset_path: str | os.PathLike[str],
    overwrite: bool = False,
    max_samples: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import player runs into the configured replay dataset format."""
    if not run_paths:
        raise ValueError("No player-run paths were provided.")

    memory = memory_factory(nb_steps=1, device="cpu")
    if overwrite:
        memory.data = []

    imported_files = 0
    imported_samples = 0
    imported_ids: list[str] = []
    for run_path in run_paths:
        payload = load_player_run(run_path)
        samples = payload["samples"]
        buffer = _append_samples_to_buffer(samples, run_id=str(payload.get("run_id")))
        try:
            memory.append(buffer)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Replay schema mismatch for '{run_path}'. "
                "The run does not match current memory/interface format."
            ) from exc
        imported_files += 1
        imported_samples += len(buffer)
        imported_ids.append(str(payload.get("run_id")))

    trimmed = 0
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be > 0 when provided.")
        trimmed = _trim_memory_data(memory, max_samples=max_samples)

    dataset_file = Path(dataset_path) / "data.pkl"
    if not dry_run:
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        dump(memory.data, dataset_file)

    return {
        "dataset_file": str(dataset_file),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "imported_files": imported_files,
        "imported_samples": imported_samples,
        "imported_run_ids": imported_ids,
        "trimmed_raw_samples": trimmed,
    }


def poll_player_runs_for_injection(
    source_dir: str | os.PathLike[str],
    seen_run_ids: set[str],
    *,
    max_files: int = 1,
    consume_on_read: bool = False,
) -> tuple[Buffer, set[str], list[str]]:
    """Poll pending player runs and return a merged buffer for trainer injection."""
    from tmrl.networking import Buffer

    root = Path(source_dir).resolve()
    if not root.exists():
        _warned = getattr(poll_player_runs_for_injection, "_warned_missing_paths", set())
        if str(root) not in _warned:
            logger.warning(
                "Player runs SOURCE_PATH does not exist (trainer must see this path): {}",
                root,
            )
            _warned.add(str(root))
            poll_player_runs_for_injection._warned_missing_paths = _warned
        return Buffer(), set(), []

    files = sorted(p for p in root.glob("*.pkl") if p.is_file())
    if not files:
        _has_imported = any(root.glob("*.pkl.imported"))
        if not _has_imported and not getattr(poll_player_runs_for_injection, "_logged_empty_dir", False):
            logger.info(
                "Player runs: no .pkl files in {} (add recordings or run --record-episode)",
                root,
            )
            poll_player_runs_for_injection._logged_empty_dir = True
    if max_files > 0:
        files = files[:max_files]

    merged = Buffer()
    imported_ids: set[str] = set()
    imported_files: list[str] = []

    for path in files:
        payload = load_player_run(path)
        run_id = str(payload.get("run_id") or path.stem)
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        imported_ids.add(run_id)
        imported_files.append(str(path))
        merged += _append_samples_to_buffer(payload["samples"], run_id=run_id)

        if consume_on_read:
            imported_path = path.with_suffix(path.suffix + ".imported")
            os.replace(path, imported_path)

    if imported_files:
        poll_player_runs_for_injection._logged_empty_dir = False
    return merged, imported_ids, imported_files
