"""Recording utilities: import player runs and save TrackMania replays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.tools.recording.player_runs import import_player_runs_to_dataset
from tmrl.util import partial

# Steps budget passed to run_episodes; large enough that nb_episodes is the effective limit.
_MAX_STEPS = 10_000


# ---------------------------------------------------------------------------
# Import player runs
# ---------------------------------------------------------------------------


@dataclass
class ImportPlayerRunsCli:
    """CLI for importing one or more player-run files."""

    paths: str
    """Comma-separated list of .pkl player-run files."""

    overwrite: bool = False
    """Overwrite existing dataset instead of appending."""

    max_samples: int = 0
    """Max raw samples to keep in resulting dataset (0 keeps all)."""

    dry_run: bool = False
    """Validate/convert only; don't write dataset file."""


def _parse_paths(paths: str) -> list[str]:
    """Split a comma-separated path string into a list of non-empty stripped paths."""
    return [p.strip() for p in paths.split(",") if p.strip()]


def import_player_runs(
    *,
    paths: list[str],
    overwrite: bool = False,
    max_samples: int | None = None,
    dry_run: bool = False,
    dataset_path: str | None = None,
) -> dict:
    """Import player-run files to the configured replay dataset."""
    if not paths:
        raise ValueError("At least one player-run path is required.")
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Run file not found: {p}")

    target_dataset_path = dataset_path or cfg.DATASET_PATH
    result = import_player_runs_to_dataset(
        paths,
        memory_factory=cfg_obj.MEMORY,
        dataset_path=target_dataset_path,
        overwrite=overwrite,
        max_samples=max_samples,
        dry_run=dry_run,
    )
    logger.info(
        "Imported {} file(s), {} samples into '{}'. dry_run={} trimmed={}",
        result["imported_files"],
        result["imported_samples"],
        result["dataset_file"],
        result["dry_run"],
        result["trimmed_raw_samples"],
    )
    return result


# ---------------------------------------------------------------------------
# Save replays
# ---------------------------------------------------------------------------


@dataclass
class SaveReplaysCli:
    """CLI for saving a fixed number of replays."""

    nb_replays: int = 0
    """Number of replays to record (0 = unlimited)."""


def save_replays(nb_replays: int | None = None) -> None:
    """Run a standalone worker that saves TrackMania replays."""
    env_config = cfg_obj.CONFIG_DICT.copy()
    env_config["interface_kwargs"] = {"save_replays": True}
    rollout_worker = RolloutWorker(
        env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": env_config}),
        actor_module_cls=partial(cfg_obj.POLICY),
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        device="cuda" if cfg.CUDA_INFERENCE else "cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        crc_debug=cfg.CRC_DEBUG,
        standalone=True,
    )
    limit: int | float = nb_replays if nb_replays is not None else np.inf
    rollout_worker.run_episodes(_MAX_STEPS, nb_episodes=limit)
