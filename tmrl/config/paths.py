"""Filesystem paths derived from the validated config and TmrlData layout."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from tmrl.config.loader import MAIN_CONFIG, TMRL_FOLDER

_SCRIPT_DIR = Path(__file__).resolve().parent
_TMRL_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _TMRL_ROOT.parent
_output_files_env: str | None = os.environ.get("TMRL_OUTPUT_FILES")
if _output_files_env:
    _OUTPUT_FILES_ROOT = Path(_output_files_env)
else:
    _OUTPUT_FILES_ROOT = _PROJECT_ROOT / "output_files"

CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
DATASET_FOLDER = TMRL_FOLDER / "dataset"
REWARD_FOLDER = TMRL_FOLDER / "reward"
TRACK_FOLDER = TMRL_FOLDER / "track"
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
CONFIG_FOLDER = TMRL_FOLDER / "config"
PATH_DATA = TMRL_FOLDER

OUTPUT_FILES_FOLDER = Path(_OUTPUT_FILES_ROOT)
TRACKS_FOLDER = OUTPUT_FILES_FOLDER / "tracks"
DEBUG_FOLDER = OUTPUT_FILES_FOLDER / "debug"
TRACKMAP_CSV_LEFT = str(TRACKS_FOLDER / "tmrl-test" / "track_left.csv")
TRACKMAP_CSV_RIGHT = str(TRACKS_FOLDER / "tmrl-test" / "track_right.csv")

RUN_NAME = MAIN_CONFIG.run.name
MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))

_MAP_NAME_RAW = (MAIN_CONFIG.environment.map_name or "").strip()
if not _MAP_NAME_RAW:
    MAP_NAME = "tmrl-test"
    warnings.warn(
        "environment.map_name is empty; using 'tmrl-test' for reward_/track_ pickles under "
        "TmrlData. Set environment.map_name in config (e.g. local.yaml) to your map id "
        "(e.g. test-3 for reward_test-3.pkl).",
        UserWarning,
        stacklevel=2,
    )
else:
    MAP_NAME = _MAP_NAME_RAW
REWARD_PATH = str(REWARD_FOLDER / ("reward_" + MAP_NAME + ".pkl"))
TRACK_PATH_LEFT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_left.pkl"))
TRACK_PATH_RIGHT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_right.pkl"))
REWARDS_CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_rew_" + MAP_NAME + "_t.tcpt"))

_raw_dataset = MAIN_CONFIG.run.dataset_path
if _raw_dataset is not None and str(_raw_dataset).strip() == "":
    DATASET_PATH = str(DATASET_FOLDER / "_no_load")
else:
    DATASET_PATH = str(_raw_dataset) if _raw_dataset is not None else str(DATASET_FOLDER)

PLAYER_RUNS_FOLDER = TMRL_FOLDER / "player_runs"
