"""Path constants for TMRL.

This module defines all file system paths used by the TMRL framework.
"""

from tmrl.config.loader import ENV_CONFIG, TMRL_CONFIG, TMRL_FOLDER

# Main folder paths
CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
DATASET_FOLDER = TMRL_FOLDER / "dataset"
REWARD_FOLDER = TMRL_FOLDER / "reward"
TRACK_FOLDER = TMRL_FOLDER / "track"
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
CONFIG_FOLDER = TMRL_FOLDER / "config"
PATH_DATA = TMRL_FOLDER

# Model paths
RUN_NAME = TMRL_CONFIG["RUN_NAME"]
MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))

# Map-related paths
MAP_NAME = ENV_CONFIG["MAP_NAME"]
REWARD_PATH = str(REWARD_FOLDER / ("reward_" + MAP_NAME + ".pkl"))
TRACK_PATH_LEFT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_left.pkl"))
TRACK_PATH_RIGHT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_right.pkl"))
REWARDS_CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_rew_" + MAP_NAME + "_t.tcpt"))

# Dataset path with override support
_raw_dataset = TMRL_CONFIG.get("DATASET_PATH")
if _raw_dataset is not None and str(_raw_dataset).strip() == "":
    DATASET_PATH = str(DATASET_FOLDER / "_no_load")  # non-existent dir so no data.pkl
else:
    DATASET_PATH = str(_raw_dataset) if _raw_dataset is not None else str(DATASET_FOLDER)

# Player runs folder
PLAYER_RUNS_FOLDER = TMRL_FOLDER / "player_runs"
