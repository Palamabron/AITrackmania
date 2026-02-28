"""TMRL configuration: load and expose settings from TmrlData/config/config.json.

Config file layout (config.json):
  __VERSION__     - Minimum compatible config version
  RUN_NAME        - Experiment name (used in paths and wandb)
  BUFFERS_MAXLEN  - Max samples per rollout worker buffer
  RW_MAX_SAMPLES_PER_EPISODE - Cap on steps per episode for workers
  CUDA_TRAINING   - Use GPU for trainer
  CUDA_INFERENCE - Use GPU for rollout workers
  VIRTUAL_GAMEPAD - Use gamepad (True) or keyboard (False)
  LOCALHOST_WORKER / LOCALHOST_TRAINER - Use 127.0.0.1 when same machine as server
  PUBLIC_IP_SERVER - Server IP for remote workers/trainers
  ENV             - Environment: RTGYM_INTERFACE, SEED, MAP_NAME, rewards,
                  failure params, image/window
  MODEL           - Training loop, memory size, CNN/RNN/MLP sizes, scheduler
  ALG             - Algorithm (SAC/TQC/REDQSAC), learning rates, gamma, etc.
  DEBUGGER        - Profiling, CRC debug, wandb debug
  WANDB_*         - Weights & Biases project, entity, API key
  PORT, LOCAL_PORT_*, PASSWORD, TLS_*, NB_WORKERS, BUFFER_SIZE, HEADER_SIZE - Networking
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from packaging import version
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Version and paths (required before loading config)
# -----------------------------------------------------------------------------

MINIMUM_CONFIG_VERSION = "0.6.0"
CONFIG_COMPATIBILITY_ERROR_MESSAGE = (
    "Perform a clean installation:\n(1) Uninstall TMRL,\n(2) Delete the TmrlData folder,\n"
    "(3) Reinstall TMRL."
)

SYSTEM = platform.system()
RTGYM_VERSION = "real-time-gym-v1" if SYSTEM == "Windows" else "real-time-gym-ts-v1"

TMRL_FOLDER = Path.home() / "TmrlData"
if not TMRL_FOLDER.exists():
    raise RuntimeError(f"Missing folder: {TMRL_FOLDER}")

# Load .env secrets from current working dir and TmrlData.
load_dotenv()
load_dotenv(TMRL_FOLDER / ".env")
load_dotenv(TMRL_FOLDER / "config" / ".env")

CONFIG_FILE_PATH = TMRL_FOLDER / "config" / "config.json"
with open(CONFIG_FILE_PATH) as f:
    TMRL_CONFIG = json.load(f)

# Environment variables take precedence over config for sensitive values.
env_wandb_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY")
if env_wandb_key:
    TMRL_CONFIG["WANDB_KEY"] = env_wandb_key
env_password = os.getenv("TMRL_PASSWORD")
if env_password:
    TMRL_CONFIG["PASSWORD"] = env_password


def _deep_merge_defaults(cfg: dict, default: dict) -> None:
    """In-place merge default into cfg; only fill missing keys."""
    for key, default_val in default.items():
        if key not in cfg:
            cfg[key] = default_val
        elif isinstance(default_val, dict) and isinstance(cfg[key], dict):
            _deep_merge_defaults(cfg[key], default_val)


# Default config for missing sections (e.g. incomplete or old config.json)
_DEFAULT_TMRL_CONFIG = {
    "RUN_NAME": "tmrl_run",
    "BUFFERS_MAXLEN": 500000,
    "RW_MAX_SAMPLES_PER_EPISODE": 1000,
    "CUDA_TRAINING": True,
    "CUDA_INFERENCE": False,
    "VIRTUAL_GAMEPAD": True,
    "LOCALHOST_WORKER": True,
    "LOCALHOST_TRAINER": True,
    "PUBLIC_IP_SERVER": "0.0.0.0",
    "PASSWORD": "YourRandomPasswordHere",
    "TLS": False,
    "TLS_HOSTNAME": "default",
    "TLS_CREDENTIALS_DIRECTORY": "",
    "NB_WORKERS": -1,
    "WANDB_PROJECT": "tmrl",
    "WANDB_ENTITY": "tmrl",
    "WANDB_KEY": "YourWandbApiKey",
    "WANDB_GRADIENTS": False,
    "WANDB_DEBUG_REWARD": True,
    "WANDB_WORKER": True,
    "PLAYER_RUNS": {
        "ONLINE_INJECTION": False,
        "SOURCE_PATH": "",
        "CONSUME_ON_READ": True,
        "MAX_FILES_PER_UPDATE": 1,
        "DEMO_INJECTION_REPEAT": 1,
    },
    "PORT": 55555,
    "LOCAL_PORT_SERVER": 55556,
    "LOCAL_PORT_TRAINER": 55557,
    "LOCAL_PORT_WORKER": 55558,
    "BUFFER_SIZE": 536870912,
    "HEADER_SIZE": 12,
    "MODEL": {
        "MAX_EPOCHS": 10000,
        "ROUNDS_PER_EPOCH": 100,
        "TRAINING_STEPS_PER_ROUND": 200,
        "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 4.0,
        "ENVIRONMENT_STEPS_BEFORE_TRAINING": 1000,
        "UPDATE_MODEL_INTERVAL": 200,
        "UPDATE_BUFFER_INTERVAL": 200,
        "SAVE_MODEL_EVERY": 0,
        "MEMORY_SIZE": 1000000,
        "BATCH_SIZE": 256,
        "SCHEDULER": {"NAME": "", "T_0": 100, "T_mult": 2, "eta_min": 1.0e-6, "last_epoch": -1},
        "NOISY_LINEAR_CRITIC": False,
        "NOISY_LINEAR_ACTOR": False,
        "OUTPUT_DROPOUT": 0.0,
        "RNN_DROPOUT": 0.0,
        "CNN_FILTERS": [32, 64, 64, 64],
        "CNN_OUTPUT_SIZE": 256,
        "RNN_LENS": [1],
        "RNN_SIZES": [64],
        "API_MLP_SIZES": [256, 256],
        "API_LAYERNORM": True,
        "MLP_LAYERNORM": False,
        "USE_RESIDUAL_MLP": False,
        "RESIDUAL_MLP_HIDDEN_DIM": 256,
        "RESIDUAL_MLP_NUM_BLOCKS": 6,
        "USE_RESIDUAL_SOPHY": False,
        "USE_FROZEN_EFFNET": False,
        "FROZEN_EFFNET_EMBED_DIM": 256,
        "FROZEN_EFFNET_WIDTH_MULT": 0.5,
    },
    "ALG": {
        "ALGORITHM": "SAC",
        "LEARN_ENTROPY_COEF": False,
        "LR_ACTOR": 1.0e-5,
        "LR_CRITIC": 5.0e-5,
        "LR_ENTROPY": 3.0e-4,
        "GAMMA": 0.995,
        "POLYAK": 0.995,
        "TARGET_ENTROPY": -0.5,
        "ALPHA": 0.01,
        "REDQ_N": 10,
        "REDQ_M": 2,
        "REDQ_Q_UPDATES_PER_POLICY_UPDATE": 20,
        "TOP_QUANTILES_TO_DROP": 2,
        "QUANTILES_NUMBER": 1,
        "N_STEPS": 1,
        "CLIPPING_WEIGHTS": False,
        "CLIP_WEIGHTS_VALUE": 1.0,
        "ACTOR_WEIGHT_DECAY": 0.0,
        "CRITIC_WEIGHT_DECAY": 0.0,
        "NUMBER_OF_POINTS": 0,
        "POINTS_DISTANCE": 0.0,
        "SPEED_BONUS": 0.0,
        "SPEED_MIN_THRESHOLD": 0.0,
        "SPEED_MEDIUM_THRESHOLD": 0.0,
        "ADAM_EPS": 1.0e-8,
        "R2D2_REWIND": 0.5,
        "OPTIMIZER_ACTOR": "adam",
        "OPTIMIZER_CRITIC": "adam",
        "BETAS_ACTOR": [0.9, 0.999],
        "BETAS_CRITIC": [0.9, 0.999],
        "L2_ACTOR": 0.0,
        "L2_CRITIC": 0.0,
        "GRAD_CLIP_ACTOR": 1.0,
        "GRAD_CLIP_CRITIC": 1.0,
        "BACKUP_CLIP_RANGE": 100.0,
        "MIXED_PRECISION": True,
        "MIXED_PRECISION_DTYPE": "bfloat16",
    },
}
_deep_merge_defaults(TMRL_CONFIG, _DEFAULT_TMRL_CONFIG)

assert "__VERSION__" in TMRL_CONFIG, (
    "config.json is outdated. " + CONFIG_COMPATIBILITY_ERROR_MESSAGE
)
CONFIG_VERSION = TMRL_CONFIG["__VERSION__"]
assert version.parse(CONFIG_VERSION) >= version.parse(MINIMUM_CONFIG_VERSION), (
    f"config.json version ({CONFIG_VERSION}) must be >= {MINIMUM_CONFIG_VERSION}. "
    + CONFIG_COMPATIBILITY_ERROR_MESSAGE
)

# -----------------------------------------------------------------------------
# Paths (all under TmrlData)
# -----------------------------------------------------------------------------

CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
DATASET_FOLDER = TMRL_FOLDER / "dataset"
REWARD_FOLDER = TMRL_FOLDER / "reward"
TRACK_FOLDER = TMRL_FOLDER / "track"
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
CONFIG_FOLDER = TMRL_FOLDER / "config"

# -----------------------------------------------------------------------------
# Run identity and buffers (top-level in config.json)
# -----------------------------------------------------------------------------

RUN_NAME = TMRL_CONFIG["RUN_NAME"]
BUFFERS_MAXLEN = TMRL_CONFIG["BUFFERS_MAXLEN"]
RW_MAX_SAMPLES_PER_EPISODE = TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# -----------------------------------------------------------------------------
# Hardware and control
# -----------------------------------------------------------------------------

CUDA_TRAINING = TMRL_CONFIG["CUDA_TRAINING"]
CUDA_INFERENCE = TMRL_CONFIG["CUDA_INFERENCE"]
USE_VIRTUAL_GAMEPAD = TMRL_CONFIG["VIRTUAL_GAMEPAD"]
USE_RNN = False

# Backward compatibility (used in config_objects and elsewhere)
PRAGMA_RNN = USE_RNN
PRAGMA_GAMEPAD = USE_VIRTUAL_GAMEPAD

# -----------------------------------------------------------------------------
# Network: where to connect (worker/trainer → server)
# -----------------------------------------------------------------------------

LOCALHOST_WORKER = TMRL_CONFIG["LOCALHOST_WORKER"]
LOCALHOST_TRAINER = TMRL_CONFIG["LOCALHOST_TRAINER"]
PUBLIC_IP_SERVER = TMRL_CONFIG["PUBLIC_IP_SERVER"]
SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"

# -----------------------------------------------------------------------------
# Environment (config.json → ENV): observation type, rewards, failure, image
# -----------------------------------------------------------------------------


class EnvConfig(BaseModel):
    """Environment and reward settings from config.json ENV section."""

    RTGYM_INTERFACE: str = Field(
        description="e.g. LIDAR, LIDARPROGRESS, TRACKMAP, FULL, MOBILEV3, ..."
    )
    SEED: int = 0
    MAP_NAME: str = ""
    MIN_NB_ZERO_REW_BEFORE_FAILURE: int = Field(
        description="Episode ends after this many steps with zero reward"
    )
    MAX_NB_ZERO_REW_BEFORE_FAILURE: int = 0
    MIN_NB_STEPS_BEFORE_FAILURE: int = Field(
        description="Minimum steps before failure condition can trigger"
    )
    OSCILLATION_PERIOD: int = 0
    NB_OBS_FORWARD: int = 0
    CRASH_PENALTY: float = 0.0
    CRASH_COOLDOWN: int = 0
    CONSTANT_PENALTY: float = Field(description="Per-step penalty (e.g. -abs(speed))")
    LAP_REWARD: float = 0.0
    LAP_COOLDOWN: int = 0
    CHECKPOINT_REWARD: float = 0.0
    END_OF_TRACK_REWARD: float = 0.0
    USE_IMAGES: bool = True
    SLEEP_TIME_AT_RESET: float = 0.0
    IMG_HIST_LEN: int = 4
    WINDOW_WIDTH: int = 640
    WINDOW_HEIGHT: int = 480
    IMG_GRAYSCALE: bool = True
    IMG_WIDTH: int = 64
    IMG_HEIGHT: int = 64
    LINUX_X_OFFSET: int = 64
    LINUX_Y_OFFSET: int = 70
    IMG_SCALE_CHECK_ENV: float = 1.0
    REWARD_CONFIG: dict = Field(
        default_factory=dict, description="REWARD_CONFIG dict for RewardFunction"
    )
    INIT_GAS_BIAS: float = Field(
        default=0.0,
        description="Bias for actor output dim 0 (gas) before tanh; e.g. 0.8 => default forward",
    )
    model_config = {"extra": "allow"}


_raw_env = dict(TMRL_CONFIG["ENV"])
# Migrate legacy finish reward key from REWARD_CONFIG into ENV.END_OF_TRACK_REWARD.
# Keep backward compatibility for old configs while enforcing one canonical key at runtime.
_legacy_finish_reward = None
if isinstance(_raw_env.get("REWARD_CONFIG"), dict):
    _legacy_finish_reward = _raw_env["REWARD_CONFIG"].get("END_OF_TRACK")
if "END_OF_TRACK_REWARD" not in _raw_env and _legacy_finish_reward is not None:
    _raw_env["END_OF_TRACK_REWARD"] = _legacy_finish_reward
if _legacy_finish_reward is not None and "END_OF_TRACK_REWARD" in _raw_env:
    try:
        if float(_legacy_finish_reward) != float(_raw_env["END_OF_TRACK_REWARD"]):
            logger.warning(
                "Config contains both ENV.END_OF_TRACK_REWARD={} and legacy "
                "ENV.REWARD_CONFIG.END_OF_TRACK={}. Using END_OF_TRACK_REWARD.",
                _raw_env["END_OF_TRACK_REWARD"],
                _legacy_finish_reward,
            )
    except Exception:
        logger.warning(
            "Could not compare END_OF_TRACK_REWARD and legacy REWARD_CONFIG.END_OF_TRACK. "
            "Using END_OF_TRACK_REWARD."
        )
# Remove legacy key to avoid redundant configuration branches in runtime code.
if isinstance(_raw_env.get("REWARD_CONFIG"), dict):
    _raw_env["REWARD_CONFIG"].pop("END_OF_TRACK", None)
# Fill missing ENV keys with defaults (handles older or incomplete config.json)
_default_env = {
    "SEED": 0,
    "MAP_NAME": "",
    "MIN_NB_ZERO_REW_BEFORE_FAILURE": 0,
    "MAX_NB_ZERO_REW_BEFORE_FAILURE": 0,
    "MIN_NB_STEPS_BEFORE_FAILURE": 0,
    "OSCILLATION_PERIOD": 0,
    "NB_OBS_FORWARD": 0,
    "CRASH_PENALTY": 0.0,
    "CRASH_COOLDOWN": 0,
    "CONSTANT_PENALTY": 0.0,
    "LAP_REWARD": 0.0,
    "LAP_COOLDOWN": 0,
    "CHECKPOINT_REWARD": 0.0,
    "END_OF_TRACK_REWARD": 0.0,
    "USE_IMAGES": True,
    "SLEEP_TIME_AT_RESET": 1.5,
    "IMG_HIST_LEN": 4,
    "WINDOW_WIDTH": 640,
    "WINDOW_HEIGHT": 480,
    "IMG_GRAYSCALE": True,
    "IMG_WIDTH": 64,
    "IMG_HEIGHT": 64,
    "LINUX_X_OFFSET": 64,
    "LINUX_Y_OFFSET": 70,
    "IMG_SCALE_CHECK_ENV": 1.0,
    "INIT_GAS_BIAS": 0.0,
    "REWARD_CONFIG": {
        "WALL_HUG_SPEED_THRESHOLD": 10.0,
        "WALL_HUG_PENALTY_FACTOR": 0.005,
        "PROXIMITY_REWARD_SHAPING": 0.1,
        "REWARD_SCALE": 3.0,
        "SPEED_TERMINAL_SCALE": 0.0,
        "PROJECTED_VELOCITY_SCALE": 0.1,
        "STEERING_DELTA_PENALTY": 0.1,
        "MAX_TRACK_WIDTH": 23.5,
        "BOUNDARY_PENALTY_WEIGHT": 2.0,
        "BOUNDARY_CRASH_PENALTY": 10.0,
        "REWARD_CLIP_FLOOR": 10.0,
        "TIME_BONUS_SCALE": 0.0,
        "CONDITIONAL_PENALTY_WHEN_BRAKING": False,
        "BRAKE_THRESHOLD": 0.3,
        "TRACK_LOOK_AHEAD_PCT": 0.0,
        "TRACK_POINT_SPACING_M": 0.0,
    },
}
if "RTGYM_CONFIG" not in _raw_env:
    _default_env["RTGYM_CONFIG"] = {
        "time_step_duration": 0.05,
        "start_obs_capture": 0.04,
        "time_step_timeout_factor": 1.0,
        "act_buf_len": 2,
        "benchmark": False,
        "wait_on_done": True,
        "ep_max_length": 1000,
    }
for _k, _v in _default_env.items():
    if _k not in _raw_env:
        _raw_env[_k] = _v
TMRL_CONFIG["ENV"] = _raw_env  # so create_config() and others see full ENV
ENV_CONFIG = _raw_env  # keep dict for code that indexes by key

# Observation type flags (derived from RTGYM_INTERFACE string)
RTGYM_INTERFACE = str(_raw_env["RTGYM_INTERFACE"]).upper()
USE_LIDAR_OBSERVATIONS = RTGYM_INTERFACE.endswith("LIDAR")
USE_CUSTOM_BACKBONE = RTGYM_INTERFACE.endswith("MOBILEV3") or RTGYM_INTERFACE.endswith("CUSTOM")
USE_LIDAR_PROGRESS = RTGYM_INTERFACE.endswith("LIDARPROGRESS")
USE_LIDAR_PROGRESS_IMAGES = "LIDARPROGRESSIMAGES" in RTGYM_INTERFACE
USE_TRACKMAP = RTGYM_INTERFACE.endswith("TRACKMAP")
USE_TRACKMAP_IMAGES = "TRACKMAPIMAGES" in RTGYM_INTERFACE
USE_BEST_INTERFACE = RTGYM_INTERFACE.endswith("BEST")
USE_BEST_TQC = RTGYM_INTERFACE.endswith("BEST_TQC")
USE_MBEST_TQC = RTGYM_INTERFACE.endswith("MTQC")
USE_TQC_GRAB = "TQCGRAB" in RTGYM_INTERFACE

PRAGMA_LIDAR = USE_LIDAR_OBSERVATIONS
PRAGMA_CUSTOM = USE_CUSTOM_BACKBONE
PRAGMA_PROGRESS = USE_LIDAR_PROGRESS
PRAGMA_LIDAR_PROGRESS_IMAGES = USE_LIDAR_PROGRESS_IMAGES
PRAGMA_TRACKMAP = USE_TRACKMAP
PRAGMA_TRACKMAP_IMAGES = USE_TRACKMAP_IMAGES
PRAGMA_BEST = USE_BEST_INTERFACE
PRAGMA_BEST_TQC = USE_BEST_TQC
PRAGMA_MBEST_TQC = USE_MBEST_TQC
PRAGMA_TQC_GRAB = USE_TQC_GRAB

if USE_LIDAR_PROGRESS or USE_TRACKMAP or USE_LIDAR_PROGRESS_IMAGES or USE_TRACKMAP_IMAGES:
    USE_LIDAR_OBSERVATIONS = True
    PRAGMA_LIDAR = True

# Env scalars (used everywhere)
SEED = _raw_env["SEED"]
MAP_NAME = _raw_env["MAP_NAME"]
MIN_NB_ZERO_REW_BEFORE_FAILURE = _raw_env["MIN_NB_ZERO_REW_BEFORE_FAILURE"]
MAX_NB_ZERO_REW_BEFORE_FAILURE = _raw_env["MAX_NB_ZERO_REW_BEFORE_FAILURE"]
MIN_NB_STEPS_BEFORE_FAILURE = _raw_env["MIN_NB_STEPS_BEFORE_FAILURE"]
OSCILLATION_PERIOD = _raw_env["OSCILLATION_PERIOD"]
NB_OBS_FORWARD = _raw_env["NB_OBS_FORWARD"]
CRASH_PENALTY = _raw_env["CRASH_PENALTY"]
CRASH_COOLDOWN = _raw_env["CRASH_COOLDOWN"]
CONSTANT_PENALTY = _raw_env["CONSTANT_PENALTY"]
LAP_REWARD = _raw_env["LAP_REWARD"]
LAP_COOLDOWN = _raw_env["LAP_COOLDOWN"]
CHECKPOINT_REWARD = _raw_env["CHECKPOINT_REWARD"]
END_OF_TRACK_REWARD = _raw_env["END_OF_TRACK_REWARD"]
USE_IMAGES = _raw_env["USE_IMAGES"]
LIDAR_BLACK_THRESHOLD = [55, 55, 55]
REWARD_CONFIG = _raw_env.get("REWARD_CONFIG", {})
SLEEP_TIME_AT_RESET = _raw_env["SLEEP_TIME_AT_RESET"]
IMG_HIST_LEN = _raw_env["IMG_HIST_LEN"]
ACT_BUF_LEN = _raw_env["RTGYM_CONFIG"]["act_buf_len"]
WINDOW_WIDTH = _raw_env["WINDOW_WIDTH"]
WINDOW_HEIGHT = _raw_env["WINDOW_HEIGHT"]
GRAYSCALE = _raw_env.get("IMG_GRAYSCALE", False)
IMG_WIDTH = _raw_env.get("IMG_WIDTH", 64)
IMG_HEIGHT = _raw_env.get("IMG_HEIGHT", 64)
LINUX_X_OFFSET = _raw_env.get("LINUX_X_OFFSET", 64)
LINUX_Y_OFFSET = _raw_env.get("LINUX_Y_OFFSET", 70)
IMG_SCALE_CHECK_ENV = _raw_env.get("IMG_SCALE_CHECK_ENV", 1.0)
INIT_GAS_BIAS = _raw_env.get("INIT_GAS_BIAS", 0.0)

# -----------------------------------------------------------------------------
# Debug / profiling (config.json → DEBUGGER)
# -----------------------------------------------------------------------------


class DebuggerConfig(BaseModel):
    """Debug and profiling options from config.json DEBUGGER section."""

    DEBUG_MODE: bool = False
    CRC_DEBUG: bool = Field(description="Enable CRC checks on samples (pipeline consistency)")
    CRC_DEBUG_SAMPLES: int = 0
    PROFILE_TRAINER: bool = Field(description="Profile each epoch with Python profiler")
    WANDB_DEBUG: bool = True
    PYTORCH_PROFILER: bool = False
    model_config = {"extra": "allow"}


_debugger_raw = TMRL_CONFIG.get(
    "DEBUGGER",
    {
        "DEBUG_MODE": False,
        "CRC_DEBUG": False,
        "CRC_DEBUG_SAMPLES": 0,
        "PROFILE_TRAINER": False,
        "WANDB_DEBUG": True,
        "PYTORCH_PROFILER": False,
    },
)
DEBUGGER_CONFIG = DebuggerConfig(**_debugger_raw)
DEBUGGER = _debugger_raw
DEBUG_MODE = DEBUGGER_CONFIG.DEBUG_MODE
CRC_DEBUG = DEBUGGER_CONFIG.CRC_DEBUG
CRC_DEBUG_SAMPLES = DEBUGGER_CONFIG.CRC_DEBUG_SAMPLES
PROFILE_TRAINER = DEBUGGER_CONFIG.PROFILE_TRAINER
SYNCHRONIZE_CUDA = PROFILE_TRAINER
WANDB_DEBUG = DEBUGGER_CONFIG.WANDB_DEBUG
PYTORCH_PROFILER = DEBUGGER_CONFIG.PYTORCH_PROFILER

PATH_DATA = TMRL_FOLDER
logger.debug(f"PATH_DATA: {PATH_DATA}")

# -----------------------------------------------------------------------------
# Model (config.json → MODEL): training loop, architecture, scheduler
# -----------------------------------------------------------------------------

MODEL_CONFIG = TMRL_CONFIG["MODEL"]
MODEL_HISTORY = MODEL_CONFIG["SAVE_MODEL_EVERY"]

# File paths built from RUN_NAME and MAP_NAME
MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))
REWARDS_CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_rew_" + MAP_NAME + "_t.tcpt"))
DATASET_PATH = str(DATASET_FOLDER)
REWARD_PATH = str(REWARD_FOLDER / ("reward_" + MAP_NAME + ".pkl"))
TRACK_PATH_LEFT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_left.pkl"))
TRACK_PATH_RIGHT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_right.pkl"))

# Track look-ahead: when TRACK_LOOK_AHEAD_PCT and TRACK_POINT_SPACING_M are both > 0,
# points_number is derived from trajectory length (see k% of track ahead, point every D m).
TRACK_POINTS_NUMBER = None
_reward_cfg = _raw_env.get("REWARD_CONFIG", {})
_track_pct = float(_reward_cfg.get("TRACK_LOOK_AHEAD_PCT", 0.0))
_track_spacing = float(_reward_cfg.get("TRACK_POINT_SPACING_M", 0.0))
if _track_pct > 0 and _track_spacing > 0 and os.path.exists(REWARD_PATH):
    try:
        import math
        import pickle

        with open(REWARD_PATH, "rb") as _f:
            _traj = pickle.load(_f)
        if hasattr(_traj, "__len__") and len(_traj) > 1:
            import numpy as _np

            _traj = _np.asarray(_traj)
            _diffs = _np.linalg.norm(_np.diff(_traj, axis=0), axis=1)
            _cum = _np.zeros(len(_traj))
            _np.cumsum(_diffs, out=_cum[1:])
            _L = float(_cum[-1])
            if _L >= _track_spacing:
                _nb = int(math.ceil(_L * (_track_pct / 100.0) / _track_spacing))
                TRACK_POINTS_NUMBER = min(200, max(1, _nb))
                logger.info(
                    "Track look-ahead: TRACK_LOOK_AHEAD_PCT={:.2f}%, TRACK_POINT_SPACING_M={:.2f} m, "
                    "trajectory length={:.1f} m -> POINTS_NUMBER={}",
                    _track_pct,
                    _track_spacing,
                    _L,
                    TRACK_POINTS_NUMBER,
                )
    except Exception:
        pass

# Player runs: recording/importing/injection
PLAYER_RUNS_FOLDER = TMRL_FOLDER / "player_runs"
_player_runs_cfg = TMRL_CONFIG.get("PLAYER_RUNS", {})
PLAYER_RUNS_ONLINE_INJECTION = bool(_player_runs_cfg.get("ONLINE_INJECTION", False))
PLAYER_RUNS_SOURCE_PATH = (
    str(_player_runs_cfg.get("SOURCE_PATH"))
    if _player_runs_cfg.get("SOURCE_PATH")
    else str(PLAYER_RUNS_FOLDER)
)
PLAYER_RUNS_CONSUME_ON_READ = bool(_player_runs_cfg.get("CONSUME_ON_READ", False))
PLAYER_RUNS_MAX_FILES_PER_UPDATE = int(_player_runs_cfg.get("MAX_FILES_PER_UPDATE", 1))
PLAYER_RUNS_DEMO_INJECTION_REPEAT = max(1, int(_player_runs_cfg.get("DEMO_INJECTION_REPEAT", 1)))
PLAYER_RUNS_DEMO_WEIGHT_DECAY_SLOWDOWN = max(
    0.1, float(_player_runs_cfg.get("DEMO_WEIGHT_DECAY_SLOWDOWN", 1.0))
)
# PER-like sampling: when > 0, episode sampling weights = (reward_sum + eps)^PER_ALPHA.
# 0 = off (use reward_sums as-is), 0.6 = moderate prioritization of high-return episodes.
PLAYER_RUNS_PER_ALPHA = max(0.0, float(_player_runs_cfg.get("PER_ALPHA", 0.0)))

# -----------------------------------------------------------------------------
# Weights & Biases (config.json → WANDB_*)
# -----------------------------------------------------------------------------

WANDB_RUN_ID = RUN_NAME
WANDB_PROJECT = TMRL_CONFIG["WANDB_PROJECT"]
WANDB_ENTITY = TMRL_CONFIG["WANDB_ENTITY"]
WANDB_KEY = TMRL_CONFIG["WANDB_KEY"]
WANDB_GRADIENTS = TMRL_CONFIG["WANDB_GRADIENTS"]
WANDB_DEBUG_REWARD = TMRL_CONFIG["WANDB_DEBUG_REWARD"]
WANDB_WORKER = TMRL_CONFIG.get("WANDB_WORKER", True)
os.environ["WANDB_API_KEY"] = WANDB_KEY

# -----------------------------------------------------------------------------
# Networking (config.json: PORT, LOCAL_PORT_*, PASSWORD, TLS_*, etc.)
# -----------------------------------------------------------------------------

PRINT_BYTESIZES = True
PORT = TMRL_CONFIG["PORT"]
LOCAL_PORT_SERVER = TMRL_CONFIG["LOCAL_PORT_SERVER"]
LOCAL_PORT_TRAINER = TMRL_CONFIG["LOCAL_PORT_TRAINER"]
LOCAL_PORT_WORKER = TMRL_CONFIG["LOCAL_PORT_WORKER"]
PASSWORD = TMRL_CONFIG["PASSWORD"]
SECURITY = "TLS" if TMRL_CONFIG["TLS"] else None
CREDENTIALS_DIRECTORY = (
    TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"]
    if TMRL_CONFIG.get("TLS_CREDENTIALS_DIRECTORY") != ""
    else None
)
HOSTNAME = TMRL_CONFIG["TLS_HOSTNAME"]
NB_WORKERS = None if TMRL_CONFIG["NB_WORKERS"] < 0 else TMRL_CONFIG["NB_WORKERS"]
BUFFER_SIZE = TMRL_CONFIG["BUFFER_SIZE"]
HEADER_SIZE = TMRL_CONFIG["HEADER_SIZE"]

# -----------------------------------------------------------------------------
# Model architecture (from MODEL: CNN/RNN/MLP sizes, dropout, scheduler)
# -----------------------------------------------------------------------------

SCHEDULER_CONFIG = MODEL_CONFIG["SCHEDULER"]
NOISY_LINEAR_CRITIC = MODEL_CONFIG["NOISY_LINEAR_CRITIC"]
NOISY_LINEAR_ACTOR = MODEL_CONFIG["NOISY_LINEAR_ACTOR"]
OUTPUT_DROPOUT = MODEL_CONFIG["OUTPUT_DROPOUT"]
RNN_DROPOUT = MODEL_CONFIG["RNN_DROPOUT"]
CNN_FILTERS = MODEL_CONFIG["CNN_FILTERS"]
CNN_OUTPUT_SIZE = MODEL_CONFIG["CNN_OUTPUT_SIZE"]
RNN_LENS = MODEL_CONFIG["RNN_LENS"]
RNN_SIZES = MODEL_CONFIG["RNN_SIZES"]
API_MLP_SIZES = MODEL_CONFIG["API_MLP_SIZES"]
API_LAYERNORM = MODEL_CONFIG["API_LAYERNORM"]
MLP_LAYERNORM = MODEL_CONFIG["MLP_LAYERNORM"]
USE_RESIDUAL_MLP = MODEL_CONFIG.get("USE_RESIDUAL_MLP", False)
RESIDUAL_MLP_HIDDEN_DIM = MODEL_CONFIG.get("RESIDUAL_MLP_HIDDEN_DIM", 256)
RESIDUAL_MLP_NUM_BLOCKS = MODEL_CONFIG.get("RESIDUAL_MLP_NUM_BLOCKS", 6)
USE_RESIDUAL_SOPHY = MODEL_CONFIG.get("USE_RESIDUAL_SOPHY", False)
USE_FROZEN_EFFNET = MODEL_CONFIG.get("USE_FROZEN_EFFNET", False)
FROZEN_EFFNET_EMBED_DIM = MODEL_CONFIG.get("FROZEN_EFFNET_EMBED_DIM", 256)
FROZEN_EFFNET_WIDTH_MULT = MODEL_CONFIG.get("FROZEN_EFFNET_WIDTH_MULT", 0.5)

# -----------------------------------------------------------------------------
# Algorithm (config.json → ALG): SAC/TQC/REDQSAC, LRs, gamma, quantiles, etc.
# -----------------------------------------------------------------------------

ALG_CONFIG = TMRL_CONFIG["ALG"]
if ALG_CONFIG["ALGORITHM"] != "TQC" and ALG_CONFIG["QUANTILES_NUMBER"] > 1:
    raise ValueError("QUANTILES_NUMBER must be 1 if it is used with SAC")
QUANTILES_NUMBER = ALG_CONFIG["QUANTILES_NUMBER"]
N_STEPS = 1 if ALG_CONFIG["N_STEPS"] <= 0 else ALG_CONFIG["N_STEPS"]
WEIGHT_CLIPPING_ENABLED = ALG_CONFIG["CLIPPING_WEIGHTS"]
WEIGHT_CLIPPING_VALUE = 1.0 if not WEIGHT_CLIPPING_ENABLED else ALG_CONFIG["CLIP_WEIGHTS_VALUE"]
ACTOR_WEIGHT_DECAY = ALG_CONFIG["ACTOR_WEIGHT_DECAY"]
CRITIC_WEIGHT_DECAY = ALG_CONFIG["CRITIC_WEIGHT_DECAY"]
POINTS_NUMBER = (
    TRACK_POINTS_NUMBER if TRACK_POINTS_NUMBER is not None else ALG_CONFIG["NUMBER_OF_POINTS"]
)
POINTS_DISTANCE = ALG_CONFIG["POINTS_DISTANCE"]
SPEED_BONUS = ALG_CONFIG["SPEED_BONUS"]
SPEED_MIN_THRESHOLD = ALG_CONFIG["SPEED_MIN_THRESHOLD"]
SPEED_MEDIUM_THRESHOLD = ALG_CONFIG["SPEED_MEDIUM_THRESHOLD"]
ADAM_EPS = ALG_CONFIG["ADAM_EPS"]
GRAD_CLIP_ACTOR = float(ALG_CONFIG.get("GRAD_CLIP_ACTOR", 1.0))
GRAD_CLIP_CRITIC = float(ALG_CONFIG.get("GRAD_CLIP_CRITIC", 1.0))
BACKUP_CLIP_RANGE = float(ALG_CONFIG.get("BACKUP_CLIP_RANGE", 100.0))


# -----------------------------------------------------------------------------
# create_config(): flat dict for training agent / checkpoint loading
# -----------------------------------------------------------------------------


def create_config() -> dict:
    """Build a flat training config dict from TMRL_CONFIG for the training agent.

    Merges model, environment, algorithm and scheduler entries into a single
    dict expected by the custom algorithms (e.g. SAC/TQC). Used when loading
    checkpoints or initializing agents that need all hyperparameters in one place.

    Returns:
        A single-level dict with keys like TRAINING_STEPS_PER_ROUND, LR_ACTOR,
        CNN_FILTERS, RNN_SIZES, CRASH_PENALTY, GAMMA, etc.

    Steps:
        1. Copy training loop and memory settings from model config.
        2. Flatten CNN/RNN/API MLP sizes into CNN_FILTER0..n, RNN_SIZE0..n, etc.
        3. Copy model flags (noisy linear, dropout, layer norm).
        4. Copy environment failure and reward parameters.
        5. Copy algorithm hyperparameters (LRs, gamma, polyak, quantiles, etc.).
        6. Copy scheduler parameters (T_0, T_mult, eta_min, last_epoch).
        7. Copy image config (width, height, grayscale, history length).
    """
    training_config: dict = {}
    alg_config = TMRL_CONFIG["ALG"]
    model_config = TMRL_CONFIG["MODEL"]
    scheduler_config = model_config["SCHEDULER"]
    env_config = TMRL_CONFIG["ENV"]

    training_config["TRAINING_STEPS_PER_ROUND"] = model_config["TRAINING_STEPS_PER_ROUND"]
    training_config["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"] = model_config[
        "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"
    ]
    training_config["ENVIRONMENT_STEPS_BEFORE_TRAINING"] = model_config[
        "ENVIRONMENT_STEPS_BEFORE_TRAINING"
    ]
    training_config["UPDATE_MODEL_INTERVAL"] = model_config["UPDATE_MODEL_INTERVAL"]
    training_config["UPDATE_BUFFER_INTERVAL"] = model_config["UPDATE_BUFFER_INTERVAL"]
    training_config["SAVE_MODEL_EVERY"] = model_config["SAVE_MODEL_EVERY"]
    training_config["MEMORY_SIZE"] = model_config["MEMORY_SIZE"]
    training_config["BATCH_SIZE"] = model_config["BATCH_SIZE"]

    training_config["CNN_FILTERS"] = model_config["CNN_FILTERS"]
    for layer_index, filter_size in enumerate(training_config["CNN_FILTERS"]):
        training_config[f"CNN_FILTER{layer_index}"] = filter_size
    training_config["CNN_OUTPUT_SIZE"] = model_config["CNN_OUTPUT_SIZE"]

    training_config["RNN_SIZES"] = model_config["RNN_SIZES"]
    for layer_index, size in enumerate(training_config["RNN_SIZES"]):
        training_config[f"RNN_SIZE{layer_index}"] = size
    training_config["RNN_LENS"] = model_config["RNN_LENS"]
    for layer_index, length in enumerate(training_config["RNN_LENS"]):
        training_config[f"RNN_LEN{layer_index}"] = length

    training_config["API_MLP_SIZES"] = model_config["API_MLP_SIZES"]
    for layer_index, size in enumerate(training_config["API_MLP_SIZES"]):
        training_config[f"API_MLP_SIZE{layer_index}"] = size

    training_config["API_LAYERNORM"] = model_config["API_LAYERNORM"]
    training_config["NOISY_LINEAR_ACTOR"] = model_config["NOISY_LINEAR_ACTOR"]
    training_config["NOISY_LINEAR_CRITIC"] = model_config["NOISY_LINEAR_CRITIC"]
    training_config["RNN_DROPOUT"] = model_config["RNN_DROPOUT"]

    training_config["USE_RESIDUAL_MLP"] = model_config.get("USE_RESIDUAL_MLP", False)
    training_config["RESIDUAL_MLP_HIDDEN_DIM"] = model_config.get("RESIDUAL_MLP_HIDDEN_DIM", 256)
    training_config["RESIDUAL_MLP_NUM_BLOCKS"] = model_config.get("RESIDUAL_MLP_NUM_BLOCKS", 6)
    training_config["USE_RESIDUAL_SOPHY"] = model_config.get("USE_RESIDUAL_SOPHY", False)
    training_config["USE_FROZEN_EFFNET"] = model_config.get("USE_FROZEN_EFFNET", False)
    training_config["FROZEN_EFFNET_EMBED_DIM"] = model_config.get("FROZEN_EFFNET_EMBED_DIM", 256)
    training_config["FROZEN_EFFNET_WIDTH_MULT"] = model_config.get("FROZEN_EFFNET_WIDTH_MULT", 0.5)

    training_config["MIN_NB_ZERO_REW_BEFORE_FAILURE"] = env_config["MIN_NB_ZERO_REW_BEFORE_FAILURE"]
    training_config["MAX_NB_ZERO_REW_BEFORE_FAILURE"] = env_config["MAX_NB_ZERO_REW_BEFORE_FAILURE"]
    training_config["MIN_NB_STEPS_BEFORE_FAILURE"] = env_config["MIN_NB_STEPS_BEFORE_FAILURE"]
    training_config["OSCILLATION_PERIOD"] = env_config["OSCILLATION_PERIOD"]
    training_config["CRASH_PENALTY"] = env_config["CRASH_PENALTY"]
    training_config["CRASH_COOLDOWN"] = env_config["CRASH_COOLDOWN"]
    training_config["CONSTANT_PENALTY"] = env_config["CONSTANT_PENALTY"]
    training_config["LAP_REWARD"] = env_config["LAP_REWARD"]
    training_config["LAP_COOLDOWN"] = env_config["LAP_COOLDOWN"]
    training_config["CHECKPOINT_REWARD"] = env_config["CHECKPOINT_REWARD"]
    training_config["CHECKPOINT_COOLDOWN"] = env_config.get("CHECKPOINT_COOLDOWN", 0)
    training_config["REWARD_END_OF_TRACK"] = env_config["END_OF_TRACK_REWARD"]

    training_config["ALGORITHM"] = alg_config["ALGORITHM"]
    training_config["QUANTILES_NUMBER"] = alg_config["QUANTILES_NUMBER"]
    training_config["LEARN_ENTROPY_COEF"] = alg_config["LEARN_ENTROPY_COEF"]
    training_config["LR_ACTOR"] = alg_config["LR_ACTOR"]
    training_config["LR_CRITIC"] = alg_config["LR_CRITIC"]
    training_config["LR_CRITIC_DIVIDED_BY_LR_ACTOR"] = (
        training_config["LR_CRITIC"] / training_config["LR_ACTOR"]
    )
    training_config["N_STEPS"] = alg_config["N_STEPS"]
    training_config["ACTOR_WEIGHT_DECAY"] = alg_config["ACTOR_WEIGHT_DECAY"]
    training_config["CRITIC_WEIGHT_DECAY"] = alg_config["CRITIC_WEIGHT_DECAY"]
    training_config["CLIPPING_WEIGHTS"] = alg_config["CLIPPING_WEIGHTS"]
    training_config["CLIP_WEIGHTS_VALUE"] = (
        1.0 if not training_config["CLIPPING_WEIGHTS"] else alg_config["CLIP_WEIGHTS_VALUE"]
    )
    training_config["POINTS_NUMBER"] = POINTS_NUMBER
    training_config["POINTS_DISTANCE"] = alg_config["POINTS_DISTANCE"]
    training_config["SPEED_BONUS"] = alg_config["SPEED_BONUS"]
    training_config["SPEED_MIN_THRESHOLD"] = alg_config["SPEED_MIN_THRESHOLD"]
    training_config["SPEED_MEDIUM_THRESHOLD"] = alg_config["SPEED_MEDIUM_THRESHOLD"]
    training_config["LR_ENTROPY"] = alg_config["LR_ENTROPY"]
    training_config["GAMMA"] = alg_config["GAMMA"]
    training_config["POLYAK"] = alg_config["POLYAK"]
    training_config["TARGET_ENTROPY"] = alg_config["TARGET_ENTROPY"]
    training_config["TOP_QUANTILES_TO_DROP"] = alg_config["TOP_QUANTILES_TO_DROP"]

    if alg_config["QUANTILES_NUMBER"] != 1 and alg_config["ALGORITHM"] == "SAC":
        raise ValueError("SAC can be only used if the QUANTILES_NUMBER equals to 1")

    training_config["R2D2_REWIND"] = alg_config["R2D2_REWIND"]
    training_config["ADAM_EPS"] = alg_config["ADAM_EPS"]

    training_config["SCHEDULER_T_0"] = scheduler_config["T_0"]
    training_config["SCHEDULER_T_mult"] = scheduler_config["T_mult"]
    training_config["SCHEDULER_eta_min"] = scheduler_config["eta_min"]
    training_config["SCHEDULER_last_epoch"] = scheduler_config["last_epoch"]

    training_config["IMG_WIDTH"] = env_config["IMG_WIDTH"]
    training_config["IMG_HEIGHT"] = env_config["IMG_HEIGHT"]
    training_config["IMG_GRAYSCALE"] = env_config.get("IMG_GRAYSCALE", False)
    training_config["IMG_HIST_LEN"] = env_config["IMG_HIST_LEN"]

    return training_config
