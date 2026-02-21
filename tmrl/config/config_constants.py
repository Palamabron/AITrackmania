# standard library imports

import logging

import os
from pathlib import Path
import json
import platform
from packaging import version

from tmrl.custom.tm.utils.compute_reward import RewardFunction

__compatibility__ = "0.6.0"

# TMRL FOLDER: =======================================================

SYSTEM = platform.system()
RTGYM_VERSION = "real-time-gym-v1" if SYSTEM == "Windows" else "real-time-gym-ts-v1"

TMRL_FOLDER = Path.home() / "TmrlData"

if not TMRL_FOLDER.exists():
    raise RuntimeError(f"Missing folder: {TMRL_FOLDER}")

CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
DATASET_FOLDER = TMRL_FOLDER / "dataset"
REWARD_FOLDER = TMRL_FOLDER / "reward"
TRACK_FOLDER = TMRL_FOLDER / "track"
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
CONFIG_FOLDER = TMRL_FOLDER / "config"

CONFIG_FILE = TMRL_FOLDER / "config" / "config.json"
with open(CONFIG_FILE) as f:
    TMRL_CONFIG = json.load(f)

# VERSION CHECK: =====================================================

__err_msg = "Perform a clean installation:\n(1) Uninstall TMRL,\n(2) Delete the TmrlData folder,\n(3) Reinstall TMRL."
assert "__VERSION__" in TMRL_CONFIG, "config.json is outdated. " + __err_msg
CONFIG_VERSION = TMRL_CONFIG["__VERSION__"]
assert version.parse(CONFIG_VERSION) >= version.parse(__compatibility__), \
    f"config.json version ({CONFIG_VERSION}) must be >= {__compatibility__}. " + __err_msg

# GENERAL: ===========================================================

RUN_NAME = TMRL_CONFIG["RUN_NAME"]
BUFFERS_MAXLEN = TMRL_CONFIG["BUFFERS_MAXLEN"]
RW_MAX_SAMPLES_PER_EPISODE = TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

PRAGMA_RNN = False  # True to use an RNN, False to use an MLP

CUDA_TRAINING = TMRL_CONFIG["CUDA_TRAINING"]  # True if CUDA, False if CPU (trainer)
CUDA_INFERENCE = TMRL_CONFIG["CUDA_INFERENCE"]  # True if CUDA, False if CPU (rollout worker)

PRAGMA_GAMEPAD = TMRL_CONFIG["VIRTUAL_GAMEPAD"]  # True to use gamepad, False to use keyboard

LOCALHOST_WORKER = TMRL_CONFIG["LOCALHOST_WORKER"]  # set to True for RolloutWorkers on the same machine as the Server
LOCALHOST_TRAINER = TMRL_CONFIG["LOCALHOST_TRAINER"]  # set to True for Trainers on the same machine as the Server
PUBLIC_IP_SERVER = TMRL_CONFIG["PUBLIC_IP_SERVER"]

SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"

# ENVIRONMENT: =======================================================

ENV_CONFIG = TMRL_CONFIG["ENV"]
RTGYM_INTERFACE = str(ENV_CONFIG["RTGYM_INTERFACE"]).upper()
SEED = ENV_CONFIG["SEED"]
MAP_NAME = ENV_CONFIG["MAP_NAME"]
MIN_NB_ZERO_REW_BEFORE_FAILURE = ENV_CONFIG["MIN_NB_ZERO_REW_BEFORE_FAILURE"]
MAX_NB_ZERO_REW_BEFORE_FAILURE = ENV_CONFIG["MAX_NB_ZERO_REW_BEFORE_FAILURE"]
MIN_NB_STEPS_BEFORE_FAILURE = ENV_CONFIG["MIN_NB_STEPS_BEFORE_FAILURE"]
OSCILLATION_PERIOD = ENV_CONFIG["OSCILLATION_PERIOD"]
NB_OBS_FORWARD = ENV_CONFIG["NB_OBS_FORWARD"]
PRAGMA_LIDAR = RTGYM_INTERFACE.endswith("LIDAR")  # True if Lidar, False if images
# True if Lidar, False if images:
PRAGMA_CUSTOM = RTGYM_INTERFACE.endswith("MOBILEV3") or RTGYM_INTERFACE.endswith("CUSTOM")
PRAGMA_PROGRESS = RTGYM_INTERFACE.endswith("LIDARPROGRESS")
PRAGMA_TRACKMAP = RTGYM_INTERFACE.endswith("TRACKMAP")
PRAGMA_BEST = RTGYM_INTERFACE.endswith("BEST")
PRAGMA_BEST_TQC = RTGYM_INTERFACE.endswith("BEST_TQC")
PRAGMA_MBEST_TQC = RTGYM_INTERFACE.endswith("MTQC")
CRASH_PENALTY = ENV_CONFIG["CRASH_PENALTY"]
CRASH_COOLDOWN = ENV_CONFIG["CRASH_COOLDOWN"]
CONSTANT_PENALTY = ENV_CONFIG["CONSTANT_PENALTY"]  # -abs(x) : added to the reward at each time step
LAP_REWARD = ENV_CONFIG["LAP_REWARD"]
LAP_COOLDOWN = ENV_CONFIG["LAP_COOLDOWN"]
CHECKPOINT_REWARD = ENV_CONFIG["CHECKPOINT_REWARD"]

# CHECKPOINT_COOLDOWN = ENV_CONFIG["CHECKPOINT_COOLDOWN"]

END_OF_TRACK_REWARD = ENV_CONFIG["END_OF_TRACK_REWARD"]  # bonus reward at the end of the track
USE_IMAGES = ENV_CONFIG["USE_IMAGES"]

if PRAGMA_PROGRESS or PRAGMA_TRACKMAP:
    PRAGMA_LIDAR = True
LIDAR_BLACK_THRESHOLD = [55, 55, 55]  # [88, 88, 88] for tiny road, [55, 55, 55] FOR BASIC ROAD
REWARD_CONFIG = ENV_CONFIG["REWARD_CONFIG"]
SLEEP_TIME_AT_RESET = ENV_CONFIG["SLEEP_TIME_AT_RESET"]  # 1.5 to start in a Markov state with the lidar
IMG_HIST_LEN = ENV_CONFIG["IMG_HIST_LEN"]  # 4 without RNN, 1 with RNN
ACT_BUF_LEN = ENV_CONFIG["RTGYM_CONFIG"]["act_buf_len"]
WINDOW_WIDTH = ENV_CONFIG["WINDOW_WIDTH"]
WINDOW_HEIGHT = ENV_CONFIG["WINDOW_HEIGHT"]
GRAYSCALE = ENV_CONFIG["IMG_GRAYSCALE"] if "IMG_GRAYSCALE" in ENV_CONFIG else False
IMG_WIDTH = ENV_CONFIG["IMG_WIDTH"] if "IMG_WIDTH" in ENV_CONFIG else 64
IMG_HEIGHT = ENV_CONFIG["IMG_HEIGHT"] if "IMG_HEIGHT" in ENV_CONFIG else 64
LINUX_X_OFFSET = ENV_CONFIG["LINUX_X_OFFSET"] if "LINUX_X_OFFSET" in ENV_CONFIG else 64
LINUX_Y_OFFSET = ENV_CONFIG["LINUX_Y_OFFSET"] if "LINUX_Y_OFFSET" in ENV_CONFIG else 70
IMG_SCALE_CHECK_ENV = ENV_CONFIG["IMG_SCALE_CHECK_ENV"] if "IMG_SCALE_CHECK_ENV" in ENV_CONFIG else 1.0

# DEBUGGING AND BENCHMARKING: ===================================
# Only for checking the consistency of the custom networking methods, set it to False otherwise.
# Caution: difficult to handle if reset transitions are collected.
DEBUGGER = TMRL_CONFIG["DEBUGGER"]
DEBUG_MODE = DEBUGGER.get("DEBUG_MODE", False)
CRC_DEBUG = DEBUGGER["CRC_DEBUG"]
CRC_DEBUG_SAMPLES = DEBUGGER["CRC_DEBUG_SAMPLES"]  # Number of samples collected in CRC_DEBUG mode
PROFILE_TRAINER = DEBUGGER["PROFILE_TRAINER"]  # Will profile each epoch in the Trainer when True
SYNCHRONIZE_CUDA = PROFILE_TRAINER  # Set to True for profiling, False otherwise
WANDB_DEBUG = DEBUGGER["WANDB_DEBUG"]
PYTORCH_PROFILER = DEBUGGER["PYTORCH_PROFILER"]

# FILE SYSTEM: =================================================

PATH_DATA = TMRL_FOLDER
logging.debug(f" PATH_DATA:{PATH_DATA}")

# 0 for not saving history, x for saving model history every x epochs new model received by RolloutWorker
MODEL_CONFIG = TMRL_CONFIG["MODEL"]
MODEL_HISTORY = MODEL_CONFIG["SAVE_MODEL_EVERY"]

MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))
REWARDS_CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_rew_" + MAP_NAME + "_t.tcpt"))
DATASET_PATH = str(DATASET_FOLDER)
REWARD_PATH = str(REWARD_FOLDER / str("reward_" + MAP_NAME + ".pkl"))
TRACK_PATH_LEFT = str(TRACK_FOLDER / str("track_" + MAP_NAME + "_left" + ".pkl"))
TRACK_PATH_RIGHT = str(TRACK_FOLDER / str("track_" + MAP_NAME + "_right" + ".pkl"))

# WANDB: =======================================================

WANDB_RUN_ID = RUN_NAME
WANDB_PROJECT = TMRL_CONFIG["WANDB_PROJECT"]
WANDB_ENTITY = TMRL_CONFIG["WANDB_ENTITY"]
WANDB_KEY = TMRL_CONFIG["WANDB_KEY"]
WANDB_GRADIENTS = TMRL_CONFIG["WANDB_GRADIENTS"]
WANDB_DEBUG_REWARD = TMRL_CONFIG["WANDB_DEBUG_REWARD"]

os.environ['WANDB_API_KEY'] = WANDB_KEY

# NETWORKING: ==================================================

PRINT_BYTESIZES = True

PORT = TMRL_CONFIG["PORT"]  # Port to listen to (non-privileged ports are > 1023)
LOCAL_PORT_SERVER = TMRL_CONFIG["LOCAL_PORT_SERVER"]
LOCAL_PORT_TRAINER = TMRL_CONFIG["LOCAL_PORT_TRAINER"]
LOCAL_PORT_WORKER = TMRL_CONFIG["LOCAL_PORT_WORKER"]
PASSWORD = TMRL_CONFIG["PASSWORD"]
SECURITY = "TLS" if TMRL_CONFIG["TLS"] else None
CREDENTIALS_DIRECTORY = TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"] if TMRL_CONFIG[
                                                                        "TLS_CREDENTIALS_DIRECTORY"] != "" else None
HOSTNAME = TMRL_CONFIG["TLS_HOSTNAME"]
NB_WORKERS = None if TMRL_CONFIG["NB_WORKERS"] < 0 else TMRL_CONFIG["NB_WORKERS"]

# (200 000 000 is large enough for 1000 images right now)
BUFFER_SIZE = TMRL_CONFIG["BUFFER_SIZE"]  # (268_435_456) socket buffer size
HEADER_SIZE = TMRL_CONFIG["HEADER_SIZE"]  # fixed number of characters used to describe the data length

# MODEL CONFIG =========================
MODEL_CONFIG = TMRL_CONFIG["MODEL"]
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

# ALG CONFIG ============================
ALG_CONFIG = TMRL_CONFIG["ALG"]
if ALG_CONFIG["ALGORITHM"] != "TQC" and ALG_CONFIG["QUANTILES_NUMBER"] > 1:
    raise ValueError("QUANTILES_NUMBER must be 1 if it is used with SAC")
QUANTILES_NUMBER = ALG_CONFIG["QUANTILES_NUMBER"]
N_STEPS = 1 if ALG_CONFIG["N_STEPS"] <= 0 else ALG_CONFIG["N_STEPS"]
WEIGHT_CLIPPING_ENABLED = ALG_CONFIG["CLIPPING_WEIGHTS"]
WEIGHT_CLIPPING_VALUE = 1.0 if not WEIGHT_CLIPPING_ENABLED else ALG_CONFIG["CLIP_WEIGHTS_VALUE"]
ACTOR_WEIGHT_DECAY = ALG_CONFIG["ACTOR_WEIGHT_DECAY"]
CRITIC_WEIGHT_DECAY = ALG_CONFIG["CRITIC_WEIGHT_DECAY"]
POINTS_NUMBER = ALG_CONFIG["NUMBER_OF_POINTS"]
POINTS_DISTANCE = ALG_CONFIG["POINTS_DISTANCE"]
SPEED_BONUS = ALG_CONFIG["SPEED_BONUS"]
SPEED_MIN_THRESHOLD = ALG_CONFIG["SPEED_MIN_THRESHOLD"]
SPEED_MEDIUM_THRESHOLD = ALG_CONFIG["SPEED_MEDIUM_THRESHOLD"]
ADAM_EPS = ALG_CONFIG["ADAM_EPS"]


# CREATE CONFIG ===================================
def create_config():
    config = dict()
    alg_config = TMRL_CONFIG["ALG"]
    model_config = TMRL_CONFIG["MODEL"]
    scheduler_config = model_config["SCHEDULER"]
    env_config = TMRL_CONFIG["ENV"]

    config["TRAINING_STEPS_PER_ROUND"] = model_config["TRAINING_STEPS_PER_ROUND"]
    config["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"] = model_config["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
    config["ENVIRONMENT_STEPS_BEFORE_TRAINING"] = model_config["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
    config["UPDATE_MODEL_INTERVAL"] = model_config["UPDATE_MODEL_INTERVAL"]
    config["UPDATE_BUFFER_INTERVAL"] = model_config["UPDATE_BUFFER_INTERVAL"]
    config["SAVE_MODEL_EVERY"] = model_config["SAVE_MODEL_EVERY"]
    config["MEMORY_SIZE"] = model_config["MEMORY_SIZE"]
    config["BATCH_SIZE"] = model_config["BATCH_SIZE"]

    config["CNN_FILTERS"] = model_config["CNN_FILTERS"]
    for index, size in enumerate(config["CNN_FILTERS"]):
        config[f"CNN_FILTER{index}"] = size

    config["CNN_OUTPUT_SIZE"] = model_config["CNN_OUTPUT_SIZE"]

    config["RNN_SIZES"] = model_config["RNN_SIZES"]

    for index, size in enumerate(config["RNN_SIZES"]):
        config[f"RNN_SIZE{index}"] = size

    config["RNN_LENS"] = model_config["RNN_LENS"]

    for index, size in enumerate(config["RNN_LENS"]):
        config[f"RNN_LEN{index}"] = size

    config["API_MLP_SIZES"] = model_config["API_MLP_SIZES"]

    for index, size in enumerate(config["API_MLP_SIZES"]):
        config[f"API_MLP_SIZE{index}"] = size

    config["API_LAYERNORM"] = model_config["API_LAYERNORM"]
    config["NOISY_LINEAR_ACTOR"] = model_config["NOISY_LINEAR_ACTOR"]
    config["NOISY_LINEAR_CRITIC"] = model_config["NOISY_LINEAR_CRITIC"]
    config["RNN_DROPOUT"] = model_config["RNN_DROPOUT"]
    config["CNN_FILTERS"] = model_config["CNN_FILTERS"]

    config["MIN_NB_ZERO_REW_BEFORE_FAILURE"] = env_config["MIN_NB_ZERO_REW_BEFORE_FAILURE"]
    config["MAX_NB_ZERO_REW_BEFORE_FAILURE"] = env_config["MAX_NB_ZERO_REW_BEFORE_FAILURE"]
    config["MIN_NB_STEPS_BEFORE_FAILURE"] = env_config["MIN_NB_STEPS_BEFORE_FAILURE"]

    config["OSCILLATION_PERIOD"] = env_config["OSCILLATION_PERIOD"]
    config["CRASH_PENALTY"] = env_config["CRASH_PENALTY"]
    config["CRASH_COOLDOWN"] = env_config["CRASH_COOLDOWN"]
    config["CONSTANT_PENALTY"] = env_config["CONSTANT_PENALTY"]
    config["LAP_REWARD"] = env_config["LAP_REWARD"]
    config["LAP_COOLDOWN"] = env_config["LAP_COOLDOWN"]
    config["CHECKPOINT_REWARD"] = env_config["CHECKPOINT_REWARD"]
    config["CHECKPOINT_COOLDOWN"] = env_config["CHECKPOINT_COOLDOWN"]

    config["REWARD_END_OF_TRACK"] = env_config["END_OF_TRACK_REWARD"]
    config["ALGORITHM"] = alg_config["ALGORITHM"]
    config["QUANTILES_NUMBER"] = alg_config["QUANTILES_NUMBER"]
    config["LEARN_ENTROPY_COEF"] = alg_config["LEARN_ENTROPY_COEF"]
    config["LR_ACTOR"] = alg_config["LR_ACTOR"]
    config["LR_CRITIC"] = alg_config["LR_CRITIC"]
    config["LR_CRITIC_DIVIDED_BY_LR_ACTOR"] = config["LR_CRITIC"] / config["LR_ACTOR"]
    config["N_STEPS"] = alg_config["N_STEPS"]
    config["ACTOR_WEIGHT_DECAY"] = alg_config["ACTOR_WEIGHT_DECAY"]
    config["CRITIC_WEIGHT_DECAY"] = alg_config["CRITIC_WEIGHT_DECAY"]
    config["CLIPPING_WEIGHTS"] = alg_config["CLIPPING_WEIGHTS"]
    config["CLIP_WEIGHTS_VALUE"] = 1.0 if not config["CLIPPING_WEIGHTS"] else alg_config["CLIP_WEIGHTS_VALUE"]
    config["POINTS_NUMBER"] = alg_config["NUMBER_OF_POINTS"]
    config["POINTS_DISTANCE"] = alg_config["POINTS_DISTANCE"]
    config["SPEED_BONUS"] = alg_config["SPEED_BONUS"]
    config["SPEED_MIN_THRESHOLD"] = alg_config["SPEED_MIN_THRESHOLD"]
    config["SPEED_MEDIUM_THRESHOLD"] = alg_config["SPEED_MEDIUM_THRESHOLD"]

    config["LR_ENTROPY"] = alg_config["LR_ENTROPY"]
    config["GAMMA"] = alg_config["GAMMA"]
    config["POLYAK"] = alg_config["POLYAK"]
    config["TARGET_ENTROPY"] = alg_config["TARGET_ENTROPY"]
    config["TOP_QUANTILES_TO_DROP"] = alg_config["TOP_QUANTILES_TO_DROP"]

    if alg_config["QUANTILES_NUMBER"] != 1 and alg_config["ALGORITHM"] == "SAC":
        ValueError("SAC can be only used if the QUANTILES_NUMBER equals to 1")

    config["QUANTILES_NUMBER"] = alg_config["QUANTILES_NUMBER"]
    config["R2D2_REWIND"] = alg_config["R2D2_REWIND"]

    config["POINTS_NUMBER"] = alg_config["NUMBER_OF_POINTS"]
    config["ADAM_EPS"] = alg_config["ADAM_EPS"]

    config["SCHEDULER_T_0"] = scheduler_config["T_0"]
    config["SCHEDULER_T_mult"] = scheduler_config["T_mult"]
    config["SCHEDULER_eta_min"] = scheduler_config["eta_min"]
    config["SCHEDULER_last_epoch"] = scheduler_config["last_epoch"]

    config["IMG_WIDTH"] = env_config["IMG_WIDTH"]
    config["IMG_HEIGHT"] = env_config["IMG_HEIGHT"]
    config["IMG_GRAYSCALE"] = env_config["IMG_GRAYSCALE"]
    config["IMG_HIST_LEN"] = env_config["IMG_HIST_LEN"]

    return config