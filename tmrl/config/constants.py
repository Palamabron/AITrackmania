"""Derived flags and scalars from validated MainConfig (single source of truth)."""

from __future__ import annotations

import os
import pickle

from loguru import logger

from tmrl.config.loader import MAIN_CONFIG
from tmrl.config.paths import (
    MAP_NAME,
    PLAYER_RUNS_FOLDER,
    REWARD_PATH,
    TRACK_PATH_LEFT,
    TRACK_PATH_RIGHT,
)
from tmrl.config.rtgym_boundary_iface import (
    rtgym_discrete_boundary_lidar_images,
    rtgym_discrete_boundary_lidar_vec,
)
from tmrl.config.spacing_lookahead import (
    points_number_from_spacing_config,
    polyline_arc_length_m,
)

M = MAIN_CONFIG

# --- Run / rollout ---
RUN_NAME = M.run.name
BUFFERS_MAXLEN = M.run.buffers_maxlen
RW_MAX_SAMPLES_PER_EPISODE = M.run.rw_max_samples_per_episode
RW_TEST_EPISODE_INTERVAL = M.run.rw_test_episode_interval
RW_TEST_EPISODES_PER_EVAL = M.run.rw_test_episodes_per_eval

# --- Devices ---
CUDA_TRAINING = M.compute.cuda_training
CUDA_INFERENCE = M.compute.cuda_inference
USE_VIRTUAL_GAMEPAD = M.compute.virtual_gamepad
# Runtime recurrent path is still gated by this global switch.
USE_RNN = False

# --- Distributed ---
LOCALHOST_WORKER = M.distributed.localhost_worker
LOCALHOST_TRAINER = M.distributed.localhost_trainer
PUBLIC_IP_SERVER = M.distributed.public_ip_server
SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"
PORT = M.distributed.server_port
LOCAL_PORT_SERVER = M.distributed.local_port_server
LOCAL_PORT_TRAINER = M.distributed.local_port_trainer
LOCAL_PORT_WORKER = M.distributed.local_port_worker
PASSWORD = M.distributed.password
SECURITY = "TLS" if M.distributed.use_tls else None
CREDENTIALS_DIRECTORY = (
    M.distributed.tls_credentials_directory
    if M.distributed.tls_credentials_directory != ""
    else None
)
HOSTNAME = M.distributed.tls_hostname
NB_WORKERS = None if M.distributed.num_workers < 0 else M.distributed.num_workers
BUFFER_SIZE = M.distributed.buffer_size
HEADER_SIZE = M.distributed.header_size
PRINT_BYTESIZES = True

# --- W&B ---
WANDB_RUN_ID = RUN_NAME
WANDB_PROJECT = M.wandb.project
WANDB_ENTITY = M.wandb.entity
WANDB_KEY = M.wandb.api_key
WANDB_GRADIENTS = M.wandb.log_gradients
WANDB_DEBUG_REWARD = M.wandb.debug_reward
WANDB_WORKER = M.wandb.log_from_worker


def ensure_wandb_api_key() -> None:
    if "WANDB_API_KEY" not in os.environ and WANDB_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_KEY


# --- Training loop (replay, schedule, checkpoint policy) ---
T = M.training
MODEL_HISTORY = T.save_model_every
BEST_CHECKPOINT_CRITERION = T.best_checkpoint_criterion
BEST_CHECKPOINT_LAP_TIME = T.best_checkpoint_lap_time
BEST_CHECKPOINT_MIN_FINISHES = T.best_checkpoint_min_finishes
COMPETITION_EVAL_CRASH_PENALTY_S = float(T.competition_eval_crash_penalty_s)
COMPETITION_EVAL_MAX_CRASHES = int(T.competition_eval_max_crashes)
MAX_EPOCHS = T.max_epochs
ROUNDS_PER_EPOCH = T.rounds_per_epoch
TRAINING_STEPS_PER_ROUND = T.training_steps_per_round
MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP = T.max_training_steps_per_environment_step
ENVIRONMENT_STEPS_BEFORE_TRAINING = T.environment_steps_before_training
UPDATE_MODEL_INTERVAL = T.update_model_interval
UPDATE_BUFFER_INTERVAL = T.update_buffer_interval
MEMORY_SIZE = T.memory_size
BATCH_SIZE = T.batch_size
BATCHES_PER_STEP = T.batches_per_step
SCHEDULER_CONFIG = T.scheduler.model_dump()
RESET_TRAINING = M.run.reset_training

# --- Model ---
R = M.model
CNN_FILTERS = R.cnn_filters
CNN_OUTPUT_SIZE = R.cnn_output_size
RNN_LENS = R.rnn_lens
RNN_SIZES = R.rnn_sizes
API_MLP_SIZES = R.api_mlp_sizes
API_LAYERNORM = R.api_layernorm
MLP_LAYERNORM = R.mlp_layernorm
NOISY_LINEAR_CRITIC = R.noisy_linear_critic
NOISY_LINEAR_ACTOR = R.noisy_linear_actor
OUTPUT_DROPOUT = R.output_dropout
RNN_DROPOUT = R.rnn_dropout
USE_RESIDUAL_MLP = R.use_residual_mlp
RESIDUAL_MLP_HIDDEN_DIM = R.residual_mlp_hidden_dim
RESIDUAL_MLP_NUM_BLOCKS = R.residual_mlp_num_blocks
_RA = R.residual_mlp_num_blocks_actor
_RC = R.residual_mlp_num_blocks_critic
RESIDUAL_MLP_NUM_BLOCKS_ACTOR = _RA if _RA > 0 else RESIDUAL_MLP_NUM_BLOCKS
RESIDUAL_MLP_NUM_BLOCKS_CRITIC = _RC if _RC > 0 else RESIDUAL_MLP_NUM_BLOCKS
USE_SOPHY_RESIDUAL_ACTOR = R.use_sophy_residual_actor
SPLIT_TRACK_OBSERVATION = R.split_track_observation
USE_SIMBAV2 = R.use_simbav2
TRACK_ENCODER = R.track_encoder
GNN_LAYERS = R.gnn_layers
GNN_HIDDEN = R.gnn_hidden
USE_RNN_MODEL = R.use_rnn
_RHS = R.rnn_hidden_size
RNN_HIDDEN_SIZE = _RHS if _RHS > 0 else RESIDUAL_MLP_HIDDEN_DIM
USE_EFFICIENTNET = R.use_efficientnet
USE_FROZEN_EFFNET = R.use_frozen_effnet
FROZEN_EFFNET_EMBED_DIM = R.frozen_effnet_embed_dim
FROZEN_EFFNET_WIDTH_MULT = R.frozen_effnet_width_mult
FROZEN_EFFNET_VARIANT = R.frozen_effnet_variant
FROZEN_EFFNET_USE_DW_STEM = R.frozen_effnet_use_dw_stem
BINARY_BRAKE = R.binary_brake

# --- Environment / interface ---
E = M.environment
RTGYM_INTERFACE = str(E.rtgym_interface).upper()
USE_LIDAR = rtgym_discrete_boundary_lidar_vec(RTGYM_INTERFACE)
USE_LIDAR_IMAGES = rtgym_discrete_boundary_lidar_images(RTGYM_INTERFACE)

# Non-boundary-lidar stacks: historical suffix tokens; flags describe what TMRL does with them.

# Screen CNN + MobileNet-style preprocessing
# (historical MOBILEV3 / CUSTOM / BEST / BEST_TQC suffixes).
USE_IMAGES_MOBILENET_PIPELINE = (
    RTGYM_INTERFACE.endswith("MOBILEV3")
    or RTGYM_INTERFACE.endswith("CUSTOM")
    or RTGYM_INTERFACE.endswith("BEST")
    or RTGYM_INTERFACE.endswith("BEST_TQC")
)

# World / vehicle telemetry fields in the observation
# (historical TQCGRAB* tokens in the interface id).
USE_OBS_WORLD_TELEMETRY_LAYOUT = "TQCGRAB" in RTGYM_INTERFACE
USE_IMAGES_WITH_WORLD_TELEMETRY_STACK = "TQCGRAB_IMAGES" in RTGYM_INTERFACE

# R2D2 replay layout: MTQC suffix or any world-telemetry interface id above.
USE_IMAGES_R2D2_SEQUENCE_BUFFER = RTGYM_INTERFACE.endswith("MTQC") or USE_OBS_WORLD_TELEMETRY_LAYOUT

SEED = E.seed
MIN_NB_ZERO_REW_BEFORE_FAILURE = E.min_zero_reward_steps_before_failure
MAX_NB_ZERO_REW_BEFORE_FAILURE = E.max_zero_reward_steps_before_failure
OSCILLATION_PERIOD = E.oscillation_period
NB_OBS_FORWARD = E.forward_obs_count
CRASH_COOLDOWN = E.crash_cooldown
CONSTANT_PENALTY = E.constant_penalty
LAP_REWARD = E.lap_reward
LAP_COOLDOWN = E.lap_cooldown
CHECKPOINT_REWARD = E.checkpoint_reward
CHECKPOINT_COOLDOWN = 0
END_OF_TRACK_REWARD = E.end_of_track_reward
USE_IMAGES = E.use_images
SLEEP_TIME_AT_RESET = E.sleep_time_at_reset
IMG_HIST_LEN = E.img_hist_len
ACT_BUF_LEN = E.rtgym.act_buf_len
RTGYM_TIME_STEP_DURATION = float(E.rtgym.time_step_duration)
WINDOW_WIDTH = E.window_width
WINDOW_HEIGHT = E.window_height
GRAYSCALE = E.img_grayscale
IMG_WIDTH = E.img_width
IMG_HEIGHT = E.img_height
LINUX_X_OFFSET = E.linux_x_offset
LINUX_Y_OFFSET = E.linux_y_offset
IMG_SCALE_CHECK_ENV = E.img_scale_check_env
INIT_GAS_BIAS = E.init_gas_bias
OBS_SPEED_SCALE = float(E.obs_speed_scale)
OBS_TRACK_SCALE = float(E.obs_track_scale)
REWARD_CONFIG = E.reward.model_dump()
# Stagnant-progress cutoff: seconds-only (merged reward + environment;
# RewardFunction reads REWARD_CONFIG).
_rsec = float(REWARD_CONFIG.get("min_seconds_before_failure", 0.0))
_esec = float(E.min_seconds_before_failure or 0.0)
_merged_sec = max(_rsec, _esec)
REWARD_CONFIG["min_seconds_before_failure"] = _merged_sec
if _esec > _rsec:
    logger.info(
        "Effective reward.min_seconds_before_failure={:.3f}s "
        "(max of reward={:.3f}s and environment={:.3f}s)",
        _merged_sec,
        _rsec,
        _esec,
    )

_ot_r = float(REWARD_CONFIG.get("off_track_seconds_before_failure", 0.5))
_ot_e = float(E.off_track_seconds_before_failure or 0.0)
_merged_ot = max(_ot_r, _ot_e)
REWARD_CONFIG["off_track_seconds_before_failure"] = _merged_ot
if _ot_e > _ot_r:
    logger.info(
        "Effective reward.off_track_seconds_before_failure={:.3f}s "
        "(max of reward={:.3f}s and environment={:.3f}s)",
        _merged_ot,
        _ot_r,
        _ot_e,
    )

# Crash penalty: single effective value for RewardFunction + all TM20 interfaces.
# Primary: environment.reward.crash_penalty. Legacy override: environment.crash_penalty
# when non-zero (same pattern as min_seconds_before_failure merge intent).
_r_cp = float(REWARD_CONFIG.get("crash_penalty", 2.0))
_e_cp = float(E.crash_penalty)
_merged_cp = _e_cp if _e_cp != 0.0 else _r_cp
REWARD_CONFIG["crash_penalty"] = _merged_cp
CRASH_PENALTY = _merged_cp
if _e_cp != 0.0 and _e_cp != _r_cp:
    logger.info(
        "Effective crash_penalty={:.4g} (environment.crash_penalty={:.4g} overrides "
        "environment.reward.crash_penalty={:.4g})",
        _merged_cp,
        _e_cp,
        _r_cp,
    )

MAX_SPEED_KMH = float(REWARD_CONFIG.get("max_speed_kmh", 300.0))

# Required assets: reward trajectory always; track pickles optional unless LIDAR+IMAGES fusion.
if not os.path.isfile(REWARD_PATH):
    raise FileNotFoundError(
        f"Reward trajectory missing for map_name={MAP_NAME!r}: expected {REWARD_PATH}. "
        "Set environment.map_name to match recorded data or record rewards into TmrlData/reward/."
    )
try:
    with open(REWARD_PATH, "rb") as _rf:
        _traj_pts = pickle.load(_rf)
    _n_reward_pts = len(_traj_pts)
except (OSError, pickle.UnpicklingError, TypeError, ValueError) as _e:
    raise RuntimeError(f"Could not load reward pickle at {REWARD_PATH}: {_e}") from _e
logger.info(
    "TmrlData map_name={!r} → reward: {} ({} centerline sample(s))",
    MAP_NAME,
    REWARD_PATH,
    _n_reward_pts,
)

if USE_LIDAR_IMAGES:
    _track_missing: list[str] = []
    if not os.path.isfile(TRACK_PATH_LEFT):
        _track_missing.append(f"left ({TRACK_PATH_LEFT})")
    if not os.path.isfile(TRACK_PATH_RIGHT):
        _track_missing.append(f"right ({TRACK_PATH_RIGHT})")
    if _track_missing:
        raise FileNotFoundError(
            f"Boundary lidar + images pipeline (rtgym_interface={RTGYM_INTERFACE!r}) "
            f"requires track boundary pickles for map_name={MAP_NAME!r}. "
            f"Missing: {', '.join(_track_missing)}."
        )
    for _side, _path in (("left", TRACK_PATH_LEFT), ("right", TRACK_PATH_RIGHT)):
        try:
            with open(_path, "rb") as _tf:
                _bound = pickle.load(_tf)
            logger.info("  track_{}_boundary: {} point(s)", _side, len(_bound))
        except (OSError, pickle.UnpicklingError, TypeError, ValueError) as _e:
            raise RuntimeError(
                f"Could not load track {_side} boundary pickle at {_path}: {_e}"
            ) from _e
else:
    for _side, _path in (("left", TRACK_PATH_LEFT), ("right", TRACK_PATH_RIGHT)):
        if os.path.isfile(_path):
            try:
                with open(_path, "rb") as _tf:
                    _nb = len(pickle.load(_tf))
                logger.info("  track_{}_boundary (optional): {} point(s) at {}", _side, _nb, _path)
            except (OSError, pickle.UnpicklingError, TypeError, ValueError) as _e:
                logger.warning(
                    "  track_{}_boundary present but unreadable at {}: {}",
                    _side,
                    _path,
                    _e,
                )

_rw = E.reward
TRACK_POINTS_NUMBER = None
_tpct = float(_rw.track_look_ahead_pct)
_tsp = float(_rw.track_point_spacing_m)
if _tpct > 0 and _tsp > 0:
    try:
        with open(REWARD_PATH, "rb") as _f:
            _traj = pickle.load(_f)
        _L = polyline_arc_length_m(_traj)
        if _L is not None:
            TRACK_POINTS_NUMBER = points_number_from_spacing_config(_L, _tpct, _tsp)
            if TRACK_POINTS_NUMBER is not None:
                logger.info(
                    "Track look-ahead: track_look_ahead_pct={:.2f}%, "
                    "track_point_spacing_m={:.2f} m, "
                    "trajectory length={:.1f} m -> POINTS_NUMBER={}",
                    _tpct,
                    _tsp,
                    _L,
                    TRACK_POINTS_NUMBER,
                )
    except (pickle.UnpicklingError, ValueError, OSError) as e:
        logger.warning(f"Failed to load reward trajectory for POINTS_NUMBER calculation: {e}")

# --- Algorithm ---
A = M.algorithm
LR_ACTOR = A.lr_actor
LR_CRITIC = A.lr_critic
LR_ENTROPY = A.lr_entropy
ALPHA = A.alpha
LEARN_ENTROPY_COEF = A.learn_entropy_coef
QUANTILES_NUMBER = A.quantiles_number
GAMMA = A.gamma
POLYAK = A.polyak
TARGET_ENTROPY = A.target_entropy
TOP_QUANTILES_TO_DROP = A.top_quantiles_to_drop
N_STEPS = 1 if A.n_steps <= 0 else A.n_steps
R2D2_REWIND = A.r2d2_rewind
R2D2_NUM_SEQUENCES = A.r2d2_num_sequences
R2D2_SEQUENCE_LENGTH = A.r2d2_sequence_length
R2D2_BURN_IN = A.r2d2_burn_in
PER_TD_BETA = float(A.per_td_beta)
FOG_DECAY_TEMPERATURE = float(A.fog_decay_temperature)
IQN_N_STEER_BINS = int(A.iqn_n_steer_bins)
MIXED_PRECISION = bool(A.mixed_precision)
MIXED_PRECISION_DTYPE = str(A.mixed_precision_dtype)
WEIGHT_CLIPPING_ENABLED = A.clipping_weights
WEIGHT_CLIPPING_VALUE = 1.0 if not WEIGHT_CLIPPING_ENABLED else A.clip_weights_value
_OPTIMIZER_WEIGHT_DECAY = float(A.weight_decay)
ACTOR_WEIGHT_DECAY = _OPTIMIZER_WEIGHT_DECAY
CRITIC_WEIGHT_DECAY = _OPTIMIZER_WEIGHT_DECAY
POINTS_NUMBER = TRACK_POINTS_NUMBER if TRACK_POINTS_NUMBER is not None else A.num_track_points
POINTS_DISTANCE = A.points_distance
SPEED_BONUS = A.speed_bonus
SPEED_MIN_THRESHOLD = A.speed_min_threshold
SPEED_MEDIUM_THRESHOLD = A.speed_medium_threshold
ADAM_EPS = A.adam_eps
GRAD_CLIP_ACTOR = float(A.grad_clip_actor)
GRAD_CLIP_CRITIC = float(A.grad_clip_critic)
BACKUP_CLIP_RANGE = float(A.backup_clip_range)
REWARD_NORMALIZE_SCALE = float(A.reward_normalize_scale)
USE_SDE = bool(A.use_sde)
LOG_STD_INIT = float(A.log_std_init)
SDE_CLIP_MEAN = float(A.sde_clip_mean)
SDE_SAMPLE_FREQ = int(A.sde_sample_freq)
ENTROPY_FLOOR = float(A.entropy_floor)
ENTROPY_SCHEDULE = str(A.entropy_schedule)
ENTROPY_COSINE_T0 = int(A.entropy_cosine_t0)
ENTROPY_COSINE_TMULT = float(A.entropy_cosine_tmult)
ENTROPY_COSINE_DECAY = float(A.entropy_cosine_decay)
PER_TD_ENABLED = bool(A.per_td_enabled)
PER_TD_ALPHA = float(A.per_td_alpha)
PER_TD_EPS = float(A.per_td_eps)

# --- Debugger ---
D = M.debugger
DEBUG_MODE = D.debug_mode
CRC_DEBUG = D.crc_debug
CRC_DEBUG_SAMPLES = D.crc_debug_samples
PROFILE_TRAINER = D.profile_trainer
SYNCHRONIZE_CUDA = PROFILE_TRAINER
WANDB_DEBUG = D.wandb_debug
PYTORCH_PROFILER = D.pytorch_profiler
OBSERVATION_BOUNDS_CHECK = D.observation_bounds_check

# --- Player runs ---
PR = M.player_runs
PLAYER_RUNS_ONLINE_INJECTION = PR.online_injection
PLAYER_RUNS_SOURCE_PATH = PR.source_path if PR.source_path else str(PLAYER_RUNS_FOLDER)
PLAYER_RUNS_CONSUME_ON_READ = PR.consume_on_read
PLAYER_RUNS_MAX_FILES_PER_UPDATE = PR.max_files_per_update
PLAYER_RUNS_DEMO_INJECTION_REPEAT = max(1, PR.demo_injection_repeat)
PLAYER_RUNS_PER_ALPHA = max(0.0, float(PR.per_alpha))
DEMO_MAX_BATCH_FRACTION = max(0.0, min(1.0, float(PR.demo_max_batch_fraction)))
DEMO_MIN_BATCH_FRACTION = max(0.0, min(1.0, float(PR.demo_min_batch_fraction)))
DEMO_SAMPLING_WEIGHT = max(0.0, float(PR.demo_sampling_weight))
DEMO_WEIGHT_DECAY_SAMPLES = max(0, int(PR.demo_weight_decay_samples))
DEMO_WEIGHT_DECAY_SLOWDOWN = max(0.0, float(PR.demo_weight_decay_slowdown))
