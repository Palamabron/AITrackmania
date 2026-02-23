"""Build runtime objects from config: interface, memory, agent, trainer.

This module reads config_constants (which loads config.json) and selects:
  - TRAIN_MODEL / POLICY   : neural net classes (MLP, CNN, RNN, IMPALA, etc.)
  - INT                    : rtgym interface class (TM2020Interface*, partial with kwargs)
  - CONFIG_DICT            : rtgym config dict (interface + RTGYM_CONFIG overrides)
  - SAMPLE_COMPRESSOR      : how to compress samples for network transfer
  - OBS_PREPROCESSOR       : observation preprocessing for the env
  - MEM / MEMORY           : replay memory class (partial with size, batch_size, etc.)
  - AGENT                  : training agent class (SAC/TQC/REDQ, partial with hyperparams)
  - TRAINER                : TorchTrainingOffline partial (epochs, rounds, steps, etc.)
  - DUMP/LOAD/UPDATER      : checkpoint helpers

Selection logic:
  - Observation type: PRAGMA_LIDAR (Lidar) vs image-based (Full, IMPALA, Sophy, TrackMap).
  - Interface: chosen from RTGYM_INTERFACE (Lidar, LidarProgress, TrackMap, IMPALA, Sophy, Full).
  - Memory: Lidar → MemoryTMLidar* ; IMPALA/Best → MemoryTMBest ;
  MTQC+images → MemoryR2D2 ; else MemoryTMFull.
  - Model: Lidar+RNN → RNNActorCritic ; Lidar → MLP or REDQ MLP ; MTQC → IMPALA/Sophy ;
  else Vanilla CNN.
"""

from __future__ import annotations

from typing import Any

import rtgym

import tmrl.config.config_constants as cfg
import tmrl.custom.models.IMPALA as impala  # noqa: N811
import tmrl.custom.models.Sophy as impalaWoImages
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_algorithms import TQCAgent as TQC_Agent
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.custom.custom_memories import (
    MemoryR2D2,
    MemoryR2D2woImages,
    MemoryTMBest,
    MemoryTMFull,
    MemoryTMLidar,
    MemoryTMLidarProgress,
    MemoryTMLidarProgressImages,
    get_local_buffer_sample_lidar,
    get_local_buffer_sample_lidar_progress,
    get_local_buffer_sample_lidar_progress_images,
    get_local_buffer_sample_mobilenet,
    get_local_buffer_sample_tm20_imgs,
)
from tmrl.custom.custom_models import (
    FrozenEffNetResidualActorCritic,
    MLPActorCritic,
    REDQMLPActorCritic,
    REDQResidualMLPActorCritic,
    ResidualMLPActorCritic,
    RNNActorCritic,
    SquashedGaussianFrozenEffNetResidualActor,
    SquashedGaussianMLPActor,
    SquashedGaussianResidualMLPActor,
    SquashedGaussianRNNActor,
    SquashedGaussianVanillaCNNActor,
    SquashedGaussianVanillaColorCNNActor,
    VanillaCNNActorCritic,
    VanillaColorCNNActorCritic,
)
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.interfaces.TM2020InterfaceIMPALA import TM2020InterfaceIMPALA
from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from tmrl.custom.interfaces.TM2020InterfaceLidarImages import TM2020InterfaceLidarProgressImages
from tmrl.custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress
from tmrl.custom.interfaces.TM2020InterfaceSophy import TM2020InterfaceIMPALASophy
from tmrl.custom.interfaces.TM2020InterfaceTQC import TM2020InterfaceTQC
from tmrl.custom.interfaces.TM2020InterfaceTrackMap import TM2020InterfaceTrackMap
from tmrl.custom.interfaces.TM2020InterfaceTrackMapImages import TM2020InterfaceTrackMapImages
from tmrl.custom.models.Sophy import SophyResidualActorCritic, SquashedActorSophyResidual
from tmrl.custom.tm.tm_preprocessors import (
    obs_preprocessor_lidar_progress_images_act_in_obs,
    obs_preprocessor_mobilenet_act_in_obs,
    obs_preprocessor_tm_act_in_obs,
    obs_preprocessor_tm_lidar_act_in_obs,
    obs_preprocessor_tm_lidar_progress_act_in_obs,
    obs_preprocessor_tqcgrab_act_in_obs,
)
from tmrl.envs import GenericGymEnv
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial

# -----------------------------------------------------------------------------
# Algorithm and model config references (from config.json)
# -----------------------------------------------------------------------------

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]
MODEL_CONFIG = cfg.TMRL_CONFIG["MODEL"]

assert ALG_NAME in ["SAC", "REDQSAC", "TQC"], (
    f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."
)

# -----------------------------------------------------------------------------
# 1. Model and policy classes (which neural net: MLP, CNN, RNN, IMPALA, Sophy)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if (cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES) and ALG_NAME == "SAC":
        _lidar_images_kw = dict(
            image_index=3,
            embed_dim=cfg.FROZEN_EFFNET_EMBED_DIM,
            hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
            num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
            width_mult=cfg.FROZEN_EFFNET_WIDTH_MULT,
        )
        TRAIN_MODEL: Any = partial(FrozenEffNetResidualActorCritic, **_lidar_images_kw)
        POLICY: Any = partial(SquashedGaussianFrozenEffNetResidualActor, **_lidar_images_kw)
    elif cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    elif cfg.USE_RESIDUAL_MLP:
        _residual_kw = dict(
            hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
            num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
        )
        TRAIN_MODEL = (
            partial(ResidualMLPActorCritic, **_residual_kw)
            if ALG_NAME == "SAC"
            else partial(REDQResidualMLPActorCritic, n=ALG_CONFIG.get("REDQ_N", 10), **_residual_kw)
        )
        POLICY = partial(SquashedGaussianResidualMLPActor, **_residual_kw)
    else:
        TRAIN_MODEL = MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic
        POLICY = SquashedGaussianMLPActor
else:
    if cfg.PRAGMA_MBEST_TQC or cfg.PRAGMA_TQC_GRAB:
        assert ALG_NAME in ("TQC", "SAC"), f"{ALG_NAME} is not implemented here."
        if (
            cfg.USE_IMAGES
            and not cfg.PRAGMA_TQC_GRAB
            and cfg.USE_FROZEN_EFFNET
            and ALG_NAME == "SAC"
        ):
            _frozen_effnet_kw = dict(
                embed_dim=cfg.FROZEN_EFFNET_EMBED_DIM,
                hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
                num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
                width_mult=cfg.FROZEN_EFFNET_WIDTH_MULT,
            )
            TRAIN_MODEL = partial(FrozenEffNetResidualActorCritic, **_frozen_effnet_kw)
            POLICY = partial(SquashedGaussianFrozenEffNetResidualActor, **_frozen_effnet_kw)
        elif cfg.USE_IMAGES and not cfg.PRAGMA_TQC_GRAB:
            TRAIN_MODEL = impala.QRCNNActorCritic
            POLICY = impala.SquashedActorQRCNN
        elif cfg.PRAGMA_TQC_GRAB and not cfg.USE_IMAGES and cfg.USE_RESIDUAL_SOPHY:
            _res_sophy_kw = dict(
                hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
                num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
            )
            TRAIN_MODEL = partial(SophyResidualActorCritic, **_res_sophy_kw)
            POLICY = partial(SquashedActorSophyResidual, **_res_sophy_kw)
        else:
            TRAIN_MODEL = impalaWoImages.SophyActorCritic
            POLICY = impalaWoImages.SquashedActorSophy
    else:
        assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = VanillaCNNActorCritic if cfg.GRAYSCALE else VanillaColorCNNActorCritic
        POLICY = (
            SquashedGaussianVanillaCNNActor
            if cfg.GRAYSCALE
            else SquashedGaussianVanillaColorCNNActor
        )

# -----------------------------------------------------------------------------
# 2. RtGym interface (TM2020* class + kwargs from env config)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_TRACKMAP_IMAGES:
        INT = partial(
            TM2020InterfaceTrackMapImages,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        )
    elif cfg.PRAGMA_LIDAR_PROGRESS_IMAGES:
        INT = partial(
            TM2020InterfaceLidarProgressImages,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        )
    elif cfg.PRAGMA_PROGRESS:
        INT = partial(
            TM2020InterfaceLidarProgress,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
    elif cfg.PRAGMA_TRACKMAP:
        INT = partial(
            TM2020InterfaceTrackMap,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
    else:
        INT = partial(
            TM2020InterfaceLidar,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
else:
    if cfg.PRAGMA_TQC_GRAB:
        INT = partial(
            TM2020InterfaceTQC,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
            crash_penalty=cfg.CRASH_PENALTY,
            constant_penalty=cfg.CONSTANT_PENALTY,
            checkpoint_reward=cfg.CHECKPOINT_REWARD,
            lap_reward=cfg.LAP_REWARD,
            min_nb_steps_before_failure=cfg.MIN_NB_STEPS_BEFORE_FAILURE,
            min_gas_warm_start=cfg.MIN_GAS_WARM_START,
        )
    elif cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        if cfg.USE_IMAGES:
            INT = partial(
                TM2020InterfaceIMPALA,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
                grayscale=cfg.GRAYSCALE,
                resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                crash_penalty=cfg.CRASH_PENALTY,
                constant_penalty=cfg.CONSTANT_PENALTY,
                checkpoint_reward=cfg.CHECKPOINT_REWARD,
                lap_reward=cfg.LAP_REWARD,
                min_nb_steps_before_failure=cfg.MIN_NB_STEPS_BEFORE_FAILURE,
            )
        else:
            INT = partial(
                TM2020InterfaceIMPALASophy,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
                grayscale=cfg.GRAYSCALE,
                resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                crash_penalty=cfg.CRASH_PENALTY,
                constant_penalty=cfg.CONSTANT_PENALTY,
                checkpoint_reward=cfg.CHECKPOINT_REWARD,
                lap_reward=cfg.LAP_REWARD,
                min_nb_steps_before_failure=cfg.MIN_NB_STEPS_BEFORE_FAILURE,
                min_gas_warm_start=cfg.MIN_GAS_WARM_START,
            )
    else:
        INT = partial(
            TM2020Interface,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        )

# RtGym config dict: default config + our interface + ENV RTGYM_CONFIG overrides
CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# -----------------------------------------------------------------------------
# 3. Sample compressor (for sending transitions over the network)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress_images
    elif cfg.PRAGMA_PROGRESS:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
else:
    if (
        cfg.PRAGMA_CUSTOM
        or cfg.PRAGMA_BEST
        or cfg.PRAGMA_BEST_TQC
        or cfg.PRAGMA_MBEST_TQC
        or cfg.PRAGMA_TQC_GRAB
    ):
        SAMPLE_COMPRESSOR = get_local_buffer_sample_mobilenet
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs

# -----------------------------------------------------------------------------
# 4. Observation preprocessor (env output → agent input)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
        OBS_PREPROCESSOR = obs_preprocessor_lidar_progress_images_act_in_obs
    elif cfg.PRAGMA_PROGRESS:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
else:
    if (
        cfg.PRAGMA_CUSTOM
        or cfg.PRAGMA_BEST
        or cfg.PRAGMA_BEST_TQC
        or cfg.PRAGMA_MBEST_TQC
        or cfg.PRAGMA_TQC_GRAB
    ):
        OBS_PREPROCESSOR = (
            obs_preprocessor_tqcgrab_act_in_obs
            if cfg.PRAGMA_TQC_GRAB
            else obs_preprocessor_mobilenet_act_in_obs
        )
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs

SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

# -----------------------------------------------------------------------------
# 5. Replay memory class (partial with size, batch_size, paths, etc.)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        raise AssertionError("not implemented")
    if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
        MEM: type[Any] = MemoryTMLidarProgressImages
    elif cfg.PRAGMA_PROGRESS:
        MEM = MemoryTMLidarProgress
    else:
        MEM = MemoryTMLidar
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC:
        MEM = MemoryTMBest
    elif cfg.PRAGMA_MBEST_TQC or cfg.PRAGMA_TQC_GRAB:
        MEM = MemoryR2D2 if (cfg.USE_IMAGES and not cfg.PRAGMA_TQC_GRAB) else MemoryR2D2woImages
    else:
        MEM = MemoryTMFull

MEMORY = partial(
    MEM,
    memory_size=MODEL_CONFIG["MEMORY_SIZE"],
    batch_size=MODEL_CONFIG["BATCH_SIZE"],
    sample_preprocessor=SAMPLE_PREPROCESSOR,
    dataset_path=cfg.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=cfg.CRC_DEBUG,
)

# -----------------------------------------------------------------------------
# 6. Training agent (SAC / TQC / REDQ with hyperparams from ALG_CONFIG)
# -----------------------------------------------------------------------------

_device = "cuda" if cfg.CUDA_TRAINING else "cpu"
_common_agent_kw = dict(
    device=_device,
    model_cls=TRAIN_MODEL,
    lr_actor=ALG_CONFIG["LR_ACTOR"],
    lr_critic=ALG_CONFIG["LR_CRITIC"],
    lr_entropy=ALG_CONFIG["LR_ENTROPY"],
    gamma=ALG_CONFIG["GAMMA"],
    polyak=ALG_CONFIG["POLYAK"],
    learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],
    target_entropy=ALG_CONFIG["TARGET_ENTROPY"],
    alpha=ALG_CONFIG["ALPHA"],
)

if ALG_NAME == "SAC":
    AGENT: Any = partial(
        SAC_Agent,
        **_common_agent_kw,
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG.get("BETAS_ACTOR"),
        betas_critic=ALG_CONFIG.get("BETAS_CRITIC"),
        l2_actor=ALG_CONFIG.get("L2_ACTOR"),
        l2_critic=ALG_CONFIG.get("L2_CRITIC"),
    )
elif ALG_NAME == "TQC":
    AGENT = partial(
        TQC_Agent,
        **_common_agent_kw,
        top_quantiles_to_drop=ALG_CONFIG["TOP_QUANTILES_TO_DROP"],
        quantiles_number=ALG_CONFIG["QUANTILES_NUMBER"],
        n_steps=ALG_CONFIG["N_STEPS"],
    )
else:
    AGENT = partial(
        REDQ_Agent,
        **_common_agent_kw,
        n=ALG_CONFIG["REDQ_N"],
        m=ALG_CONFIG["REDQ_M"],
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"],
    )

# -----------------------------------------------------------------------------
# 7. Trainer (TorchTrainingOffline partial: epochs, rounds, steps, intervals)
# -----------------------------------------------------------------------------

ENV_CLS = partial(
    GenericGymEnv,
    id=cfg.RTGYM_VERSION,
    gym_kwargs={"config": CONFIG_DICT},
)

_trainer_kw = dict(
    env_cls=ENV_CLS,
    memory_cls=MEMORY,
    epochs=MODEL_CONFIG["MAX_EPOCHS"],
    rounds=MODEL_CONFIG["ROUNDS_PER_EPOCH"],
    steps=MODEL_CONFIG["TRAINING_STEPS_PER_ROUND"],
    update_model_interval=MODEL_CONFIG["UPDATE_MODEL_INTERVAL"],
    update_buffer_interval=MODEL_CONFIG["UPDATE_BUFFER_INTERVAL"],
    max_training_steps_per_env_step=MODEL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
    python_profiling=cfg.PROFILE_TRAINER,
    pytorch_profiling=cfg.PYTORCH_PROFILER,
    training_agent_cls=AGENT,
    agent_scheduler=None,
    start_training=MODEL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"],
)

TRAINER = partial(TorchTrainingOffline, **_trainer_kw)

# -----------------------------------------------------------------------------
# 8. Checkpoint helpers (dump/load run instance, updater for SAC/TQC/REDQ)
# -----------------------------------------------------------------------------

DUMP_RUN_INSTANCE_FN = None
LOAD_RUN_INSTANCE_FN = None
UPDATER_FN = update_run_instance if ALG_NAME in ["SAC", "REDQSAC", "TQC"] else None
