"""Build runtime objects from validated MainConfig (Hydra + Pydantic)."""

from __future__ import annotations

from typing import Any

import rtgym

import tmrl.config.constants as cfg
import tmrl.config.loader as loader
import tmrl.config.paths as cfg_paths
import tmrl.custom.models.IMPALA as impala_module  # noqa: N811
import tmrl.custom.models.Sophy as Sophy_models
from tmrl.config.models import MainConfig
from tmrl.custom.custom_algorithms import IQNAgent
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_algorithms import TQCAgent as TQC_Agent
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.interfaces.TM2020InterfaceIMPALA import TM2020InterfaceIMPALA
from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from tmrl.custom.interfaces.TM2020InterfaceLidarImages import TM2020InterfaceLidarProgressImages
from tmrl.custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress
from tmrl.custom.interfaces.TM2020InterfaceSophy import TM2020InterfaceIMPALASophy
from tmrl.custom.interfaces.TM2020InterfaceTQC import TM2020InterfaceTQC
from tmrl.custom.interfaces.TM2020InterfaceTrackMap import TM2020InterfaceTrackMap
from tmrl.custom.interfaces.TM2020InterfaceTrackMapImages import TM2020InterfaceTrackMapImages
from tmrl.custom.memories import (
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
from tmrl.custom.models import (
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
from tmrl.custom.models.DQNNet import DQNActor
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

M: MainConfig = loader.MAIN_CONFIG
ALG = M.ALG
MOD = M.MODEL

ALG_CONFIG = loader.TMRL_CONFIG["ALG"]
ALG_NAME = ALG.ALGORITHM
MODEL_CONFIG = loader.TMRL_CONFIG["MODEL"]

if ALG_NAME not in ("SAC", "REDQSAC", "TQC", "IQN"):
    raise ValueError(
        f"Unknown algorithm '{ALG_NAME}'. Must be one of: SAC, REDQSAC, TQC, IQN. "
        f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."
    )

_USE_CUSTOM_OR_BEST = (
    cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC
)
_USE_ADVANCED_RTGYM_INTERFACE = _USE_CUSTOM_OR_BEST or cfg.PRAGMA_TQC_GRAB


def _train_model_and_policy() -> tuple[Any, Any]:
    """Select (train_model_cls_or_partial, policy_partial) from architecture + algorithm."""
    if cfg.PRAGMA_LIDAR:
        if (cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES) and ALG_NAME == "SAC":
            lidar_images_kw = {
                "image_index": 3,
                "embed_dim": MOD.FROZEN_EFFNET_EMBED_DIM,
                "hidden_dim": MOD.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": MOD.RESIDUAL_MLP_NUM_BLOCKS,
                "width_mult": MOD.FROZEN_EFFNET_WIDTH_MULT,
            }
            return (
                partial(FrozenEffNetResidualActorCritic, **lidar_images_kw),
                partial(SquashedGaussianFrozenEffNetResidualActor, **lidar_images_kw),
            )
        if cfg.PRAGMA_RNN:
            assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
            return RNNActorCritic, SquashedGaussianRNNActor
        if MOD.USE_RESIDUAL_MLP:
            residual_kw = {
                "hidden_dim": MOD.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": MOD.RESIDUAL_MLP_NUM_BLOCKS,
            }
            train_model = (
                partial(ResidualMLPActorCritic, **residual_kw)
                if ALG_NAME == "SAC"
                else partial(REDQResidualMLPActorCritic, n=ALG.REDQ_N, **residual_kw)
            )
            return train_model, partial(SquashedGaussianResidualMLPActor, **residual_kw)
        return (
            MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic,
            SquashedGaussianMLPActor,
        )

    if cfg.PRAGMA_MBEST_TQC or cfg.PRAGMA_TQC_GRAB:
        assert ALG_NAME in ("TQC", "SAC", "IQN"), f"{ALG_NAME} is not implemented here."
        if ALG_NAME == "IQN":
            iqn_kw = {
                "hidden_dim": MOD.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": MOD.RESIDUAL_MLP_NUM_BLOCKS,
                "n_cos": ALG.IQN_N_COS,
                "dueling": ALG.IQN_DUELING,
                "n_actions": ALG.IQN_N_ACTIONS,
                "n_quantiles_eval": ALG.IQN_NUM_QUANTILES_EVAL,
                "epsilon": ALG.IQN_EPSILON_START,
                "explore_repeat_steps": ALG.IQN_EXPLORE_REPEAT_STEPS,
            }
            return None, partial(DQNActor, **iqn_kw)
        if (
            cfg.USE_IMAGES
            and not cfg.PRAGMA_TQC_GRAB
            and MOD.USE_FROZEN_EFFNET
            and ALG_NAME == "SAC"
        ):
            frozen_effnet_kw = {
                "embed_dim": MOD.FROZEN_EFFNET_EMBED_DIM,
                "hidden_dim": MOD.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": MOD.RESIDUAL_MLP_NUM_BLOCKS,
                "width_mult": MOD.FROZEN_EFFNET_WIDTH_MULT,
            }
            return (
                partial(FrozenEffNetResidualActorCritic, **frozen_effnet_kw),
                partial(SquashedGaussianFrozenEffNetResidualActor, **frozen_effnet_kw),
            )
        if cfg.USE_IMAGES and not cfg.PRAGMA_TQC_GRAB:
            return impala_module.QRCNNActorCritic, impala_module.SquashedActorQRCNN
        if cfg.PRAGMA_TQC_GRAB and not cfg.USE_IMAGES and MOD.USE_RESIDUAL_SOPHY:
            res_sophy_kw = {
                "hidden_dim": MOD.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": MOD.RESIDUAL_MLP_NUM_BLOCKS,
            }
            return (
                partial(SophyResidualActorCritic, **res_sophy_kw),
                partial(SquashedActorSophyResidual, **res_sophy_kw),
            )
        return Sophy_models.SophyActorCritic, Sophy_models.SquashedActorSophy

    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
    assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
    if cfg.GRAYSCALE:
        return VanillaCNNActorCritic, SquashedGaussianVanillaCNNActor
    return VanillaColorCNNActorCritic, SquashedGaussianVanillaColorCNNActor


TRAIN_MODEL, POLICY = _train_model_and_policy()


def _rtgym_interface_partial() -> Any:
    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_TRACKMAP_IMAGES:
            return partial(
                TM2020InterfaceTrackMapImages,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
                grayscale=cfg.GRAYSCALE,
                resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
            )
        if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES:
            return partial(
                TM2020InterfaceLidarProgressImages,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
                grayscale=cfg.GRAYSCALE,
                resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
            )
        if cfg.PRAGMA_PROGRESS:
            return partial(
                TM2020InterfaceLidarProgress,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
            )
        if cfg.PRAGMA_TRACKMAP:
            return partial(
                TM2020InterfaceTrackMap,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
            )
        return partial(
            TM2020InterfaceLidar,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )

    common_image = {
        "img_hist_len": cfg.IMG_HIST_LEN,
        "gamepad": cfg.PRAGMA_GAMEPAD,
        "grayscale": cfg.GRAYSCALE,
        "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
    }
    common_reward = {
        "crash_penalty": cfg.CRASH_PENALTY,
        "constant_penalty": cfg.CONSTANT_PENALTY,
        "checkpoint_reward": cfg.CHECKPOINT_REWARD,
        "lap_reward": cfg.LAP_REWARD,
        "min_nb_steps_before_failure": cfg.MIN_NB_STEPS_BEFORE_FAILURE,
    }
    if cfg.PRAGMA_TQC_GRAB:
        return partial(TM2020InterfaceTQC, **common_image, **common_reward)
    if _USE_CUSTOM_OR_BEST:
        if cfg.USE_IMAGES:
            return partial(TM2020InterfaceIMPALA, **common_image, **common_reward)
        return partial(TM2020InterfaceIMPALASophy, **common_image, **common_reward)
    return partial(TM2020Interface, **common_image)


RTGYM_INTERFACE_CLASS = _rtgym_interface_partial()


def _interface_display_name() -> str:
    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_TRACKMAP_IMAGES:
            return "TrackMapImages"
        if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES:
            return "LidarProgressImages"
        if cfg.PRAGMA_PROGRESS:
            return "LidarProgress"
        if cfg.PRAGMA_TRACKMAP:
            return "TrackMap"
        return "Lidar"
    if cfg.PRAGMA_TQC_GRAB:
        return "TQCGrab"
    if _USE_CUSTOM_OR_BEST:
        return "IMPALA" if cfg.USE_IMAGES else "IMPALASophy"
    return "Full"


INTERFACE_DISPLAY_NAME = _interface_display_name()

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = RTGYM_INTERFACE_CLASS
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v


def _pick_sample_compressor() -> Any:
    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
            return get_local_buffer_sample_lidar_progress_images
        if cfg.PRAGMA_PROGRESS:
            return get_local_buffer_sample_lidar_progress
        return get_local_buffer_sample_lidar
    if _USE_ADVANCED_RTGYM_INTERFACE:
        return get_local_buffer_sample_mobilenet
    return get_local_buffer_sample_tm20_imgs


SAMPLE_COMPRESSOR = _pick_sample_compressor()


def _pick_obs_preprocessor() -> Any:
    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
            return obs_preprocessor_lidar_progress_images_act_in_obs
        if cfg.PRAGMA_PROGRESS:
            return obs_preprocessor_tm_lidar_progress_act_in_obs
        return obs_preprocessor_tm_lidar_act_in_obs
    if _USE_ADVANCED_RTGYM_INTERFACE:
        return (
            obs_preprocessor_tqcgrab_act_in_obs
            if cfg.PRAGMA_TQC_GRAB
            else obs_preprocessor_mobilenet_act_in_obs
        )
    return obs_preprocessor_tm_act_in_obs


OBS_PREPROCESSOR = _pick_obs_preprocessor()
SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"


def _pick_memory_class() -> type[Any]:
    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
            return MemoryTMLidarProgressImages
        if cfg.PRAGMA_PROGRESS:
            return MemoryTMLidarProgress
        return MemoryTMLidar
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC:
        return MemoryTMBest
    if cfg.PRAGMA_MBEST_TQC or cfg.PRAGMA_TQC_GRAB:
        return MemoryR2D2 if (cfg.USE_IMAGES and not cfg.PRAGMA_TQC_GRAB) else MemoryR2D2woImages
    return MemoryTMFull


MEM = _pick_memory_class()

MEMORY = partial(
    MEM,
    memory_size=MOD.MEMORY_SIZE,
    batch_size=MOD.BATCH_SIZE,
    sample_preprocessor=SAMPLE_PREPROCESSOR,
    dataset_path=cfg_paths.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=cfg.CRC_DEBUG,
)

_device = "cuda" if cfg.CUDA_TRAINING else "cpu"
_common_agent_kw = {
    "device": _device,
    "model_cls": TRAIN_MODEL,
    "lr_actor": ALG.LR_ACTOR,
    "lr_critic": ALG.LR_CRITIC,
    "lr_entropy": ALG.LR_ENTROPY,
    "gamma": ALG.GAMMA,
    "polyak": ALG.POLYAK,
    "learn_entropy_coef": ALG.LEARN_ENTROPY_COEF,
    "target_entropy": ALG.TARGET_ENTROPY,
    "alpha": ALG.ALPHA,
}


def _build_agent() -> Any:
    if ALG_NAME == "SAC":
        return partial(
            SAC_Agent,
            **_common_agent_kw,
            optimizer_actor=ALG.OPTIMIZER_ACTOR,
            optimizer_critic=ALG.OPTIMIZER_CRITIC,
            betas_actor=ALG.BETAS_ACTOR,
            betas_critic=ALG.BETAS_CRITIC,
            l2_actor=ALG.L2_ACTOR,
            l2_critic=ALG.L2_CRITIC,
        )
    if ALG_NAME == "TQC":
        return partial(
            TQC_Agent,
            **_common_agent_kw,
            top_quantiles_to_drop=ALG.TOP_QUANTILES_TO_DROP,
            quantiles_number=ALG.QUANTILES_NUMBER,
            n_steps=ALG.N_STEPS,
        )
    if ALG_NAME == "REDQSAC":
        return partial(
            REDQ_Agent,
            **_common_agent_kw,
            n=ALG.REDQ_N,
            m=ALG.REDQ_M,
            q_updates_per_policy_update=ALG.REDQ_Q_UPDATES_PER_POLICY_UPDATE,
        )
    if ALG_NAME == "IQN":
        return partial(
            IQNAgent,
            device=_device,
            hidden_dim=MOD.RESIDUAL_MLP_HIDDEN_DIM,
            num_blocks=MOD.RESIDUAL_MLP_NUM_BLOCKS,
            n_quantiles_train=ALG.IQN_NUM_QUANTILES_TRAIN,
            n_quantiles_target=ALG.IQN_NUM_QUANTILES_TARGET,
            n_quantiles_eval=ALG.IQN_NUM_QUANTILES_EVAL,
            n_cos=ALG.IQN_N_COS,
            lr=ALG.IQN_LR,
            gamma=ALG.GAMMA,
            epsilon_start=ALG.IQN_EPSILON_START,
            epsilon_end=ALG.IQN_EPSILON_END,
            epsilon_decay_steps=ALG.IQN_EPSILON_DECAY_STEPS,
            epsilon_schedule_mode=ALG.IQN_EPSILON_SCHEDULE_MODE,
            epsilon_cosine_t0=ALG.IQN_EPSILON_COSINE_T0,
            epsilon_cosine_tmult=ALG.IQN_EPSILON_COSINE_TMULT,
            epsilon_cosine_decay=ALG.IQN_EPSILON_COSINE_DECAY,
            epsilon_cosine_initial_amplitude=ALG.IQN_EPSILON_COSINE_INITIAL_AMPLITUDE,
            epsilon_cosine_floor_fraction=ALG.IQN_EPSILON_COSINE_FLOOR_FRACTION,
            epsilon_cosine_floor_steps=ALG.IQN_EPSILON_COSINE_FLOOR_STEPS,
            explore_repeat_steps=int(ALG.IQN_EXPLORE_REPEAT_STEPS),
            n_steps=ALG.N_STEPS,
            target_update_freq=ALG.IQN_TARGET_UPDATE_FREQ,
            double_dqn=ALG.IQN_DOUBLE_DQN,
            dueling=ALG.IQN_DUELING,
        )
    raise ValueError(f"Unknown algorithm: {ALG_NAME}")


AGENT: Any = _build_agent()

ENV_CLS = partial(
    GenericGymEnv,
    id=loader.RTGYM_VERSION,
    gym_kwargs={"config": CONFIG_DICT},
)

TRAINER = partial(
    TorchTrainingOffline,
    env_cls=ENV_CLS,
    memory_cls=MEMORY,
    epochs=MOD.MAX_EPOCHS,
    rounds=MOD.ROUNDS_PER_EPOCH,
    steps=MOD.TRAINING_STEPS_PER_ROUND,
    update_model_interval=MOD.UPDATE_MODEL_INTERVAL,
    update_buffer_interval=MOD.UPDATE_BUFFER_INTERVAL,
    max_training_steps_per_env_step=MOD.MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP,
    python_profiling=cfg.PROFILE_TRAINER,
    pytorch_profiling=cfg.PYTORCH_PROFILER,
    training_agent_cls=AGENT,
    agent_scheduler=None,
    start_training=MOD.ENVIRONMENT_STEPS_BEFORE_TRAINING,
)

DUMP_RUN_INSTANCE_FN = None
LOAD_RUN_INSTANCE_FN = None
UPDATER_FN = update_run_instance
