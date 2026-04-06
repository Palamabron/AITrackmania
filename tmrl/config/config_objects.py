"""Build runtime objects from validated MainConfig (Hydra + Pydantic)."""

from __future__ import annotations

from typing import Any

import rtgym

import tmrl.config.constants as cfg
import tmrl.config.loader as loader
import tmrl.config.paths as cfg_paths
import tmrl.custom.models.image_input.impala as impala_module
import tmrl.custom.models.hybrid_input.sophy as Sophy_models
from tmrl.config.schema.main import MainConfig
from tmrl.custom.custom_algorithms import IQNAgent
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_algorithms import TQCAgent as TQC_Agent
from tmrl.custom.custom_algorithms.sdsac import SDSACAgent
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
from tmrl.custom.models.discrete_actions.iqn_discrete_q_network import DQNActor
from tmrl.custom.models.hybrid_input.sophy import (
    SophyResidualActorCritic,
    SquashedActorSophyResidual,
)
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
algorithm = M.algorithm
model_cfg = M.model

ALG_NAME = algorithm.name

if ALG_NAME not in ("SAC", "REDQSAC", "TQC", "IQN", "SDSAC"):
    raise ValueError(f"Unknown algorithm {ALG_NAME!r}. Supported: SAC, REDQSAC, TQC, IQN, SDSAC.")

_USE_IMAGES_MOBILENET_PIPELINE = cfg.USE_IMAGES_MOBILENET_PIPELINE
_USE_NON_LIDAR_IMAGE_STACK = _USE_IMAGES_MOBILENET_PIPELINE or cfg.USE_IMAGES_R2D2_SEQUENCE_BUFFER


def _validate_runtime_compatibility() -> None:
    """Fail fast on unsupported algorithm/interface combinations.

    These checks intentionally mirror runtime selection branches so users get a clear
    config-time error message instead of an opaque assertion deeper in model selection.
    """
    advanced_iface = cfg.USE_IMAGES_MOBILENET_PIPELINE or cfg.USE_IMAGES_R2D2_SEQUENCE_BUFFER
    vanilla_image_iface = not cfg.PRAGMA_LIDAR and not advanced_iface
    rtgym_iface = M.environment.rtgym_interface

    if vanilla_image_iface and ALG_NAME != "SAC":
        raise ValueError(
            f"Unsupported combination: algorithm.name={ALG_NAME!r} "
            f"with environment.rtgym_interface={rtgym_iface!r}. "
            "For non-lidar vanilla image interfaces only SAC is supported "
            "(set algorithm.name='SAC' or choose a lidar/advanced interface)."
        )

    if advanced_iface and ALG_NAME not in ("TQC", "SAC", "IQN", "SDSAC"):
        raise ValueError(
            f"Unsupported combination: algorithm.name={ALG_NAME!r} "
            f"with advanced interface {rtgym_iface!r}. "
            "Supported algorithms here: TQC, SAC, IQN, SDSAC."
        )

    if cfg.PRAGMA_LIDAR and cfg.PRAGMA_RNN and ALG_NAME != "SAC":
        raise ValueError(
            f"Unsupported combination: PRAGMA_RNN=true with algorithm.name={ALG_NAME!r}. "
            "RNN runtime path currently supports only SAC."
        )


def _train_model_and_policy() -> tuple[Any, Any]:
    """Select (train_model_cls_or_partial, policy_partial) from model + algorithm.

    Branch conditions must stay aligned with :func:`tmrl.config.effective_config.model_policy_route`
    so ``--explain-active-config`` matches runtime selection.
    """
    alg = algorithm
    arch = model_cfg
    rtgym_iface = M.environment.rtgym_interface
    if cfg.PRAGMA_LIDAR:
        if ALG_NAME in ("IQN", "SDSAC"):
            iqn_kw = {
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
                "n_cos": alg.iqn_n_cos,
                "dueling": alg.iqn_dueling,
                "n_actions": alg.iqn_n_actions,
                "n_quantiles_eval": alg.iqn_num_quantiles_eval,
                "epsilon": alg.iqn_epsilon_start,
                "explore_repeat_steps": alg.iqn_explore_repeat_steps,
            }
            return None, partial(DQNActor, **iqn_kw)
        if (cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES) and ALG_NAME == "SAC":
            lidar_images_kw = {
                "image_index": 3,
                "embed_dim": arch.frozen_effnet_embed_dim,
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
                "width_mult": arch.frozen_effnet_width_mult,
            }
            return (
                partial(FrozenEffNetResidualActorCritic, **lidar_images_kw),
                partial(SquashedGaussianFrozenEffNetResidualActor, **lidar_images_kw),
            )
        if cfg.PRAGMA_RNN:
            assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
            return RNNActorCritic, SquashedGaussianRNNActor
        if arch.use_residual_mlp:
            residual_kw = {
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
            }
            train_model = (
                partial(ResidualMLPActorCritic, **residual_kw)
                if ALG_NAME == "SAC"
                else partial(REDQResidualMLPActorCritic, n=alg.redq_n, **residual_kw)
            )
            return train_model, partial(SquashedGaussianResidualMLPActor, **residual_kw)
        return (
            MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic,
            SquashedGaussianMLPActor,
        )

    if cfg.USE_IMAGES_R2D2_SEQUENCE_BUFFER:
        if ALG_NAME in ("IQN", "SDSAC"):
            iqn_kw = {
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
                "n_cos": alg.iqn_n_cos,
                "dueling": alg.iqn_dueling,
                "n_actions": alg.iqn_n_actions,
                "n_quantiles_eval": alg.iqn_num_quantiles_eval,
                "epsilon": alg.iqn_epsilon_start,
                "explore_repeat_steps": alg.iqn_explore_repeat_steps,
            }
            return None, partial(DQNActor, **iqn_kw)
        if (
            cfg.USE_IMAGES
            and not cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT
            and arch.use_frozen_effnet
            and ALG_NAME == "SAC"
        ):
            frozen_effnet_kw = {
                "embed_dim": arch.frozen_effnet_embed_dim,
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
                "width_mult": arch.frozen_effnet_width_mult,
            }
            return (
                partial(FrozenEffNetResidualActorCritic, **frozen_effnet_kw),
                partial(SquashedGaussianFrozenEffNetResidualActor, **frozen_effnet_kw),
            )
        if cfg.USE_IMAGES and not cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT:
            return impala_module.QRCNNActorCritic, impala_module.SquashedActorQRCNN
        if (
            cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT
            and not cfg.USE_IMAGES
            and arch.use_sophy_residual_actor
        ):
            res_sophy_kw = {
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
            }
            return (
                partial(SophyResidualActorCritic, **res_sophy_kw),
                partial(SquashedActorSophyResidual, **res_sophy_kw),
            )
        return Sophy_models.SophyActorCritic, Sophy_models.SquashedActorSophy

    if cfg.PRAGMA_RNN:
        raise ValueError(
            "Unsupported combination: PRAGMA_RNN=true with non-lidar interface "
            f"{rtgym_iface!r}."
        )
    if ALG_NAME != "SAC":
        raise ValueError(
            f"Unsupported combination: algorithm.name={ALG_NAME!r} "
            f"for interface {rtgym_iface!r}; "
            "only SAC is supported in this runtime branch."
        )
    if cfg.GRAYSCALE:
        return VanillaCNNActorCritic, SquashedGaussianVanillaCNNActor
    return VanillaColorCNNActorCritic, SquashedGaussianVanillaColorCNNActor


_validate_runtime_compatibility()
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
    }
    if cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT:
        return partial(TM2020InterfaceTQC, **common_image, **common_reward)
    if _USE_IMAGES_MOBILENET_PIPELINE:
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
    if cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT:
        return "ImagesWorldTelemetry" if cfg.USE_IMAGES else "WorldTelemetry"
    if _USE_IMAGES_MOBILENET_PIPELINE:
        return "ImagesMobilenet" if cfg.USE_IMAGES else "ImagesMobilenetVector"
    return "Full"


INTERFACE_DISPLAY_NAME = _interface_display_name()

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = RTGYM_INTERFACE_CLASS
CONFIG_DICT_MODIFIERS = M.environment.rtgym_config_dict()
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v


def _pick_sample_compressor() -> Any:
    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
            return get_local_buffer_sample_lidar_progress_images
        if cfg.PRAGMA_PROGRESS:
            return get_local_buffer_sample_lidar_progress
        return get_local_buffer_sample_lidar
    if _USE_NON_LIDAR_IMAGE_STACK:
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
    if _USE_NON_LIDAR_IMAGE_STACK:
        return (
            obs_preprocessor_tqcgrab_act_in_obs
            if cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT
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
    if cfg.USE_IMAGES_MOBILENET_PIPELINE:
        return MemoryTMBest
    if cfg.USE_IMAGES_R2D2_SEQUENCE_BUFFER:
        return (
            MemoryR2D2
            if (cfg.USE_IMAGES and not cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT)
            else MemoryR2D2woImages
        )
    return MemoryTMFull


MEM = _pick_memory_class()

MEMORY = partial(
    MEM,
    memory_size=cfg.MEMORY_SIZE,
    batch_size=cfg.BATCH_SIZE,
    sample_preprocessor=SAMPLE_PREPROCESSOR,
    dataset_path=cfg_paths.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=cfg.CRC_DEBUG,
)

_device = "cuda" if cfg.CUDA_TRAINING else "cpu"
alg = algorithm
_common_agent_kw = {
    "device": _device,
    "model_cls": TRAIN_MODEL,
    "lr_actor": alg.lr_actor,
    "lr_critic": alg.lr_critic,
    "lr_entropy": alg.lr_entropy,
    "gamma": alg.gamma,
    "polyak": alg.polyak,
    "learn_entropy_coef": alg.learn_entropy_coef,
    "target_entropy": alg.target_entropy,
    "alpha": alg.alpha,
}


def _build_agent() -> Any:
    if ALG_NAME == "SAC":
        return partial(
            SAC_Agent,
            **_common_agent_kw,
            optimizer_actor=alg.optimizer_actor,
            optimizer_critic=alg.optimizer_critic,
            betas_actor=alg.betas_actor,
            betas_critic=alg.betas_critic,
            l2_actor=alg.l2_actor,
            l2_critic=alg.l2_critic,
        )
    if ALG_NAME == "TQC":
        return partial(
            TQC_Agent,
            **_common_agent_kw,
            top_quantiles_to_drop=alg.top_quantiles_to_drop,
            quantiles_number=alg.quantiles_number,
            n_steps=alg.n_steps,
        )
    if ALG_NAME == "REDQSAC":
        return partial(
            REDQ_Agent,
            **_common_agent_kw,
            n=alg.redq_n,
            m=alg.redq_m,
            q_updates_per_policy_update=alg.redq_q_updates_per_policy_update,
        )
    if ALG_NAME == "IQN":
        return partial(
            IQNAgent,
            device=_device,
            hidden_dim=model_cfg.residual_mlp_hidden_dim,
            num_blocks=model_cfg.residual_mlp_num_blocks,
            n_quantiles_train=alg.iqn_num_quantiles_train,
            n_quantiles_target=alg.iqn_num_quantiles_target,
            n_quantiles_eval=alg.iqn_num_quantiles_eval,
            n_cos=alg.iqn_n_cos,
            lr=alg.iqn_lr,
            gamma=alg.gamma,
            epsilon_start=alg.iqn_epsilon_start,
            epsilon_end=alg.iqn_epsilon_end,
            epsilon_decay_steps=alg.iqn_epsilon_decay_steps,
            epsilon_schedule_mode=alg.iqn_epsilon_schedule_mode,
            epsilon_cosine_t0=alg.iqn_epsilon_cosine_t0,
            epsilon_cosine_tmult=alg.iqn_epsilon_cosine_tmult,
            epsilon_cosine_decay=alg.iqn_epsilon_cosine_decay,
            epsilon_cosine_initial_amplitude=alg.iqn_epsilon_cosine_initial_amplitude,
            epsilon_cosine_floor_fraction=alg.iqn_epsilon_cosine_floor_fraction,
            epsilon_cosine_floor_steps=alg.iqn_epsilon_cosine_floor_steps,
            explore_repeat_steps=int(alg.iqn_explore_repeat_steps),
            n_steps=alg.n_steps,
            target_update_freq=alg.iqn_target_update_freq,
            double_dqn=alg.iqn_double_dqn,
            dueling=alg.iqn_dueling,
        )
    if ALG_NAME == "SDSAC":
        return partial(
            SDSACAgent,
            device=_device,
            hidden_dim=model_cfg.residual_mlp_hidden_dim,
            num_blocks_actor=cfg.RESIDUAL_MLP_NUM_BLOCKS_ACTOR,
            num_blocks_critic=cfg.RESIDUAL_MLP_NUM_BLOCKS_CRITIC,
            n_cos=alg.iqn_n_cos,
            n_actions=alg.iqn_n_actions,
            gamma=alg.gamma,
            lr_actor=alg.lr_actor,
            lr_critic=alg.lr_critic,
            lr_alpha=alg.lr_entropy,
            tau_polyak=float(1.0 - alg.polyak),
            n_steps=alg.n_steps if alg.n_steps > 0 else 1,
            auto_alpha=alg.learn_entropy_coef,
            alpha_init=alg.alpha,
            use_avg_q=alg.sdsac_avg_q,
            use_clip_q=alg.sdsac_clip_q,
            clip_q_epsilon=alg.sdsac_clip_q_epsilon,
            use_entropy_penalty=alg.sdsac_entropy_penalty,
            entropy_penalty_beta=alg.sdsac_entropy_penalty_beta,
            eder_oversample_ratio=alg.eder_oversample_ratio,
            weight_decay=alg.actor_weight_decay,
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
    epochs=cfg.MAX_EPOCHS,
    rounds=cfg.ROUNDS_PER_EPOCH,
    steps=cfg.TRAINING_STEPS_PER_ROUND,
    update_model_interval=cfg.UPDATE_MODEL_INTERVAL,
    update_buffer_interval=cfg.UPDATE_BUFFER_INTERVAL,
    max_training_steps_per_env_step=cfg.MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP,
    python_profiling=cfg.PROFILE_TRAINER,
    pytorch_profiling=cfg.PYTORCH_PROFILER,
    training_agent_cls=AGENT,
    agent_scheduler=None,
    start_training=cfg.ENVIRONMENT_STEPS_BEFORE_TRAINING,
)

DUMP_RUN_INSTANCE_FN = None
LOAD_RUN_INSTANCE_FN = None
UPDATER_FN = update_run_instance
