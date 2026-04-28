"""Build runtime objects from validated MainConfig (Hydra + Pydantic)."""

from __future__ import annotations

from typing import Any

import rtgym
from loguru import logger

import tmrl.config.constants as cfg
import tmrl.config.loader as loader
import tmrl.config.paths as cfg_paths

# Side-effect imports: trigger @register decorators for interfaces, memories, and models
import tmrl.custom.custom_algorithms.iqn
import tmrl.custom.custom_algorithms.redq_sac
import tmrl.custom.custom_algorithms.sac
import tmrl.custom.custom_algorithms.sdsac
import tmrl.custom.custom_algorithms.tqc
import tmrl.custom.interfaces.boundary
import tmrl.custom.interfaces.car_state
import tmrl.custom.interfaces.lidar
import tmrl.custom.interfaces.vision
import tmrl.custom.memories.base
import tmrl.custom.memories.r2d2
import tmrl.custom.memories.tm_best
import tmrl.custom.memories.tm_full
import tmrl.custom.memories.tm_lidar
import tmrl.custom.models.discrete_actions.iqn_discrete_q_network
import tmrl.custom.models.hybrid_input.gnn_effnet_sophy
import tmrl.custom.models.hybrid_input.sophy
import tmrl.custom.models.image_input.impala  # noqa: F401
from tmrl.config.schema.main import MainConfig
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.custom.memories import (
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
from tmrl.custom.tm.tm_preprocessors import (
    obs_preprocessor_lidar_progress_images_act_in_obs,
    obs_preprocessor_mobilenet_act_in_obs,
    obs_preprocessor_tm_act_in_obs,
    obs_preprocessor_tm_lidar_act_in_obs,
    obs_preprocessor_tm_lidar_progress_act_in_obs,
    obs_preprocessor_tqcgrab_act_in_obs,
)
from tmrl.envs import GenericGymEnv
from tmrl.registry import ALGORITHMS, INTERFACES, MEMORIES, MODELS
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
    vanilla_image_iface = not cfg.USE_LIDAR_OBSERVATIONS and not advanced_iface
    rtgym_iface = M.environment.rtgym_interface

    if cfg.USE_LIDAR_OBSERVATIONS and ALG_NAME not in ("SAC", "REDQSAC", "IQN", "SDSAC"):
        raise ValueError(
            f"Unsupported combination: algorithm.name={ALG_NAME!r} "
            f"with LIDAR interface {rtgym_iface!r}. "
            "Supported LIDAR algorithms: SAC, REDQSAC, IQN, SDSAC."
        )

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

    if cfg.USE_LIDAR_OBSERVATIONS and cfg.USE_RNN and ALG_NAME != "SAC":
        raise ValueError(
            f"Unsupported combination: USE_RNN=true with algorithm.name={ALG_NAME!r}. "
            "RNN runtime path currently supports only SAC."
        )


def _model_arch_kwargs() -> dict[str, Any]:
    """Extract model architecture params from Pydantic ModelConfig for constructor injection."""
    arch = model_cfg
    _rnn_hidden = arch.rnn_hidden_size if arch.rnn_hidden_size > 0 else None
    return {
        "seed": int(M.environment.seed),
        "split_track_observation": arch.split_track_observation,
        "track_encoder": arch.track_encoder,
        "use_rnn": arch.use_rnn,
        "rnn_hidden_size": _rnn_hidden,
        "api_layernorm": arch.api_layernorm,
        "mlp_layernorm": arch.mlp_layernorm,
        "use_simbav2": arch.use_simbav2,
        "output_dropout": arch.output_dropout,
        "noisy_linear_critic": arch.noisy_linear_critic,
        "noisy_linear_actor": arch.noisy_linear_actor,
        "binary_brake": arch.binary_brake,
        "gnn_hidden": arch.gnn_hidden,
        "gnn_layers": arch.gnn_layers,
        "r2d2_sequence_length": int(algorithm.r2d2_sequence_length),
        "r2d2_burn_in": int(algorithm.r2d2_burn_in),
        "quantiles_number": int(algorithm.quantiles_number),
        "init_gas_bias": 0.0,
        "rnn_sizes": list(arch.rnn_sizes),
        "rnn_lens": list(arch.rnn_lens),
        "rnn_dropout": arch.rnn_dropout,
    }


def _train_model_and_policy() -> tuple[Any, Any]:
    """Select (train_model_cls_or_partial, policy_partial) from model + algorithm.

    Branch conditions must stay aligned with :func:`tmrl.config.effective_config.model_policy_route`
    so ``--explain-active-config`` matches runtime selection.
    """
    alg = algorithm
    arch = model_cfg
    rtgym_iface = M.environment.rtgym_interface
    _arch_kw = _model_arch_kwargs()

    dqn_actor_cls = MODELS.get("dqn_actor")

    if cfg.USE_LIDAR_OBSERVATIONS:
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
                **_arch_kw,
            }
            return None, partial(dqn_actor_cls, **iqn_kw)
        if (cfg.USE_LIDAR_PROGRESS_IMAGES or cfg.USE_TRACKMAP_IMAGES) and ALG_NAME == "SAC":
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
        if cfg.USE_RNN:
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
                **_arch_kw,
            }
            return None, partial(dqn_actor_cls, **iqn_kw)
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
            impala_ac_cls = MODELS.get("impala_ac")
            impala_actor_cls = MODELS.get("impala_qr_actor")
            _impala_kw = {
                "rnn_sizes": list(arch.rnn_sizes),
                "rnn_lens": list(arch.rnn_lens),
                "api_mlp_sizes": list(arch.api_mlp_sizes),
                "seed": int(M.environment.seed),
            }
            return (
                partial(impala_ac_cls, **_impala_kw),
                partial(
                    impala_actor_cls,
                    **_impala_kw,
                    **{
                        k: _arch_kw[k]
                        for k in (
                            "api_layernorm",
                            "mlp_layernorm",
                            "output_dropout",
                            "noisy_linear_actor",
                            "rnn_dropout",
                        )
                    },
                    grayscale=cfg.GRAYSCALE,
                ),
            )
        if (
            cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT
            and not cfg.USE_IMAGES
            and arch.use_sophy_residual_actor
        ):
            sophy_res_ac_cls = MODELS.get("sophy_residual_ac")
            sophy_res_actor_cls = MODELS.get("sophy_residual_actor")
            res_sophy_kw = {
                "hidden_dim": arch.residual_mlp_hidden_dim,
                "num_blocks": arch.residual_mlp_num_blocks,
                "seed": int(M.environment.seed),
            }
            _actor_kw = {
                **res_sophy_kw,
                **{
                    k: _arch_kw[k]
                    for k in (
                        "split_track_observation",
                        "use_rnn",
                        "rnn_hidden_size",
                        "track_encoder",
                        "api_layernorm",
                        "binary_brake",
                        "init_gas_bias",
                        "output_dropout",
                        "r2d2_sequence_length",
                        "r2d2_burn_in",
                        "use_simbav2",
                        "gnn_hidden",
                        "gnn_layers",
                    )
                },
            }
            return (
                partial(sophy_res_ac_cls, **res_sophy_kw),
                partial(sophy_res_actor_cls, **_actor_kw),
            )
        sophy_ac_cls = MODELS.get("sophy_ac")
        sophy_actor_cls = MODELS.get("sophy_actor")
        _sophy_kw = {
            "rnn_sizes": list(arch.rnn_sizes),
            "rnn_lens": list(arch.rnn_lens),
            "api_mlp_sizes": list(arch.api_mlp_sizes),
            "seed": int(M.environment.seed),
        }
        return (
            partial(sophy_ac_cls, **_sophy_kw),
            partial(
                sophy_actor_cls,
                **_sophy_kw,
                **{
                    k: _arch_kw[k]
                    for k in (
                        "api_layernorm",
                        "mlp_layernorm",
                        "noisy_linear_actor",
                        "output_dropout",
                        "init_gas_bias",
                    )
                },
            ),
        )

    if cfg.USE_RNN:
        raise ValueError(
            f"Unsupported combination: USE_RNN=true with non-lidar interface {rtgym_iface!r}."
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


def _determine_interface_name() -> str:
    """Derive the interface registry key from interface feature flags."""
    if cfg.USE_LIDAR_OBSERVATIONS:
        if cfg.USE_TRACKMAP_IMAGES:
            return "trackmap_images"
        if cfg.USE_LIDAR_PROGRESS_IMAGES:
            return "lidar_progress_images"
        if cfg.USE_LIDAR_PROGRESS:
            return "lidar_progress"
        if cfg.USE_TRACKMAP:
            return "trackmap"
        return "lidar"
    if cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT:
        return "tqc"
    if _USE_IMAGES_MOBILENET_PIPELINE:
        if cfg.USE_IMAGES:
            return "impala"
        return "sophy"
    return "vision"


def _rtgym_interface_partial() -> Any:
    name = _determine_interface_name()
    iface_cls = INTERFACES.get(name)
    alg = algorithm

    env = M.environment
    _n_steer = int(alg.iqn_n_steer_bins) if ALG_NAME in ("IQN", "SDSAC") else 0

    common: dict[str, Any] = {
        "img_hist_len": cfg.IMG_HIST_LEN,
        "gamepad": cfg.USE_VIRTUAL_GAMEPAD,
        # IQN / SDSAC: composite discrete table on the interface so workers map indices → control
        # (trainer already uses algorithm.iqn_n_steer_bins in IQNAgent / memory).
        "discrete_n_steer_bins": _n_steer,
    }

    if name in ("trackmap_images", "lidar_progress_images"):
        common["grayscale"] = cfg.GRAYSCALE
        common["resize_to"] = (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)

    if name in ("impala", "sophy", "tqc", "vision"):
        common["grayscale"] = cfg.GRAYSCALE
        common["resize_to"] = (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)

    if name in ("impala", "sophy", "tqc"):
        common["crash_penalty"] = float(env.crash_penalty)
        common["constant_penalty"] = float(env.constant_penalty)
        common["checkpoint_reward"] = float(env.checkpoint_reward)
        common["lap_reward"] = float(env.lap_reward)
        common["include_camera_images"] = cfg.USE_IMAGES
        common["include_lidar"] = bool(cfg.REWARD_CONFIG.get("RL_INTERFACE_INCLUDE_LIDAR", False))

    return partial(iface_cls, **common)


INTERFACE_NAME = _determine_interface_name()
RTGYM_INTERFACE_CLASS = _rtgym_interface_partial()

logger.info(
    "Interface: registry_key={}, class={}, mixed_precision={}, mixed_precision_dtype={}",
    INTERFACE_NAME,
    RTGYM_INTERFACE_CLASS.func.__name__,
    algorithm.mixed_precision,
    algorithm.mixed_precision_dtype,
)


INTERFACE_DISPLAY_NAME = INTERFACE_NAME

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = RTGYM_INTERFACE_CLASS
CONFIG_DICT_MODIFIERS = M.environment.rtgym_config_dict()
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v


def _pick_sample_compressor() -> Any:
    if cfg.USE_LIDAR_OBSERVATIONS:
        if cfg.USE_LIDAR_PROGRESS_IMAGES or cfg.USE_TRACKMAP_IMAGES:
            return get_local_buffer_sample_lidar_progress_images
        if cfg.USE_LIDAR_PROGRESS:
            return get_local_buffer_sample_lidar_progress
        return get_local_buffer_sample_lidar
    if _USE_NON_LIDAR_IMAGE_STACK:
        return get_local_buffer_sample_mobilenet
    return get_local_buffer_sample_tm20_imgs


SAMPLE_COMPRESSOR = _pick_sample_compressor()


def _pick_obs_preprocessor() -> Any:
    if cfg.USE_LIDAR_OBSERVATIONS:
        if cfg.USE_LIDAR_PROGRESS_IMAGES or cfg.USE_TRACKMAP_IMAGES:
            return obs_preprocessor_lidar_progress_images_act_in_obs
        if cfg.USE_LIDAR_PROGRESS:
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

assert not cfg.USE_RNN, "RNNs not supported yet"


def _determine_memory_name() -> str:
    """Map interface feature flags to a MEMORIES registry key."""
    explicit = M.memory.memory_type
    if explicit != "auto":
        return explicit
    if cfg.USE_LIDAR_OBSERVATIONS:
        if cfg.USE_LIDAR_PROGRESS_IMAGES or cfg.USE_TRACKMAP_IMAGES:
            return "lidar_progress_images"
        if cfg.USE_LIDAR_PROGRESS:
            return "lidar_progress"
        return "lidar"
    if cfg.USE_IMAGES_MOBILENET_PIPELINE:
        return "best"
    if cfg.USE_IMAGES_R2D2_SEQUENCE_BUFFER:
        if cfg.USE_IMAGES and not cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT:
            return "r2d2"
        return "r2d2_wo_images"
    return "full"


_mem_name = _determine_memory_name()
MEM = MEMORIES.get(_mem_name)

_memory_kwargs: dict[str, Any] = {
    "memory_size": cfg.MEMORY_SIZE,
    "batch_size": cfg.BATCH_SIZE,
    "sample_preprocessor": SAMPLE_PREPROCESSOR,
    "dataset_path": cfg_paths.DATASET_PATH,
    "imgs_obs": cfg.IMG_HIST_LEN,
    "act_buf_len": cfg.ACT_BUF_LEN,
    "crc_debug": cfg.CRC_DEBUG,
    "discrete_n_steer_bins": int(algorithm.iqn_n_steer_bins) if ALG_NAME in ("IQN", "SDSAC") else 0,
}

_is_r2d2_memory = _mem_name.startswith("r2d2")
if _is_r2d2_memory:
    _memory_kwargs.update(
        rewards_index=19 if cfg.USE_IMAGES else 18,
        r2d2_rewind=float(cfg.R2D2_REWIND),
        per_td_enabled=bool(cfg.PER_TD_ENABLED),
        per_td_alpha=float(cfg.PER_TD_ALPHA),
        per_td_beta=float(cfg.PER_TD_BETA),
        per_td_eps=float(cfg.PER_TD_EPS),
        r2d2_num_sequences=int(cfg.R2D2_NUM_SEQUENCES),
        r2d2_sequence_length=int(cfg.R2D2_SEQUENCE_LENGTH),
        player_runs_per_alpha=float(cfg.PLAYER_RUNS_PER_ALPHA),
        fog_decay_temperature=float(cfg.FOG_DECAY_TEMPERATURE),
        demo_min_batch_fraction=float(cfg.DEMO_MIN_BATCH_FRACTION),
        demo_max_batch_fraction=float(cfg.DEMO_MAX_BATCH_FRACTION),
    )

MEMORY = partial(MEM, **_memory_kwargs)

logger.info(
    "Memory: registry_key={!r}, class={}, r2d2_params={}",
    _mem_name,
    MEM.__name__,
    _is_r2d2_memory,
)

_device = "cuda" if M.compute.cuda_training else "cpu"
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
        sac_cls = ALGORITHMS.get("SAC")
        _wd = float(alg.weight_decay)
        return partial(
            sac_cls,
            **_common_agent_kw,
            optimizer_actor=alg.optimizer_actor,
            optimizer_critic=alg.optimizer_critic,
            betas_actor=alg.betas_actor,
            betas_critic=alg.betas_critic,
            l2_actor=_wd if _wd > 0.0 else None,
            l2_critic=_wd if _wd > 0.0 else None,
            debug_mode=M.debugger.debug_mode,
            mixed_precision=bool(alg.mixed_precision),
            mixed_precision_dtype=str(alg.mixed_precision_dtype),
            seed=int(M.environment.seed),
        )
    if ALG_NAME == "TQC":
        tqc_cls = ALGORITHMS.get("TQC")
        _wd = float(alg.weight_decay)
        sched = M.training.scheduler
        return partial(
            tqc_cls,
            **_common_agent_kw,
            top_quantiles_to_drop=alg.top_quantiles_to_drop,
            quantiles_number=alg.quantiles_number,
            n_steps=alg.n_steps,
            actor_weight_decay=_wd,
            critic_weight_decay=_wd,
            adam_eps=float(alg.adam_eps),
            betas_actor=alg.betas_actor,
            betas_critic=alg.betas_critic,
            entropy_schedule=str(alg.entropy_schedule),
            entropy_floor=float(alg.entropy_floor),
            entropy_cosine_t0=int(alg.entropy_cosine_t0),
            entropy_cosine_tmult=float(alg.entropy_cosine_tmult),
            entropy_cosine_decay=float(alg.entropy_cosine_decay),
            reward_normalize_scale=float(alg.reward_normalize_scale),
            backup_clip_range=float(alg.backup_clip_range),
            grad_clip_actor=float(alg.grad_clip_actor),
            grad_clip_critic=float(alg.grad_clip_critic),
            weight_clipping_enabled=bool(alg.clipping_weights),
            clip_weights_value=float(alg.clip_weights_value),
            mean_penalty_coef=float(alg.mean_penalty_coef),
            bc_lambda=float(alg.bc_lambda),
            bc_lambda_start=float(alg.bc_lambda_start),
            bc_lambda_end=float(alg.bc_lambda_end),
            bc_anneal_steps_start=int(alg.bc_anneal_steps_start),
            bc_anneal_steps_end=int(alg.bc_anneal_steps_end),
            dynamic_truncation_enabled=bool(alg.dynamic_truncation_enabled),
            dynamic_truncation_variance_pct=float(alg.dynamic_truncation_variance_pct),
            vcse_enabled=bool(alg.vcse_enabled),
            vcse_alpha_base=float(alg.vcse_alpha_base),
            vcse_lambda=float(alg.vcse_lambda),
            r2d2_burn_in=int(alg.r2d2_burn_in),
            r2d2_sequence_length=int(alg.r2d2_sequence_length),
            per_td_enabled=bool(alg.per_td_enabled),
            wandb_debug=M.wandb.debug_reward,
            wandb_gradients=M.wandb.log_gradients,
            scheduler_name=str(sched.name or ""),
            scheduler_t_0=int(sched.t_0),
            scheduler_t_mult=int(sched.t_mult),
            scheduler_eta_min=float(sched.eta_min),
            scheduler_last_epoch=int(sched.last_epoch),
            mixed_precision=bool(alg.mixed_precision),
            mixed_precision_dtype=str(alg.mixed_precision_dtype),
            seed=int(M.environment.seed),
        )
    if ALG_NAME == "REDQSAC":
        redq_cls = ALGORITHMS.get("REDQSAC")
        return partial(
            redq_cls,
            **_common_agent_kw,
            n=alg.redq_n,
            m=alg.redq_m,
            q_updates_per_policy_update=alg.redq_q_updates_per_policy_update,
            weight_decay=float(alg.weight_decay),
            mixed_precision=bool(alg.mixed_precision),
            mixed_precision_dtype=str(alg.mixed_precision_dtype),
            seed=int(M.environment.seed),
        )
    if ALG_NAME == "IQN":
        iqn_cls = ALGORITHMS.get("IQN")
        _iqn_arch_kw = _model_arch_kwargs()
        _rnn_hs = _iqn_arch_kw.get("rnn_hidden_size")
        return partial(
            iqn_cls,
            device=_device,
            hidden_dim=model_cfg.residual_mlp_hidden_dim,
            num_blocks=model_cfg.residual_mlp_num_blocks,
            n_actions=int(alg.iqn_n_actions),
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
            huber_kappa=float(alg.iqn_huber_kappa),
            use_value_rescaling=bool(alg.iqn_use_value_rescaling),
            value_rescaling_eps=float(alg.iqn_value_rescaling_eps),
            soft_target_tau=float(alg.iqn_soft_target_tau),
            log_target_stats=bool(alg.iqn_log_target_stats),
            sort_quantiles=bool(alg.iqn_sort_quantiles),
            monotonicity_regularization=bool(alg.iqn_monotonicity_regularization),
            monotonicity_lambda=float(alg.iqn_monotonicity_lambda),
            munchausen_enabled=bool(alg.iqn_munchausen_enabled),
            munchausen_alpha=float(alg.iqn_munchausen_alpha),
            munchausen_tau=float(alg.iqn_munchausen_tau),
            munchausen_clip_min=float(alg.iqn_munchausen_clip_min),
            munchausen_clip_max=float(alg.iqn_munchausen_clip_max),
            eder_oversample_ratio=int(alg.eder_oversample_ratio),
            weight_decay=float(alg.weight_decay),
            adam_eps=float(alg.adam_eps),
            grad_clip=float(alg.iqn_grad_clip),
            iqn_n_steer_bins=int(alg.iqn_n_steer_bins),
            reward_normalize_scale=float(alg.reward_normalize_scale),
            backup_clip_range=float(alg.backup_clip_range),
            mixed_precision=bool(alg.mixed_precision),
            mixed_precision_dtype=str(alg.mixed_precision_dtype),
            split_track_observation=bool(_iqn_arch_kw["split_track_observation"]),
            track_encoder=str(_iqn_arch_kw["track_encoder"]),
            use_rnn=bool(_iqn_arch_kw["use_rnn"]),
            rnn_hidden_size=int(_rnn_hs) if _rnn_hs is not None else None,
            api_layernorm=bool(_iqn_arch_kw["api_layernorm"]),
            use_simbav2=bool(_iqn_arch_kw["use_simbav2"]),
            r2d2_sequence_length=int(_iqn_arch_kw["r2d2_sequence_length"]),
            r2d2_burn_in=int(_iqn_arch_kw["r2d2_burn_in"]),
            gnn_hidden=int(_iqn_arch_kw["gnn_hidden"]),
            gnn_layers=int(_iqn_arch_kw["gnn_layers"]),
            seed=int(M.environment.seed),
        )
    if ALG_NAME == "SDSAC":
        sdsac_cls = ALGORITHMS.get("SDSAC")
        _nba = model_cfg.residual_mlp_num_blocks_actor
        _nbc = model_cfg.residual_mlp_num_blocks_critic
        _nb = model_cfg.residual_mlp_num_blocks
        return partial(
            sdsac_cls,
            device=_device,
            hidden_dim=model_cfg.residual_mlp_hidden_dim,
            num_blocks_actor=_nba if _nba > 0 else _nb,
            num_blocks_critic=_nbc if _nbc > 0 else _nb,
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
            weight_decay=float(alg.weight_decay),
            reward_normalize_scale=float(alg.reward_normalize_scale),
            r2d2_burn_in=int(alg.r2d2_burn_in),
            r2d2_sequence_length=int(alg.r2d2_sequence_length),
            mixed_precision=bool(alg.mixed_precision),
            mixed_precision_dtype=str(alg.mixed_precision_dtype),
            seed=int(M.environment.seed),
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
    epochs=M.training.max_epochs,
    rounds=M.training.rounds_per_epoch,
    steps=M.training.training_steps_per_round,
    update_model_interval=M.training.update_model_interval,
    update_buffer_interval=M.training.update_buffer_interval,
    max_training_steps_per_env_step=M.training.max_training_steps_per_environment_step,
    python_profiling=M.debugger.profile_trainer,
    pytorch_profiling=M.debugger.pytorch_profiler,
    training_agent_cls=AGENT,
    agent_scheduler=None,
    start_training=M.training.environment_steps_before_training,
)

DUMP_RUN_INSTANCE_FN = None
LOAD_RUN_INSTANCE_FN = None
UPDATER_FN = update_run_instance
