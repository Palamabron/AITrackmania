"""Compose Hydra YAML, apply optional overrides/secrets, validate with Pydantic."""

from __future__ import annotations

import copy
import json
import os
import platform
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from packaging import version

from tmrl.config.schema.main import MainConfig

MINIMUM_SCHEMA_VERSION = "0.6.0"
CONFIG_UPGRADE_HINT = (
    "Update ~/TmrlData/config/local.yaml to match the current schema (snake_case keys). "
    "See tmrl/config/defaults/config.yaml in the package."
)

SYSTEM = platform.system()
RTGYM_VERSION = "real-time-gym-v1" if SYSTEM == "Windows" else "real-time-gym-ts-v1"

TMRL_FOLDER = Path.home() / "TmrlData"
if not TMRL_FOLDER.exists():
    raise RuntimeError(f"Missing folder: {TMRL_FOLDER}")

CONFIG_DIR = TMRL_FOLDER / "config"
LOCAL_OVERRIDE_PATH = CONFIG_DIR / "local.yaml"
HYDRA_OVERRIDES_ENV = "TMRL_HYDRA_OVERRIDES"

load_dotenv()
load_dotenv(TMRL_FOLDER / ".env")
load_dotenv(CONFIG_DIR / ".env")

_HYDRA_CONF_DIR = Path(__file__).resolve().parent / "defaults"


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, val in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(val, dict):
            _deep_merge(cast(dict[str, Any], dst[key]), val)
        else:
            dst[key] = val


def _compose_hydra_dict() -> dict[str, Any]:
    hydra_overrides_raw = os.environ.get(HYDRA_OVERRIDES_ENV, "").strip()
    hydra_overrides: list[str] = []
    if hydra_overrides_raw:
        try:
            parsed = json.loads(hydra_overrides_raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            if not all(isinstance(item, str) for item in parsed):
                raise TypeError(f"{HYDRA_OVERRIDES_ENV} JSON list must contain only strings")
            hydra_overrides = [item.strip() for item in parsed if item.strip()]
        else:
            hydra_overrides = [
                item.strip() for item in hydra_overrides_raw.split(",") if item.strip()
            ]
    if not _HYDRA_CONF_DIR.is_dir():
        raise RuntimeError(f"Missing Hydra config directory: {_HYDRA_CONF_DIR}")
    with initialize_config_dir(version_base=None, config_dir=str(_HYDRA_CONF_DIR)):
        hydra_cfg = compose(config_name="config", overrides=hydra_overrides)
    if hydra_overrides:
        logger.info(
            "Applied {} Hydra override(s) from {}", len(hydra_overrides), HYDRA_OVERRIDES_ENV
        )
    out = OmegaConf.to_container(hydra_cfg, resolve=True)
    if not isinstance(out, dict):
        raise TypeError("Hydra root must be a mapping")
    return cast(dict[str, Any], out)


def _load_local_overrides() -> dict[str, Any] | None:
    if not LOCAL_OVERRIDE_PATH.is_file():
        return None
    with open(LOCAL_OVERRIDE_PATH) as f:
        data = yaml.safe_load(f)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError(f"{LOCAL_OVERRIDE_PATH} must contain a YAML mapping at the root")
    return cast(dict[str, Any], data)


def _apply_env_secrets(cfg: dict[str, Any]) -> None:
    wandb_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY")
    if wandb_key:
        wandb = cfg.setdefault("wandb", {})
        if isinstance(wandb, dict):
            wandb["api_key"] = wandb_key
    password = os.getenv("TMRL_PASSWORD")
    if password:
        dist = cfg.setdefault("distributed", {})
        if isinstance(dist, dict):
            dist["password"] = password


def _build_raw_config() -> dict[str, Any]:
    merged = _compose_hydra_dict()
    local = _load_local_overrides()
    if local:
        _deep_merge(merged, local)
        logger.info("Loaded user overrides from {}", LOCAL_OVERRIDE_PATH)
    _apply_env_secrets(merged)

    overrides = os.environ.get("TMRL_CONFIG_OVERRIDES", "").strip()
    if overrides:
        try:
            patch = json.loads(overrides)
        except json.JSONDecodeError as e:
            raise ValueError(
                "TMRL_CONFIG_OVERRIDES must be a JSON object, e.g. "
                '\'{"algorithm":{"lr_actor":3e-5}}\''
            ) from e
        if not isinstance(patch, dict):
            raise TypeError("TMRL_CONFIG_OVERRIDES must be a JSON object at the root")
        _deep_merge(merged, patch)
        logger.info("Applied TMRL_CONFIG_OVERRIDES ({} top-level keys)", len(patch))

    if "schema_version" not in merged:
        raise ValueError("Missing schema_version. " + CONFIG_UPGRADE_HINT)
    if version.parse(merged["schema_version"]) < version.parse(MINIMUM_SCHEMA_VERSION):
        raise ValueError(
            f"schema_version {merged['schema_version']} is below minimum {MINIMUM_SCHEMA_VERSION}. "
            + CONFIG_UPGRADE_HINT
        )
    return merged


_RAW_CONFIG = _build_raw_config()
MAIN_CONFIG = MainConfig.model_validate(_RAW_CONFIG)
CONFIG_VERSION = MAIN_CONFIG.schema_version


def _log_resolved_config() -> None:
    """Log the full resolved config (secrets redacted) before objects are built."""
    d = MAIN_CONFIG.model_dump(mode="json")
    for section_key in ("wandb", "distributed"):
        section = d.get(section_key)
        if isinstance(section, dict):
            for secret in ("api_key", "password"):
                if section.get(secret):
                    section[secret] = "<redacted>"
    logger.info(
        "Resolved TMRL configuration (schema_version={}):\n{}",
        CONFIG_VERSION,
        yaml.safe_dump(d, sort_keys=False, default_flow_style=False),
    )


_log_resolved_config()


def merged_config_snapshot_redacted() -> dict[str, Any]:
    """Post-merge config dict (Hydra + local.yaml + env), secrets stripped for archiving."""
    out = copy.deepcopy(_RAW_CONFIG)
    wandb = out.get("wandb")
    if isinstance(wandb, dict) and wandb.get("api_key"):
        wandb["api_key"] = "<redacted>"
    dist = out.get("distributed")
    if isinstance(dist, dict) and dist.get("password"):
        dist["password"] = "<redacted>"
    return out


def main_config_snapshot_redacted() -> dict[str, Any]:
    """Validated :class:`~tmrl.config.schema.main.MainConfig` as JSON-friendly dict.

    Secrets are redacted in the returned dict.
    """
    d = MAIN_CONFIG.model_dump(mode="json")
    w = d.get("wandb")
    if isinstance(w, dict) and w.get("api_key"):
        w = dict(w)
        w["api_key"] = "<redacted>"
        d["wandb"] = w
    dist = d.get("distributed")
    if isinstance(dist, dict) and dist.get("password"):
        dist = dict(dist)
        dist["password"] = "<redacted>"
        d["distributed"] = dist
    return d


MINIMUM_CONFIG_VERSION = MINIMUM_SCHEMA_VERSION
CONFIG_COMPATIBILITY_ERROR_MESSAGE = CONFIG_UPGRADE_HINT
CONFIG_FILE_PATH = LOCAL_OVERRIDE_PATH
DEBUGGER = MAIN_CONFIG.debugger
DEBUGGER_CONFIG = MAIN_CONFIG.debugger.model_dump()


def create_config() -> dict[str, Any]:
    """Flat snake_case dict for W&B logging and legacy helpers expecting one-level keys."""
    from tmrl.config.constants import (
        POINTS_NUMBER,
        REWARD_CONFIG,
    )

    m = MAIN_CONFIG
    a = m.algorithm
    t = m.training
    r = m.model
    e = m.environment
    sched = t.scheduler

    flat: dict[str, Any] = {
        "training_steps_per_round": t.training_steps_per_round,
        "max_training_steps_per_environment_step": t.max_training_steps_per_environment_step,
        "environment_steps_before_training": t.environment_steps_before_training,
        "update_model_interval": t.update_model_interval,
        "update_buffer_interval": t.update_buffer_interval,
        "save_model_every": t.save_model_every,
        "memory_size": t.memory_size,
        "batch_size": t.batch_size,
        "cnn_filters": list(r.cnn_filters),
        "cnn_output_size": r.cnn_output_size,
        "rnn_sizes": list(r.rnn_sizes),
        "rnn_lens": list(r.rnn_lens),
        "api_mlp_sizes": list(r.api_mlp_sizes),
        "api_layernorm": r.api_layernorm,
        "noisy_linear_actor": r.noisy_linear_actor,
        "noisy_linear_critic": r.noisy_linear_critic,
        "rnn_dropout": r.rnn_dropout,
        "use_residual_mlp": r.use_residual_mlp,
        "residual_mlp_hidden_dim": r.residual_mlp_hidden_dim,
        "residual_mlp_num_blocks": r.residual_mlp_num_blocks,
        "residual_mlp_num_blocks_actor": r.residual_mlp_num_blocks_actor,
        "residual_mlp_num_blocks_critic": r.residual_mlp_num_blocks_critic,
        "use_sophy_residual_actor": r.use_sophy_residual_actor,
        "split_track_observation": r.split_track_observation,
        "use_simbav2": r.use_simbav2,
        "track_encoder": r.track_encoder,
        "gnn_layers": r.gnn_layers,
        "gnn_hidden": r.gnn_hidden,
        "binary_brake": r.binary_brake,
        "use_rnn": r.use_rnn,
        "rnn_hidden_size": r.rnn_hidden_size,
        "use_efficientnet": r.use_efficientnet,
        "use_frozen_effnet": r.use_frozen_effnet,
        "frozen_effnet_embed_dim": r.frozen_effnet_embed_dim,
        "frozen_effnet_width_mult": r.frozen_effnet_width_mult,
        "frozen_effnet_variant": r.frozen_effnet_variant,
        "frozen_effnet_use_dw_stem": r.frozen_effnet_use_dw_stem,
        "min_zero_reward_steps_before_failure": e.min_zero_reward_steps_before_failure,
        "max_zero_reward_steps_before_failure": e.max_zero_reward_steps_before_failure,
        "min_seconds_before_failure": float(REWARD_CONFIG["min_seconds_before_failure"]),
        "off_track_seconds_before_failure": float(
            REWARD_CONFIG["off_track_seconds_before_failure"]
        ),
        "oscillation_period": e.oscillation_period,
        "crash_penalty": e.crash_penalty,
        "crash_cooldown": e.crash_cooldown,
        "constant_penalty": e.constant_penalty,
        "lap_reward": e.lap_reward,
        "lap_cooldown": e.lap_cooldown,
        "checkpoint_reward": e.checkpoint_reward,
        "checkpoint_cooldown": 0,
        "reward_end_of_track": e.end_of_track_reward,
        "algorithm": a.name,
        "quantiles_number": a.quantiles_number,
        "learn_entropy_coef": a.learn_entropy_coef,
        "lr_actor": a.lr_actor,
        "lr_critic": a.lr_critic,
        "lr_critic_divided_by_lr_actor": a.lr_critic / a.lr_actor if a.lr_actor else 0.0,
        "n_steps": a.n_steps,
        "weight_decay": a.weight_decay,
        "actor_weight_decay": a.weight_decay,
        "critic_weight_decay": a.weight_decay,
        "clipping_weights": a.clipping_weights,
        "clip_weights_value": 1.0 if not a.clipping_weights else a.clip_weights_value,
        "points_number": POINTS_NUMBER,
        "points_distance": a.points_distance,
        "speed_bonus": a.speed_bonus,
        "speed_min_threshold": a.speed_min_threshold,
        "speed_medium_threshold": a.speed_medium_threshold,
        "lr_entropy": a.lr_entropy,
        "gamma": a.gamma,
        "polyak": a.polyak,
        "target_entropy": a.target_entropy,
        "top_quantiles_to_drop": a.top_quantiles_to_drop,
        "bc_lambda": float(a.bc_lambda),
        "bc_lambda_start": float(a.bc_lambda_start),
        "bc_lambda_end": float(a.bc_lambda_end),
        "bc_anneal_steps_start": a.bc_anneal_steps_start,
        "bc_anneal_steps_end": a.bc_anneal_steps_end,
        "r2d2_rewind": a.r2d2_rewind,
        "r2d2_num_sequences": a.r2d2_num_sequences,
        "r2d2_sequence_length": a.r2d2_sequence_length,
        "r2d2_burn_in": a.r2d2_burn_in,
        "adam_eps": a.adam_eps,
        "scheduler_t_0": sched.t_0,
        "scheduler_t_mult": sched.t_mult,
        "scheduler_eta_min": sched.eta_min,
        "scheduler_last_epoch": sched.last_epoch,
        "img_width": e.img_width,
        "img_height": e.img_height,
        "img_grayscale": e.img_grayscale,
        "img_hist_len": e.img_hist_len,
    }
    for i, f in enumerate(flat["cnn_filters"]):
        flat[f"cnn_filter{i}"] = f
    for i, s in enumerate(flat["rnn_sizes"]):
        flat[f"rnn_size{i}"] = s
    for i, ln in enumerate(flat["rnn_lens"]):
        flat[f"rnn_len{i}"] = ln
    for i, s in enumerate(flat["api_mlp_sizes"]):
        flat[f"api_mlp_size{i}"] = s
    return flat
