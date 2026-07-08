"""Record one or more episodes as player-run files for later replay import."""

from __future__ import annotations

import itertools
from pathlib import Path

import gymnasium
import numpy as np
from loguru import logger

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.custom.tm.utils.control_keyboard import is_del_pressed
from tmrl.envs import GenericGymEnv
from tmrl.networking.buffer import Buffer
from tmrl.tools.recording.player_runs import align_observation_to_space, save_player_run
from tmrl.util import partial


def _extract_human_action_from_obs_tqcgrab(obs):
    """Extract [gas, brake, steer] from TQCGRAB observation (indices 5,6,7 = steer, gas, brake)."""
    gas = float(np.asarray(obs[6]).flat[0])
    brake = float(np.asarray(obs[7]).flat[0])
    steer = float(np.asarray(obs[5]).flat[0])
    return np.array([gas, brake, steer], dtype=np.float32)


def _extract_human_action_from_record_info(info, *, fallback: np.ndarray) -> np.ndarray:
    """Boundary lidar path: interfaces add ``human_control_vec`` when ``record_human``."""
    if isinstance(info, dict):
        vec = info.get("human_control_vec")
        if vec is not None:
            a = np.asarray(vec, dtype=np.float32).ravel()
            if a.size >= 3:
                return np.array([float(a[0]), float(a[1]), float(a[2])], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32).copy()


def _collect_human_episode(env, max_samples, obs_preprocessor, crc_debug):
    """Collect one episode using human control (neutral sent so human drives) and Del to end."""
    neutral_action = np.zeros(3, dtype=np.float32)
    buffer_memory = []
    ret = 0.0
    steps = 0

    obs, info = env.reset()
    if obs_preprocessor is not None:
        obs = obs_preprocessor(obs)

    iterator = range(max_samples) if max_samples != np.inf else itertools.count()
    for i in iterator:
        new_obs, rew, terminated, truncated, info = env.step(neutral_action)
        if obs_preprocessor is not None:
            new_obs = obs_preprocessor(new_obs)

        if i == max_samples - 1 and not terminated:
            truncated = True

        if cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT:
            act_for_sample = _extract_human_action_from_obs_tqcgrab(new_obs)
        elif cfg.USE_LIDAR or cfg.USE_LIDAR_IMAGES:
            act_for_sample = _extract_human_action_from_record_info(info, fallback=neutral_action)
        else:
            raise NotImplementedError(
                "Human recording is only implemented for TM20LIDAR / TM20TRACKMAP / "
                "TM20TRACKMAPIMAGES / TM20LIDARIMAGES layouts "
                "or TQCGRAB-style layouts."
            )
        if is_del_pressed():
            truncated = True

        if crc_debug:
            info = dict(info)
            info["crc_sample"] = (obs, act_for_sample, new_obs, rew, terminated, truncated)
            info["crc_sample_ts"] = (0, steps)
        info_stored = dict(info) if isinstance(info, dict) else {}
        info_stored.pop("human_control_vec", None)
        sample = (act_for_sample, new_obs, rew, terminated, truncated, info_stored)
        buffer_memory.append(sample)

        ret += rew
        steps += 1
        obs = new_obs

        if terminated or truncated:
            break

    return buffer_memory, ret, steps


def _rewrite_discrete_action_slots(samples: list, act_buf_len: int) -> list:
    """Replace recorded obs action-buffer slots with the human's quantized actions.

    During recording the env receives a neutral placeholder action, so rtgym's
    ``act_in_obs`` slots would claim the previous action was the placeholder
    (later trimmed by space alignment to index 0 = full-left). Rewriting them
    with the quantized human controls keeps demo observations consistent with
    worker rollouts, scaled like ``obs_preprocessor_lidar_act_in_obs`` scales
    live action slots.
    """
    if act_buf_len <= 0 or not samples:
        return samples

    from tmrl.custom.tm.tm_preprocessors import discrete_action_index_scale
    from tmrl.custom.tm.utils.discrete_control import (
        build_brake_tap_action_table,
        continuous_control_to_discrete_index,
    )

    _, table = build_brake_tap_action_table(n_steer=cfg.IQN_N_STEER_BINS)
    scale = discrete_action_index_scale()
    indices = [
        continuous_control_to_discrete_index(np.asarray(s[0], dtype=np.float32), table)
        for s in samples
    ]
    out = []
    for i, s in enumerate(samples):
        obs = list(s[1])
        if len(obs) <= act_buf_len:
            out.append(s)
            continue
        # rtgym act_buf is oldest -> newest; the last slot is the action that
        # produced this observation. The first sample has no predecessor, so its
        # own action fills the older slots.
        window = [indices[max(0, i - k)] for k in range(act_buf_len - 1, -1, -1)]
        for slot, idx in zip(range(-act_buf_len, 0), window, strict=True):
            obs[slot] = np.asarray(float(idx) * scale, dtype=np.float32)
        out.append((s[0], tuple(obs), s[2], s[3], s[4], s[5]))
    return out


def _maybe_apply_finish_time_bonus(samples: list) -> tuple[list, float]:
    """Align with ``RolloutWorker.collect_train_episode``: spread ``time_bonus_scale`` on finish."""
    if not samples:
        return samples, 0.0
    last_info = samples[-1][5]
    if isinstance(last_info, dict) and last_info.get("end_of_track", False):
        time_bonus_scale = float(cfg.REWARD_CONFIG.get("time_bonus_scale", 0.0))
        reward_scale = float(cfg.REWARD_CONFIG.get("reward_scale", 1.0))
        if time_bonus_scale > 0 and reward_scale > 0:
            buf = Buffer()
            buf.memory = list(samples)
            buf.apply_speed_bonus(time_bonus_scale * reward_scale)
            samples = buf.memory
    ep_return = float(sum(float(s[2]) for s in samples))
    return samples, ep_return


def record_episode(
    *,
    nb_episodes: int = 1,
    output_dir: str | None = None,
    max_samples_per_episode: int | None = None,
    save_replays: bool = False,
) -> list[Path]:
    """Collect episodes and save them as standalone player-run files.

    Uses human control (no model): sends neutral (0,0,0) so you drive with a physical
    gamepad. Press Del to end the current episode early.
    """
    if nb_episodes <= 0:
        raise ValueError("nb_episodes must be > 0")

    if not cfg.USE_OBS_WORLD_TELEMETRY_LAYOUT and not cfg.USE_LIDAR and not cfg.USE_LIDAR_IMAGES:
        raise NotImplementedError(
            "Human recording needs environment.rtgym_interface boundary lidar tokens "
            "(TM20LIDAR / *TRACKMAP*) or fused lidar+images (*TRACKMAPIMAGES*, *LIDARIMAGES*), "
            "or a TQCGRAB* token (world-telemetry layout)."
        )

    env_config = cfg_obj.CONFIG_DICT.copy()
    interface_kwargs = dict(env_config.get("interface_kwargs") or {})
    interface_kwargs["record_human"] = True
    if save_replays:
        interface_kwargs["save_replays"] = True
    env_config["interface_kwargs"] = interface_kwargs
    # Ensure record_human reaches the interface (rtgym may not merge interface_kwargs)
    _int = env_config["interface"]
    if hasattr(_int, "func") and hasattr(_int, "keywords"):
        env_config["interface"] = partial(_int.func, record_human=True, **_int.keywords)

    max_samples = (
        int(max_samples_per_episode)
        if max_samples_per_episode is not None
        else cfg.RW_MAX_SAMPLES_PER_EPISODE
    )

    env_cls = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": env_config})

    saved_paths: list[Path] = []
    with env_cls() as env:
        for ep in range(nb_episodes):
            input(
                f"Press Enter to start episode {ep + 1}/{nb_episodes} "
                "(be IN MAP with car on track) ... "
            )
            logger.info(
                "Recording episode {}/{} (human control; press Del to end) ...",
                ep + 1,
                nb_episodes,
            )
            samples, _ep_ret, ep_steps = _collect_human_episode(
                env,
                max_samples=max_samples,
                obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                crc_debug=cfg.CRC_DEBUG,
            )
            samples, ep_return = _maybe_apply_finish_time_bonus(samples)
            if isinstance(env.action_space, gymnasium.spaces.Discrete):
                samples = _rewrite_discrete_action_slots(samples, int(cfg.ACT_BUF_LEN))

            obs_space = env.observation_space
            samples = [
                (
                    s[0],
                    align_observation_to_space(s[1], obs_space),
                    s[2],
                    s[3],
                    s[4],
                    s[5],
                )
                for s in samples
            ]

            metadata = {
                "episode_index": ep,
                "episode_return": float(ep_return),
                "episode_steps": int(ep_steps),
                "map_name": cfg.MAP_NAME,
                "run_name": cfg.RUN_NAME,
                "memory_class": cfg_obj.MEMORY.func.__name__
                if hasattr(cfg_obj.MEMORY, "func")
                else "unknown",
            }
            out_path = save_player_run(samples, output_dir=output_dir, metadata=metadata)
            saved_paths.append(out_path)
            logger.info(
                "Saved {} samples to '{}'. return={} steps={}",
                len(samples),
                out_path,
                metadata["episode_return"],
                metadata["episode_steps"],
            )

    logger.info("Recorded {} episode file(s).", len(saved_paths))
    return saved_paths
