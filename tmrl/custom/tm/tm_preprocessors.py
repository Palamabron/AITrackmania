import numpy as np

from tmrl.custom.tm.observation_constants import WorldTelemetryObsIndex as _Obs

# OBSERVATION PREPROCESSING ==================================


def obs_preprocessor_tm_act_in_obs(obs):
    """
    Preprocessor for TM2020 full environment with grayscale images
    """
    grayscale_images = obs[3]
    grayscale_images = grayscale_images.astype(np.float32) / 255.0
    obs = (
        obs[0] / 1000.0,
        obs[1] / 10.0,
        obs[2] / 10000.0,
        grayscale_images,
        *obs[4:],
    )  # >= 1 action
    return obs


def discrete_action_index_scale() -> float:
    """``1 / (n_actions - 1)`` for scaling rtgym action-buffer indices into [0, 1].

    Reads the configured IQN action count on each call (avoids stale module-global
    cache across config reloads); falls back to the default 78-action brake-tap table.
    """
    try:
        import tmrl.config.constants as _cfg

        n_actions = int(getattr(_cfg, "IQN_N_ACTIONS", 0))
    except Exception:
        n_actions = 0
    if n_actions <= 1:
        n_actions = 78
    return 1.0 / float(n_actions - 1)


def _scale_action_tail(parts):
    """Scale rtgym act_in_obs slots: 0-d integer indices -> [0, 1] float32.

    Continuous action vectors (e.g. SAC ``(3,)`` controls already in [-1, 1])
    pass through unchanged, so raw 0..n_actions-1 indices stop dwarfing the
    normalized physics channels.
    """
    out = []
    for p in parts:
        a = np.asarray(p)
        if a.ndim == 0 and np.issubdtype(a.dtype, np.integer):
            out.append(np.asarray(float(a) * discrete_action_index_scale(), dtype=np.float32))
        else:
            out.append(p)
    return out


def obs_preprocessor_lidar_act_in_obs(obs):
    """Boundary lidar obs: (track_60, speed, gear, rpm, accel, steer, slip x4, crash, fc).

    The failure counter (obs[8]) is already normalized to [0, 1] by the interface
    (fraction of the no-progress timeout consumed), so it is only clipped here.
    """
    return (
        np.clip(obs[0].astype(np.float32) / 300.0, -1.0, 1.0),
        np.clip(obs[1].astype(np.float32) / 1000.0, 0.0, 1.0),
        np.clip(obs[2].astype(np.float32) / 6.0, 0.0, 1.0),
        np.clip(obs[3].astype(np.float32) / 10000.0, 0.0, 1.0),
        np.clip(obs[4].astype(np.float32) / 100.0, -1.0, 1.0),
        np.clip(obs[5].astype(np.float32), -1.0, 1.0),
        np.clip(obs[6].astype(np.float32), 0.0, 1.0),
        obs[7].astype(np.float32),
        np.clip(obs[8].astype(np.float32), 0.0, 1.0),
        *_scale_action_tail(obs[9:]),
    )


def obs_preprocessor_lidar_images_act_in_obs(obs):
    """Boundary lidar + camera: normalize speed/progress; keep track geometry and images as-is.

    Obs = (speed, progress, track, images). Images are already [0, 1] from the interface.
    """
    speed = np.clip(obs[0] / 1000.0, 0.0, 1.0).astype(np.float32)
    progress = np.clip(obs[1], 0.0, 1.0).astype(np.float32)
    return (speed, progress, obs[2], obs[3], *obs[4:])


def obs_preprocessor_mobilenet_act_in_obs(obs):
    return obs


# Legacy default for ``obs_preprocessor_world_telemetry_act_in_obs`` when no divisor is injected.
TRACK_COORDS_SCALE = 100.0


def make_world_telemetry_obs_preprocessor(track_coords_divisor: float):
    """Build world-telemetry preprocessor with a configurable track geometry scale.

    ``obs[0]`` is divided by ``track_coords_divisor`` and clipped to ``[-1, 1]``.
    Lower divisors amplify signal for small local-frame coordinates; large world-frame
    coordinates may saturate at ``±1``.

    Args:
        track_coords_divisor: Positive scale (typically 40-100; legacy default ``100``).
    """

    divisor = float(track_coords_divisor)
    if divisor <= 0.0:
        raise ValueError("track_coords_divisor must be positive")

    def _preprocess(obs):
        obs = list(obs)
        if len(obs) < len(_Obs):
            return tuple(obs)
        track = np.asarray(obs[_Obs.TRACK_INFO], dtype=np.float32)
        if track.size > 0:
            obs[_Obs.TRACK_INFO] = np.clip(track / divisor, -1.0, 1.0).astype(np.float32)
        obs[_Obs.SPEED] = np.clip(obs[_Obs.SPEED].astype(np.float32) / 500.0, 0.0, 1.0)
        obs[_Obs.ACCELERATION] = np.clip(
            obs[_Obs.ACCELERATION].astype(np.float32) / 50.0, -1.0, 1.0
        )
        obs[_Obs.JERK] = np.clip(obs[_Obs.JERK].astype(np.float32) / 5.0, -1.0, 1.0)
        obs[_Obs.RACE_PROGRESS] = np.clip(obs[_Obs.RACE_PROGRESS].astype(np.float32), 0.0, 1.0)
        obs[_Obs.INPUT_STEER] = np.clip(obs[_Obs.INPUT_STEER].astype(np.float32), -1.0, 1.0)
        obs[_Obs.INPUT_GAS_PEDAL] = np.clip(obs[_Obs.INPUT_GAS_PEDAL].astype(np.float32), 0.0, 1.0)
        obs[_Obs.INPUT_BRAKE] = np.clip(obs[_Obs.INPUT_BRAKE].astype(np.float32), 0.0, 1.0)
        obs[_Obs.GEAR] = np.clip(obs[_Obs.GEAR].astype(np.float32) / 6.0, 0.0, 1.0)
        obs[_Obs.AIM_YAW] = np.clip(
            obs[_Obs.AIM_YAW].astype(np.float32) / np.float32(np.pi), -1.0, 1.0
        )
        obs[_Obs.AIM_PITCH] = np.clip(
            obs[_Obs.AIM_PITCH].astype(np.float32) / np.float32(np.pi / 2), -1.0, 1.0
        )
        obs[_Obs.STEER_ANGLE] = np.clip(obs[_Obs.STEER_ANGLE].astype(np.float32) / 30.0, -1.0, 1.0)
        obs[_Obs.SLIP_COEF] = np.clip(obs[_Obs.SLIP_COEF].astype(np.float32), 0.0, 1.0)
        # The interface already normalizes the failure counter by the no-progress
        # timeout horizon (car_state.py: fc / max_no_progress_steps in [0, 1]);
        # dividing by 15 again would crush the signal to ~[0, 0.07].
        obs[_Obs.FAILURE_COUNTER] = np.clip(obs[_Obs.FAILURE_COUNTER].astype(np.float32), 0.0, 1.0)
        return tuple(obs)

    return _preprocess


def obs_preprocessor_world_telemetry_act_in_obs(obs):
    """
    Preprocessor for world-telemetry interface (TQC_GrabData plugin): normalize speed and
    progress to [0,1], scale other API channels to bounded ranges for stable SAC/TQC training.
    Track (obs[0]) is normalized to ~[-1, 1] so it matches scale of other inputs.
    Obs = (track, speed, accel, jerk, race_progress, steer, gas, brake, gear, aim_yaw, aim_pitch,
          steer_angle(2), slip_coef(2), failure_counter[, optional action buffer...]).
    """
    return make_world_telemetry_obs_preprocessor(TRACK_COORDS_SCALE)(obs)


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample
# this is useful in particular for data augmentation
# be careful: consistency after this will NOT be checked by CRC


def sample_preprocessor_lidar_act_in_obs(last_obs, act, rew, new_obs, terminated, truncated):
    return last_obs, act, rew, new_obs, terminated, truncated
