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


def obs_preprocessor_lidar_act_in_obs(obs):
    """Boundary lidar obs: (track_60, speed, gear, rpm, accel, steer, slip x4, crash, fc)."""
    return (
        np.clip(obs[0].astype(np.float32) / 300.0, -1.0, 1.0),
        np.clip(obs[1].astype(np.float32) / 1000.0, 0.0, 1.0),
        np.clip(obs[2].astype(np.float32) / 6.0, 0.0, 1.0),
        np.clip(obs[3].astype(np.float32) / 10000.0, 0.0, 1.0),
        np.clip(obs[4].astype(np.float32) / 100.0, -1.0, 1.0),
        np.clip(obs[5].astype(np.float32), -1.0, 1.0),
        np.clip(obs[6].astype(np.float32), 0.0, 1.0),
        obs[7].astype(np.float32),
        np.clip(obs[8].astype(np.float32) / 15.0, 0.0, 1.0),
        *obs[9:],
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
        obs[_Obs.FAILURE_COUNTER] = np.clip(
            obs[_Obs.FAILURE_COUNTER].astype(np.float32) / 15.0, 0.0, 1.0
        )
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
