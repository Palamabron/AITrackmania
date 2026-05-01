# third-party imports

import numpy as np

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


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening LIDARs
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), *obs[2:])  # >= 1  action
    return obs


def obs_preprocessor_tm_lidar_progress_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening LIDARs
    """
    obs = (obs[0], obs[1], np.ndarray.flatten(obs[2]), *obs[3:])  # >= 1  action
    return obs


def obs_preprocessor_lidar_progress_images_act_in_obs(obs):
    """
    Preprocessor for LIDAR + images: normalize speed/progress, keep lidar and images as-is.
    Obs = (speed, progress, lidar, images). Images are already [0,1] from interface.
    """
    speed = np.clip(obs[0] / 1000.0, 0.0, 1.0).astype(np.float32)
    progress = np.clip(obs[1], 0.0, 1.0).astype(np.float32)
    return (speed, progress, obs[2], obs[3], *obs[4:])


def obs_preprocessor_mobilenet_act_in_obs(obs):
    return obs


# Legacy default for ``obs_preprocessor_tqcgrab_act_in_obs`` when no divisor is injected.
TRACK_COORDS_SCALE = 100.0


def make_tqcgrab_obs_preprocessor(track_coords_divisor: float):
    """Build TQCGRAB preprocessor with a configurable track geometry scale.

    ``obs[0]`` is divided by ``track_coords_divisor`` and clipped to ``[-1, 1]``.
    Lower divisors amplify signal for small local-frame coordinates; large world-frame
    coordinates may saturate at ``±1``.

    Args:
        track_coords_divisor: Positive scale (typically 40–100; legacy default ``100``).
    """

    divisor = float(track_coords_divisor)
    if divisor <= 0.0:
        raise ValueError("track_coords_divisor must be positive")

    def _preprocess(obs):
        obs = list(obs)
        if len(obs) < 14:
            return tuple(obs)
        track = np.asarray(obs[0], dtype=np.float32)
        if track.size > 0:
            obs[0] = np.clip(track / divisor, -1.0, 1.0).astype(np.float32)
        obs[1] = np.clip(obs[1].astype(np.float32) / 500.0, 0.0, 1.0)
        obs[2] = np.clip(obs[2].astype(np.float32) / 50.0, -1.0, 1.0)
        obs[3] = np.clip(obs[3].astype(np.float32) / 5.0, -1.0, 1.0)
        obs[4] = np.clip(obs[4].astype(np.float32), 0.0, 1.0)
        obs[5] = np.clip(obs[5].astype(np.float32), -1.0, 1.0)
        obs[6] = np.clip(obs[6].astype(np.float32), 0.0, 1.0)
        obs[7] = np.clip(obs[7].astype(np.float32), 0.0, 1.0)
        obs[8] = np.clip(obs[8].astype(np.float32) / 6.0, 0.0, 1.0)
        obs[9] = np.clip(obs[9].astype(np.float32) / np.float32(np.pi), -1.0, 1.0)
        obs[10] = np.clip(obs[10].astype(np.float32) / np.float32(np.pi / 2), -1.0, 1.0)
        obs[11] = np.clip(obs[11].astype(np.float32) / 30.0, -1.0, 1.0)
        obs[12] = np.clip(obs[12].astype(np.float32), 0.0, 1.0)
        obs[13] = np.clip(obs[13].astype(np.float32) / 15.0, 0.0, 1.0)
        return tuple(obs)

    return _preprocess


def obs_preprocessor_tqcgrab_act_in_obs(obs):
    """
    Preprocessor for TQCGRAB (TQC_GrabData plugin): normalize speed and progress to [0,1],
    scale other API channels to bounded ranges for stable SAC/TQC training.
    Track (obs[0]) is normalized to ~[-1, 1] so it matches scale of other inputs.
    Obs = (track, speed, accel, jerk, race_progress, steer, gas, brake, gear, aim_yaw, aim_pitch,
          steer_angle(2), slip_coef(2), failure_counter[, optional action buffer...]).
    """
    return make_tqcgrab_obs_preprocessor(TRACK_COORDS_SCALE)(obs)


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample
# this is useful in particular for data augmentation
# be careful: consistency after this will NOT be checked by CRC


def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, terminated, truncated):
    return last_obs, act, rew, new_obs, terminated, truncated
