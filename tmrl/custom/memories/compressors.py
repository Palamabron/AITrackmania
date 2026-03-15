"""Sample compressors for local buffer storage.

These functions compress observations before storing them in local buffers
for network transmission, reducing bandwidth requirements.

Note:
    In the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs)),
    i.e., the observation is what step returned after being fed act.
    CAUTION: prev_act is the action that comes BEFORE obs
    (i.e., prev_obs, prev_act(prev_obs), obs(prev_act)).
"""

import numpy as np


def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):
    """Sample compressor for LIDAR interface.

    Compresses observations by keeping only speed and most recent LIDAR readings.

    Args:
        prev_act: Action from the previous observation.
        obs: Current observation tuple (speed, lidar_history).
        rew: Reward received.
        terminated: Whether the episode terminated.
        truncated: Whether the episode was truncated.
        info: Additional information dictionary.

    Returns:
        Tuple of (prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info).
    """
    obs_mod = (obs[0], obs[1][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    return prev_act, obs_mod, rew_mod, terminated, truncated, info


def get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info):
    """Sample compressor for LIDAR + progress interface.

    Args:
        prev_act: Action from the previous observation.
        obs: Current observation tuple (speed, progress, lidar_history).
        rew: Reward received.
        terminated: Whether the episode terminated.
        truncated: Whether the episode was truncated.
        info: Additional information dictionary.

    Returns:
        Tuple of (prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info).
    """
    obs_mod = (obs[0], obs[1], obs[2][-19:])  # speed, progress, and most recent LIDAR
    rew_mod = np.float32(rew)
    return prev_act, obs_mod, rew_mod, terminated, truncated, info


def get_local_buffer_sample_lidar_progress_images(prev_act, obs, rew, terminated, truncated, info):
    """Sample compressor for LIDAR + images interface.

    Args:
        prev_act: Action from the previous observation.
        obs: Current observation tuple (speed, progress, lidar, images).
        rew: Reward received.
        terminated: Whether the episode terminated.
        truncated: Whether the episode was truncated.
        info: Additional information dictionary.

    Returns:
        Tuple of (prev_act, obs, rew_mod, terminated, truncated, info).
    """
    rew_mod = np.float32(rew)
    return prev_act, obs, rew_mod, terminated, truncated, info


def get_local_buffer_sample_mobilenet(prev_act, obs, rew, terminated, truncated, info):
    """Sample compressor for MobileNet-based interfaces.

    Converts reward to float32 for consistent storage.

    Args:
        prev_act: Action from the previous observation.
        obs: Current observation.
        rew: Reward received.
        terminated: Whether the episode terminated.
        truncated: Whether the episode was truncated.
        info: Additional information dictionary.

    Returns:
        Tuple of (prev_act, obs, rew_mod, terminated, truncated, info).
    """
    rew_mod = np.float32(rew)
    return prev_act, obs, rew_mod, terminated, truncated, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):
    """Sample compressor for full image-based TM2020 interface.

    Converts images to uint8 for compression (scaling by 256).

    Args:
        prev_act: Action from the previous observation.
        obs: Current observation tuple (speed, gear, rpm, images).
        rew: Reward received.
        terminated: Whether the episode terminated.
        truncated: Whether the episode was truncated.
        info: Additional information dictionary.

    Returns:
        Tuple of (prev_act, obs_mod, rew, terminated, truncated, info).
    """
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))
    return prev_act, obs_mod, rew, terminated, truncated, info
