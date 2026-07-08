"""Sample compressors for local buffer storage.

These functions compress observations before storing them in local buffers
for network transmission, reducing bandwidth requirements.

Buffer sample order: (prev_act, obs, rew, terminated, truncated, info)
where prev_act is the action that yielded obs (i.e. prev_obs -> prev_act -> obs).
"""

import numpy as np


def _compress_last_obs_image_to_uint8(obs):
    """Compress image tensor in the last obs slot to uint8 when present."""
    if not isinstance(obs, (tuple, list)) or len(obs) == 0:
        return obs
    last = obs[-1]
    arr = np.asarray(last)
    if arr.ndim < 2:
        return obs
    if arr.dtype == np.uint8:
        img_u8 = arr
    else:
        # Some interfaces produce float32 in [0, 1], others [0, 255].
        max_val = float(np.max(arr)) if arr.size > 0 else 0.0
        if max_val <= 1.5:
            arr = arr * 255.0
        img_u8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    obs_mod = list(obs)
    obs_mod[-1] = img_u8
    return tuple(obs_mod)


def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):
    """No compression; boundary lidar observations are already compact tuples of ndarrays."""
    return prev_act, obs, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_lidar_images(prev_act, obs, rew, terminated, truncated, info):
    """Compress boundary lidar + images: float32 reward; uint8 image stack."""
    obs_mod = _compress_last_obs_image_to_uint8(obs)
    return prev_act, obs_mod, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_mobilenet(prev_act, obs, rew, terminated, truncated, info):
    """Compress for MobileNet interface: cast reward to float32."""
    obs_mod = _compress_last_obs_image_to_uint8(obs)
    return prev_act, obs_mod, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):
    """Compress for full TM2020 image interface: quantize images to uint8."""
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))
    return prev_act, obs_mod, rew, terminated, truncated, info
