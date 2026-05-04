"""Observation space helpers for training_offline."""

from typing import cast

import gymnasium
import numpy as np
import torch

import tmrl.config.constants as cfg


def _observation_space_from_sample(observation) -> gymnasium.spaces.Space:
    """Build a gymnasium observation space from a single observation (e.g. tuple of arrays).

    Use this when the replay buffer already has data so the model is built with the same
    observation shape as the data (avoids LayerNorm / backbone shape mismatch).
    """
    if isinstance(observation, (list, tuple)):
        spaces_list = []
        for s in observation:
            arr = np.asarray(s)
            spaces_list.append(
                gymnasium.spaces.Box(
                    low=np.float32(-np.inf),
                    high=np.float32(np.inf),
                    shape=arr.shape,
                    dtype=np.float32,
                )
            )
        return gymnasium.spaces.Tuple(tuple(spaces_list))
    else:
        arr = np.asarray(observation)
        return gymnasium.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=arr.shape,
            dtype=np.float32,
        )


def _observation_dim(space: gymnasium.spaces.Space) -> int:
    """Total dimension of an observation space (Tuple of Box or single Box)."""
    import math

    if isinstance(space, gymnasium.spaces.Tuple):
        return sum(math.prod(s.shape or ()) for s in space.spaces)
    return math.prod(space.shape or ())


def _one_obs_from_batch(batch_obs) -> np.ndarray | tuple:
    """Extract a single observation (numpy) from batch observation (tensor or tuple of tensors)."""
    if isinstance(batch_obs, (list, tuple)):
        return tuple(
            cast(np.ndarray, t[0].cpu().numpy() if hasattr(t, "cpu") else np.asarray(t[0]))
            for t in batch_obs
        )
    if hasattr(batch_obs, "cpu"):
        return cast(np.ndarray, batch_obs[0].cpu().numpy())
    return cast(np.ndarray, np.asarray(batch_obs[0]))


def _batch_observation_dim(batch) -> int:
    """Total observation dimension from a training batch (batch[0] = prev_obs)."""
    one_obs = _one_obs_from_batch(batch[0])
    return _observation_dim(_observation_space_from_sample(one_obs))


def _check_observation_integrity(batch) -> None:
    """Assert batch observations are finite (no NaN/Inf) when OBSERVATION_BOUNDS_CHECK is True."""
    if not getattr(cfg, "OBSERVATION_BOUNDS_CHECK", False):
        return
    for name, obs in (("prev_obs", batch[0]), ("next_obs", batch[3])):
        if isinstance(obs, (tuple, list)):
            for i, t in enumerate(obs):
                if (
                    isinstance(t, torch.Tensor)
                    and t.is_floating_point()
                    and (torch.isnan(t).any() or torch.isinf(t).any())
                ):
                    raise ValueError(
                        f"Observation integrity check failed: {name}[{i}] contains NaN or Inf"
                    )
        elif (
            isinstance(obs, torch.Tensor)
            and obs.is_floating_point()
            and (torch.isnan(obs).any() or torch.isinf(obs).any())
        ):
            raise ValueError(f"Observation integrity check failed: {name} contains NaN or Inf")
