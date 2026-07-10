from collections.abc import Mapping, Sequence
from typing import Any, cast

import gymnasium
import numpy as np


class AffineObservationWrapper(gymnasium.ObservationWrapper):
    """Gymnasium wrapper that applies an affine transform ``(obs + shift) * scale`` to observations.

    Only ``gymnasium.spaces.Box`` observation spaces are supported. The observation
    space bounds are transformed alongside the observations.

    Args:
        env: The environment to wrap.
        shift: Value added to each observation before scaling.
        scale: Value by which the shifted observation is multiplied.
    """

    def __init__(self, env, shift, scale):
        super().__init__(env)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        self.shift = shift
        self.scale = scale
        self.observation_space = gymnasium.spaces.Box(
            self.observation(env.observation_space.low),
            self.observation(env.observation_space.high),
            dtype=cast(Any, env.observation_space.dtype),
        )

    def observation(self, observation):
        """Apply the affine transform to a single observation.

        Args:
            observation: Raw observation from the wrapped environment.

        Returns:
            Transformed observation: ``(observation + self.shift) * self.scale``.
        """
        return (observation + self.shift) * self.scale


def _space_to_float32(space):
    """Return a copy of the space with float Box dtypes set to np.float32."""
    if isinstance(space, gymnasium.spaces.Box):
        if np.issubdtype(space.dtype, np.floating):
            return gymnasium.spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=np.float32,
            )
        return space
    if isinstance(space, gymnasium.spaces.Dict):
        return gymnasium.spaces.Dict({k: _space_to_float32(v) for k, v in space.spaces.items()})
    if isinstance(space, gymnasium.spaces.Tuple):
        return gymnasium.spaces.Tuple([_space_to_float32(s) for s in space.spaces])
    return space


class Float64ToFloat32(gymnasium.ObservationWrapper):
    """Converts np.float64 arrays in the observations to np.float32 arrays."""

    def __init__(self, env):
        """Initialize the wrapper and convert float Box spaces to float32.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.observation_space = _space_to_float32(env.observation_space)
        self.action_space = _space_to_float32(env.action_space)

    def observation(self, observation):
        """Cast all float64 numpy arrays in the observation to float32.

        Args:
            observation: Raw observation from the wrapped environment.

        Returns:
            Observation with the same structure, but float64 arrays replaced by float32.
        """
        observation = deepmap(
            {
                np.ndarray: float64_to_float32,
                float: float_to_float32,
                int: int_to_float32,
                np.float32: float_to_float32,
                np.float64: float_to_float32,
            },
            observation,
        )
        return observation

    def step(self, action):
        """Step the environment; observations are cast to float32 by :meth:`observation`.

        Args:
            action: Action to pass to the wrapped environment.

        Returns:
            Tuple of (observation, reward, done, terminated, info) with float32 observations.
        """
        observation, reward, done, terminated, info = super().step(action)
        return observation, reward, done, terminated, info


# === Utilities ================================================================


def deepmap(f, m):
    """Apply functions to the leaves of a dictionary or list, depending type of the leaf value."""
    for cls in f:
        if isinstance(m, cls):
            return f[cls](m)
    if isinstance(m, Sequence) and not isinstance(m, (str, bytes, bytearray)):
        ctor: Any = type(m)
        return ctor(deepmap(f, x) for x in m)
    if isinstance(m, Mapping):
        ctor_map: Any = type(m)
        return ctor_map((k, deepmap(f, m[k])) for k in m)
    else:
        raise AttributeError(f"m is a {type(m)}, not a Sequence nor a Mapping: {m}")


def float64_to_float32(x):
    """Cast a ``float64`` numpy array to ``float32``; return other dtypes unchanged.

    Args:
        x: A numpy array.

    Returns:
        numpy.ndarray: ``float32`` array if input dtype is ``float64``, otherwise ``x`` unchanged.
    """
    return (
        np.asarray(
            [
                x,
            ],
            np.float32,
        )
        if x.dtype == np.float64
        else x
    )


def float_to_float32(x):
    """Wrap a Python float in a single-element ``float32`` numpy array.

    Args:
        x: A Python float or compatible scalar.

    Returns:
        numpy.ndarray: Shape ``(1,)`` array of dtype ``float32``.
    """
    return np.asarray(
        [
            x,
        ],
        np.float32,
    )


def int_to_float32(x):
    """Wrap a Python int in a single-element ``float32`` numpy array.

    Args:
        x: A Python int or compatible scalar.

    Returns:
        numpy.ndarray: Shape ``(1,)`` array of dtype ``float32``.
    """
    return np.asarray(
        [
            float(x),
        ],
        np.float32,
    )
