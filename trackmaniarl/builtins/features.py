"""Feature pipelines supplied with TrackmaniaRL for built-in off-policy algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from trackmaniarl.core.data import Transition
from trackmaniarl.core.pytree import PyTree, sanitize_finite, tree_collate


def _typed_tensor(values: Sequence[float | bool], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _transition_scalars(transitions: list[Transition]) -> dict[str, torch.Tensor]:
    return {
        "rewards": _typed_tensor([item.reward for item in transitions], torch.float32),
        "terminated": _typed_tensor([item.terminated for item in transitions], torch.bool),
        "truncated": _typed_tensor([item.truncated for item in transitions], torch.bool),
    }


class GymnasiumObservationCollator:
    """Validate and batch observations described by a Gymnasium space."""

    def __init__(self, observation_space: gymnasium.Space[Any]) -> None:
        self.observation_space = observation_space

    def collate_observations(self, observations: Sequence[Any]) -> PyTree:
        if not observations:
            raise ValueError("Cannot collate an empty observation sequence")
        return self._collate(self.observation_space, observations, "observation")

    def collate_transitions(self, transitions: list[Transition]) -> Mapping[str, PyTree]:
        return {
            "_trackmaniarl_batch_collated": True,
            "observations": self.collate_observations(
                [transition.observation for transition in transitions]
            ),
            "actions": tree_collate([transition.action for transition in transitions]),
            "next_observations": self.collate_observations(
                [transition.next_observation for transition in transitions]
            ),
            **_transition_scalars(transitions),
        }

    def _collate(
        self,
        space: gymnasium.Space[Any],
        values: Sequence[Any],
        path: str,
    ) -> PyTree:
        if isinstance(space, spaces.Dict):
            return self._collate_dict(space, values, path)
        if isinstance(space, spaces.Tuple):
            return self._collate_tuple(space, values, path)
        if isinstance(space, spaces.Box):
            return self._collate_box(space, values, path)
        if isinstance(space, spaces.Discrete):
            return self._collate_discrete(space, values, path)
        if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
            return self._collate_integer_array(space, values, path)
        raise TypeError(f"Unsupported Gymnasium observation space {type(space).__name__}")

    def _collate_dict(
        self, space: spaces.Dict, values: Sequence[Any], path: str
    ) -> dict[str, PyTree]:
        expected = tuple(space.spaces)
        if not all(isinstance(value, Mapping) and tuple(value) == expected for value in values):
            raise ValueError(f"{path} must match Dict keys {expected}")
        return {
            key: self._collate(
                child,
                [cast(Mapping[str, Any], value)[key] for value in values],
                f"{path}.{key}",
            )
            for key, child in space.spaces.items()
        }

    def _collate_tuple(
        self, space: spaces.Tuple, values: Sequence[Any], path: str
    ) -> tuple[PyTree, ...]:
        valid = all(
            isinstance(value, tuple) and len(value) == len(space.spaces) for value in values
        )
        if not valid:
            raise ValueError(f"{path} must match Tuple length {len(space.spaces)}")
        return tuple(
            self._collate(
                child,
                [cast(tuple[Any, ...], value)[index] for value in values],
                f"{path}[{index}]",
            )
            for index, child in enumerate(space.spaces)
        )

    @staticmethod
    def _collate_discrete(
        space: spaces.Discrete[Any], values: Sequence[Any], path: str
    ) -> torch.Tensor:
        if any(not np.issubdtype(np.asarray(value).dtype, np.integer) for value in values):
            raise ValueError(f"{path} must contain integer Discrete values")
        tensor = torch.as_tensor(values, dtype=torch.int64)
        if tensor.shape != (len(values),):
            raise ValueError(f"{path} must contain scalar discrete values")
        start = int(cast(Any, space.start))
        count = int(cast(Any, space.n))
        if bool(((tensor < start) | (tensor >= start + count)).any()):
            raise ValueError(f"{path} contains values outside the Discrete space")
        return tensor

    @staticmethod
    def _collate_integer_array(
        space: spaces.MultiBinary | spaces.MultiDiscrete,
        values: Sequence[Any],
        path: str,
    ) -> torch.Tensor:
        arrays = [np.asarray(value) for value in values]
        if any(not np.issubdtype(array.dtype, np.integer) for array in arrays):
            raise ValueError(f"{path} must contain integer values")
        tensor = torch.as_tensor(np.stack(arrays))
        if tensor.shape != (len(values), *space.shape):
            raise ValueError(f"{path} must have batched shape {(len(values), *space.shape)}")
        _validate_integer_bounds(space, tensor, path)
        return tensor.to(torch.int64)

    @staticmethod
    def _collate_box(space: spaces.Box, values: Sequence[Any], path: str) -> torch.Tensor:
        if all(isinstance(value, torch.Tensor) for value in values):
            return _collate_tensor_box(space, cast(Sequence[torch.Tensor], values), path)
        return _collate_array_box(space, values, path)


def _validate_integer_bounds(
    space: spaces.MultiBinary | spaces.MultiDiscrete, tensor: torch.Tensor, path: str
) -> None:
    if isinstance(space, spaces.MultiBinary):
        if bool(((tensor != 0) & (tensor != 1)).any()):
            raise ValueError(f"{path} contains values outside the MultiBinary space")
        return
    array_start = np.asarray(cast(Any, space.start), dtype=np.int64)
    counts = np.asarray(cast(Any, space.nvec), dtype=np.int64)
    low = torch.as_tensor(array_start, dtype=tensor.dtype)
    high = low + torch.as_tensor(counts, dtype=tensor.dtype)
    if bool(((tensor < low) | (tensor >= high)).any()):
        raise ValueError(f"{path} contains values outside the MultiDiscrete space")


def _collate_tensor_box(
    space: spaces.Box, tensors: Sequence[torch.Tensor], path: str
) -> torch.Tensor:
    if any(tuple(tensor.shape) != space.shape for tensor in tensors):
        raise ValueError(f"{path} must have shape {space.shape}")
    expected_dtype = torch.from_numpy(np.empty((), dtype=space.dtype)).dtype
    if any(tensor.dtype != expected_dtype for tensor in tensors):
        raise ValueError(f"{path} must have dtype {space.dtype}")
    stacked = torch.stack(tuple(tensors))
    if stacked.is_floating_point() and not torch.isfinite(stacked).all():
        raise ValueError(f"{path} contains non-finite values")
    low = torch.as_tensor(space.low, dtype=stacked.dtype, device=stacked.device)
    high = torch.as_tensor(space.high, dtype=stacked.dtype, device=stacked.device)
    if bool((stacked < low).any()) or bool((stacked > high).any()):
        raise ValueError(f"{path} contains values outside the Box space")
    return stacked


def _collate_array_box(space: spaces.Box, values: Sequence[Any], path: str) -> torch.Tensor:
    arrays = [
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for value in values
    ]
    if any(array.shape != space.shape for array in arrays):
        raise ValueError(f"{path} must have shape {space.shape}")
    if any(array.dtype != space.dtype for array in arrays):
        raise ValueError(f"{path} must have dtype {space.dtype}")
    stacked = np.stack(arrays)
    if np.issubdtype(stacked.dtype, np.floating) and not np.isfinite(stacked).all():
        raise ValueError(f"{path} contains non-finite values")
    if np.any(stacked < space.low) or np.any(stacked > space.high):
        raise ValueError(f"{path} contains values outside the Box space")
    return torch.from_numpy(np.ascontiguousarray(stacked))


class TransitionFeaturePipeline:
    """Collate finite standard transitions for TrackmaniaRL 2.0 learners."""

    def __init__(self, observation_space: gymnasium.Space[Any] | None = None) -> None:
        self.observation_space = observation_space
        self._collator = (
            GymnasiumObservationCollator(observation_space)
            if observation_space is not None
            else None
        )

    def transform_observation(self, observation: Any) -> Any:
        return sanitize_finite(observation)

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        if self._collator is not None:
            return dict(self._collator.collate_transitions(transitions))
        return {
            "_trackmaniarl_batch_collated": True,
            "observations": tree_collate([item.observation for item in transitions]),
            "actions": tree_collate([item.action for item in transitions]),
            "rewards": tree_collate([item.reward for item in transitions]),
            "next_observations": tree_collate([item.next_observation for item in transitions]),
            "terminated": tree_collate([item.terminated for item in transitions]),
            "truncated": tree_collate([item.truncated for item in transitions]),
        }
