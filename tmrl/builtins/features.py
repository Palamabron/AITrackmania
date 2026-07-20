"""Feature pipelines supplied with TMRL for built-in off-policy algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from tmrl.core.data import Transition
from tmrl.core.pytree import PyTree, sanitize_finite, tree_collate


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
            "_tmrl_batch_collated": True,
            "observations": self.collate_observations(
                [transition.observation for transition in transitions]
            ),
            "actions": tree_collate([transition.action for transition in transitions]),
            "rewards": torch.tensor(
                [transition.reward for transition in transitions], dtype=torch.float32
            ),
            "next_observations": self.collate_observations(
                [transition.next_observation for transition in transitions]
            ),
            "terminated": torch.tensor(
                [transition.terminated for transition in transitions], dtype=torch.bool
            ),
            "truncated": torch.tensor(
                [transition.truncated for transition in transitions], dtype=torch.bool
            ),
        }

    def _collate(
        self,
        space: gymnasium.Space[Any],
        values: Sequence[Any],
        path: str,
    ) -> PyTree:
        if isinstance(space, spaces.Dict):
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
        if isinstance(space, spaces.Tuple):
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
        if isinstance(space, spaces.Box):
            return self._collate_box(space, values, path)
        if isinstance(space, spaces.Discrete):
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
        if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
            arrays = [np.asarray(value) for value in values]
            if any(not np.issubdtype(array.dtype, np.integer) for array in arrays):
                raise ValueError(f"{path} must contain integer values")
            tensor = torch.as_tensor(np.stack(arrays))
            if tensor.shape != (len(values), *space.shape):
                raise ValueError(f"{path} must have batched shape {(len(values), *space.shape)}")
            if isinstance(space, spaces.MultiBinary):
                if bool(((tensor != 0) & (tensor != 1)).any()):
                    raise ValueError(f"{path} contains values outside the MultiBinary space")
            else:
                array_start = np.asarray(cast(Any, space.start), dtype=np.int64)
                counts = np.asarray(cast(Any, space.nvec), dtype=np.int64)
                low = torch.as_tensor(array_start, dtype=tensor.dtype)
                high = low + torch.as_tensor(counts, dtype=tensor.dtype)
                if bool(((tensor < low) | (tensor >= high)).any()):
                    raise ValueError(f"{path} contains values outside the MultiDiscrete space")
            return tensor.to(torch.int64)
        raise TypeError(f"Unsupported Gymnasium observation space {type(space).__name__}")

    @staticmethod
    def _collate_box(space: spaces.Box, values: Sequence[Any], path: str) -> torch.Tensor:
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
    """Collate finite standard transitions for TMRL 1.0 learners."""

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
            "_tmrl_batch_collated": True,
            "observations": tree_collate([item.observation for item in transitions]),
            "actions": tree_collate([item.action for item in transitions]),
            "rewards": tree_collate([item.reward for item in transitions]),
            "next_observations": tree_collate([item.next_observation for item in transitions]),
            "terminated": tree_collate([item.terminated for item in transitions]),
            "truncated": tree_collate([item.truncated for item in transitions]),
        }
