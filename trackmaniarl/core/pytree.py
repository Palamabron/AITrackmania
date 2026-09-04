"""Small, explicit PyTree helpers used at the collection/training boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from numbers import Number
from typing import Any, Literal

import numpy as np
import torch

type PyTree = Any
type DeviceTransferMode = Literal["blocking", "non_blocking"]


def tree_map(function: Callable[[Any], Any], value: PyTree) -> PyTree:
    """Apply ``function`` to every leaf while preserving tuple/list/dict structure."""

    if isinstance(value, tuple):
        return tuple(tree_map(function, item) for item in value)
    if isinstance(value, list):
        return [tree_map(function, item) for item in value]
    if isinstance(value, Mapping):
        return {key: tree_map(function, item) for key, item in value.items()}
    return function(value)


def tree_to_device(
    value: PyTree,
    device: torch.device | str,
    *,
    mode: DeviceTransferMode = "blocking",
) -> PyTree:
    """Move a tensor PyTree to ``device`` without silently coercing unsupported leaves."""

    def move(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device, non_blocking=mode == "non_blocking")
        if isinstance(leaf, (bool, int, float, str, type(None))):
            return leaf
        raise TypeError(
            "PyTree inference inputs must contain tensors or scalar metadata; "
            f"got unsupported leaf {type(leaf).__name__}"
        )

    return tree_map(move, value)


def tree_collate(values: Sequence[PyTree]) -> PyTree:
    """Stack homogeneous tensor/array/scalar PyTrees into a batch."""

    if not values:
        raise ValueError("Cannot collate an empty PyTree sequence")
    first = values[0]
    if isinstance(first, torch.Tensor):
        return _collate_tensors(values)
    if isinstance(first, np.ndarray):
        return _collate_arrays(values)
    if isinstance(first, tuple):
        return _collate_sequence(values, first, tuple)
    if isinstance(first, list):
        return _collate_sequence(values, first, list)
    if isinstance(first, Mapping):
        return _collate_mapping(values, first)
    if isinstance(first, Number):
        return torch.as_tensor(values)
    raise TypeError(f"Cannot collate unsupported PyTree leaf {type(first).__name__}")


def _collate_tensors(values: Sequence[PyTree]) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Cannot collate mixed tensor and non-tensor leaves")
    return torch.stack(list(values))


def _collate_arrays(values: Sequence[PyTree]) -> torch.Tensor:
    if not all(isinstance(value, np.ndarray) for value in values):
        raise TypeError("Cannot collate mixed ndarray and non-ndarray leaves")
    return torch.as_tensor(np.stack(list(values)))


def _collate_sequence(
    values: Sequence[PyTree],
    first: Sequence[PyTree],
    container: type[list[Any]] | type[tuple[Any, ...]],
) -> PyTree:
    if not all(isinstance(value, container) and len(value) == len(first) for value in values):
        raise TypeError(f"Cannot collate {container.__name__}s with different structures")
    collated = [tree_collate([value[index] for value in values]) for index in range(len(first))]
    return tuple(collated) if container is tuple else collated


def _collate_mapping(values: Sequence[PyTree], first: Mapping[Any, Any]) -> PyTree:
    keys = tuple(first)
    if not all(isinstance(value, Mapping) and tuple(value) == keys for value in values):
        raise TypeError("Cannot collate mappings with different keys or key order")
    return {key: tree_collate([value[key] for value in values]) for key in keys}


def tree_snapshot(value: PyTree) -> PyTree:
    """Take ownership of mutable PyTree leaves at an asynchronous boundary."""

    def snapshot(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.detach().clone()
        if isinstance(leaf, np.ndarray):
            return np.array(leaf, copy=True, order="C")
        return deepcopy(leaf)

    return tree_map(snapshot, value)


def sanitize_finite(value: PyTree) -> PyTree:
    """Replace non-finite floating tensor/array values with zero at model boundaries."""

    def sanitize(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return torch.nan_to_num(leaf) if leaf.is_floating_point() else leaf
        if isinstance(leaf, np.ndarray):
            return np.nan_to_num(leaf) if np.issubdtype(leaf.dtype, np.floating) else leaf
        if isinstance(leaf, (bool, int, float, str, type(None))):
            return leaf
        raise TypeError(f"Cannot sanitize unsupported PyTree leaf {type(leaf).__name__}")

    return tree_map(sanitize, value)
