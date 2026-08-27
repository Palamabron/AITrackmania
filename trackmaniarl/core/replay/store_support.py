"""Contracts and column storage used by in-memory replay."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from numbers import Number
from typing import Any, Protocol, TypeGuard

import numpy as np
import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import Transition, TransitionId


class _IncrementalReplayStore(Protocol):
    """Optional high-throughput hooks supplied by ``InMemoryReplayStore``."""

    capacity: int

    def append(self, transition: Transition) -> TransitionId: ...

    def get(self, transition_ids: list[TransitionId]) -> list[Transition]: ...

    def available_ids(self) -> list[TransitionId]: ...

    def contains(self, transition_id: TransitionId) -> bool: ...

    def __len__(self) -> int: ...

    def eligible_transition_ids(self, n_step: int) -> list[TransitionId]: ...

    def is_n_step_eligible(self, transition_id: TransitionId, n_step: int) -> bool: ...

    def affected_n_step_starts(
        self, transition_id: TransitionId, n_step: int
    ) -> list[TransitionId]: ...

    def n_step_ids(self, transition_id: TransitionId, n_step: int) -> list[TransitionId]: ...

    def history_ids(
        self, transition_id: TransitionId, sequence_length: int
    ) -> list[TransitionId]: ...

    def next_history_observations(
        self, transition_id: TransitionId, n_step: int, sequence_length: int
    ) -> list[Any]: ...

    def sampling_pace_s(self, transition_id: TransitionId) -> float: ...

    def demo_flags(self, transition_ids: list[TransitionId]) -> list[bool]: ...

    def changes_since(
        self, revision: int | None
    ) -> tuple[int, list[tuple[TransitionId, TransitionId | None]] | None]: ...

    def sampling_transaction(self) -> AbstractContextManager[None]: ...


def _is_incremental_store(store: ReplayStore) -> TypeGuard[_IncrementalReplayStore]:
    return all(
        callable(getattr(store, name, None))
        for name in (
            "changes_since",
            "eligible_transition_ids",
            "is_n_step_eligible",
            "affected_n_step_starts",
            "n_step_ids",
            "history_ids",
            "sampling_transaction",
        )
    ) and isinstance(getattr(store, "capacity", None), int)


class _TreeColumns:
    """Fixed-shape numeric PyTree stored in contiguous capacity-first arrays."""

    def __init__(self, capacity: int, example: Any) -> None:
        self.capacity = capacity
        self.arrays: list[np.ndarray[Any, Any]] = []
        self.spec = self._build(example)

    def _build(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return (
                "mapping",
                tuple(value),
                tuple(self._build(value[key]) for key in value),
            )
        if isinstance(value, tuple):
            return ("tuple", tuple(self._build(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(self._build(item) for item in value))
        array, leaf_kind = self._leaf(value)
        index = len(self.arrays)
        self.arrays.append(np.empty((self.capacity, *array.shape), dtype=array.dtype))
        return ("leaf", index, leaf_kind, array.dtype.str, array.shape)

    @staticmethod
    def _leaf(value: Any) -> tuple[np.ndarray[Any, Any], str]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy(), "tensor"
        if isinstance(value, np.ndarray):
            return value, "ndarray"
        if isinstance(value, (Number, np.generic)):
            return np.asarray(value), "scalar"
        raise TypeError(
            "Array replay PyTrees require tensor, ndarray, or numeric leaves; "
            f"got {type(value).__name__}"
        )

    def write(self, slot: int, value: Any) -> None:
        self._write_node(self.spec, slot, value)

    def _write_node(self, spec: Any, slot: int, value: Any) -> None:
        kind = spec[0]
        if kind == "mapping":
            if not isinstance(value, Mapping) or tuple(value) != spec[1]:
                raise TypeError("Replay PyTree mapping structure changed after allocation")
            for key, child in zip(spec[1], spec[2], strict=True):
                self._write_node(child, slot, value[key])
            return
        if kind in {"tuple", "list"}:
            expected = tuple if kind == "tuple" else list
            if not isinstance(value, expected) or len(value) != len(spec[1]):
                raise TypeError(f"Replay PyTree {kind} structure changed after allocation")
            for child, item in zip(spec[1], value, strict=True):
                self._write_node(child, slot, item)
            return
        array, _ = self._leaf(value)
        if array.shape != spec[4] or array.dtype.str != spec[3]:
            raise TypeError("Replay PyTree leaf shape or dtype changed after allocation")
        self.arrays[spec[1]][slot] = array

    def read(self, slot: int) -> Any:
        return self._read_node(self.spec, slot)

    def _read_node(self, spec: Any, slot: int) -> Any:
        kind = spec[0]
        if kind == "mapping":
            return {
                key: self._read_node(child, slot)
                for key, child in zip(spec[1], spec[2], strict=True)
            }
        if kind == "tuple":
            return tuple(self._read_node(child, slot) for child in spec[1])
        if kind == "list":
            return [self._read_node(child, slot) for child in spec[1]]
        value = self.arrays[spec[1]][slot]
        if spec[2] == "tensor":
            return torch.from_numpy(value)
        if spec[2] == "ndarray":
            return value
        return value.item()

    def snapshot(self, slots: slice | np.ndarray[Any, np.dtype[np.int64]]) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "arrays": [np.array(array[slots], copy=True, order="C") for array in self.arrays],
        }

    @classmethod
    def restore(
        cls,
        capacity: int,
        snapshot: Mapping[str, Any],
        slots: np.ndarray[Any, np.dtype[np.int64]],
    ) -> _TreeColumns:
        value = cls.__new__(cls)
        value.capacity = capacity
        value.spec = snapshot["spec"]
        packed = list(snapshot["arrays"])
        value.arrays = [
            np.empty((capacity, *array.shape[1:]), dtype=array.dtype) for array in packed
        ]
        for target, source in zip(value.arrays, packed, strict=True):
            target[slots] = source
        return value


def _is_demo(info: Mapping[str, Any]) -> bool:
    return bool(info.get("is_demo", False))
