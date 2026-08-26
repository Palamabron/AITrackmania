"""Columnar in-memory replay storage with stable transition IDs."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from math import isfinite
from numbers import Number
from threading import RLock
from typing import Any, Protocol, TypeGuard, cast

import numpy as np
import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, Transition, TransitionId
from trackmaniarl.core.pytree import tree_snapshot


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
            keys = spec[1]
            if not isinstance(value, Mapping) or tuple(value) != keys:
                raise TypeError("Replay PyTree mapping structure changed after allocation")
            for key, child in zip(keys, spec[2], strict=True):
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
    return bool(info.get("is_demo", False) or info.get("source") == "demo")


class InMemoryReplayStore:
    """Preallocated columnar FIFO replay with stable transition IDs."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._ids = np.full(capacity, -1, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float64)
        self._terminated = np.empty(capacity, dtype=np.bool_)
        self._truncated = np.empty(capacity, dtype=np.bool_)
        self._episode_codes = np.full(capacity, -1, dtype=np.int64)
        self._steps = np.full(capacity, -1, dtype=np.int64)
        self._previous_ids = np.full(capacity, -1, dtype=np.int64)
        self._next_ids = np.full(capacity, -1, dtype=np.int64)
        self._sampling_pace = np.full(capacity, np.inf, dtype=np.float32)
        self._observations: _TreeColumns | None = None
        self._actions: _TreeColumns | None = None
        self._next_overrides: dict[TransitionId, Any] = {}
        self._info: dict[TransitionId, Mapping[str, Any]] = {}
        self._episode_names: dict[int, str] = {}
        self._episode_codes_by_name: dict[str, int] = {}
        self._episode_steps: dict[int, dict[int, TransitionId]] = {}
        self._episode_terminal_steps: dict[int, int] = {}
        self._episode_refcounts: dict[int, int] = {}
        self._next_episode_code = 0
        self._demo_flags = np.zeros(capacity, dtype=np.bool_)
        self._demo_count = 0
        self._next_index = 0
        self._size = 0
        self._lock = RLock()
        self._revision = 0
        self._changes: deque[tuple[int, TransitionId, TransitionId | None]] = deque(
            maxlen=min(capacity, 65_536)
        )

    def append(self, transition: Transition) -> TransitionId:
        """Append one transition; demonstrations displaced by the ring are re-appended."""

        with self._lock:
            transition_id, resurrected = self._append_locked(transition)
            while resurrected is not None:
                _, resurrected = self._append_locked(resurrected)
            return transition_id

    def _append_locked(self, transition: Transition) -> tuple[TransitionId, Transition | None]:
        episode_code = self._episode_code(transition.episode_id)
        previous_id = self._previous_transition(transition, episode_code)
        transition_id = self._next_index
        slot = transition_id % self.capacity
        evicted = int(self._ids[slot]) if self._ids[slot] >= 0 else None
        resurrected: Transition | None = None
        if evicted is not None:
            if self._demo_flags[slot]:
                if self._demo_count * 2 >= self.capacity:
                    raise RuntimeError(
                        "replay capacity is too small to protect demonstration transitions"
                    )
                resurrected = self._resurrectable_transition(evicted, slot)
                self._demo_count -= 1
            self._info.pop(evicted, None)
            self._next_overrides.pop(evicted, None)
            self._release_episode_reference(int(self._episode_codes[slot]))
        self._allocate_columns(transition)
        assert self._observations is not None
        assert self._actions is not None
        self._observations.write(slot, transition.observation)
        self._actions.write(slot, transition.action)
        self._ids[slot] = transition_id
        self._rewards[slot] = transition.reward
        self._terminated[slot] = transition.terminated
        self._truncated[slot] = transition.truncated
        self._episode_codes[slot] = episode_code
        self._steps[slot] = transition.step if transition.step is not None else -1
        self._previous_ids[slot] = previous_id
        self._next_ids[slot] = -1
        transition_info = cast(dict[str, Any], tree_snapshot(dict(transition.info)))
        self._sampling_pace[slot] = float(
            transition_info.pop("sampling/projected_lap_time_s", np.inf)
        )
        is_demo = _is_demo(transition_info)
        self._demo_flags[slot] = is_demo
        self._demo_count += int(is_demo)
        if episode_code >= 0:
            self._episode_refcounts[episode_code] = self._episode_refcounts.get(episode_code, 0) + 1
        self._next_overrides[transition_id] = tree_snapshot(transition.next_observation)
        if transition_info:
            self._info[transition_id] = transition_info
        self._link_previous(previous_id, transition_id, transition.observation)
        if episode_code >= 0 and transition.step is not None:
            self._register_episode_step(episode_code, transition.step, transition_id)
        self._next_index += 1
        self._size = min(self.capacity, self._size + 1)
        self._revision += 1
        self._changes.append((self._revision, transition_id, evicted))
        return transition_id, resurrected

    def _resurrectable_transition(self, transition_id: TransitionId, slot: int) -> Transition:
        resurrected = self._transition(transition_id)
        pace = float(self._sampling_pace[slot])
        if not isfinite(pace):
            return resurrected
        return replace(
            resurrected,
            info={**resurrected.info, "sampling/projected_lap_time_s": pace},
        )

    def _release_episode_reference(self, episode_code: int) -> None:
        if episode_code < 0:
            return
        remaining = self._episode_refcounts.get(episode_code, 0) - 1
        if remaining > 0:
            self._episode_refcounts[episode_code] = remaining
            return
        self._episode_refcounts.pop(episode_code, None)
        name = self._episode_names.pop(episode_code, None)
        if name is not None:
            self._episode_codes_by_name.pop(name, None)
        self._episode_steps.pop(episode_code, None)
        self._episode_terminal_steps.pop(episode_code, None)

    @contextmanager
    def sampling_transaction(self) -> Iterator[None]:
        """Keep sampled IDs valid until their batch is fully materialized."""

        with self._lock:
            yield

    def _allocate_columns(self, transition: Transition) -> None:
        if self._observations is None:
            self._observations = _TreeColumns(self.capacity, transition.observation)
            self._actions = _TreeColumns(self.capacity, transition.action)

    def _episode_code(self, episode_id: str | None) -> int:
        if episode_id is None:
            return -1
        existing = self._episode_codes_by_name.get(episode_id)
        if existing is not None:
            return existing
        code = self._next_episode_code
        self._next_episode_code += 1
        self._episode_names[code] = episode_id
        self._episode_codes_by_name[episode_id] = code
        return code

    def _previous_transition(self, transition: Transition, episode_code: int) -> TransitionId:
        if episode_code < 0 or transition.step is None:
            candidate = self._next_index - 1
            if not self.contains(candidate):
                return -1
            slot = candidate % self.capacity
            if self._terminated[slot] or self._truncated[slot]:
                return -1
            if int(self._episode_codes[slot]) != episode_code:
                return -1
            return candidate
        steps = self._episode_steps.setdefault(episode_code, {})
        existing = self._episode_step(steps, transition.step)
        if existing >= 0:
            episode_id = self._episode_names[episode_code]
            raise ValueError(
                f"duplicate replay episode step: episode={episode_id!r}, step={transition.step}"
            )
        return self._episode_step(steps, transition.step - 1)

    def _register_episode_step(
        self, episode_code: int, step: int, transition_id: TransitionId
    ) -> None:
        steps = self._episode_steps.setdefault(episode_code, {})
        steps[step] = transition_id
        successor = self._episode_step(steps, step + 1)
        if successor >= 0:
            assert self._observations is not None
            self._link_previous(
                transition_id,
                successor,
                self._observations.read(successor % self.capacity),
            )
        slot = transition_id % self.capacity
        if self._terminated[slot] or self._truncated[slot]:
            self._episode_terminal_steps[episode_code] = step
        self._release_completed_episode(episode_code)

    def _episode_step(self, steps: dict[int, TransitionId], step: int) -> TransitionId:
        transition_id = steps.get(step, -1)
        if transition_id >= 0 and not self.contains(transition_id):
            steps.pop(step, None)
            return -1
        return transition_id

    def _release_completed_episode(self, episode_code: int) -> None:
        terminal_step = self._episode_terminal_steps.get(episode_code)
        if terminal_step is None:
            return
        steps = self._episode_steps[episode_code]
        if len(steps) < terminal_step + 1:
            return
        if all(self._episode_step(steps, step) >= 0 for step in range(terminal_step + 1)):
            self._episode_steps.pop(episode_code)
            self._episode_terminal_steps.pop(episode_code)

    def _link_previous(
        self,
        previous_id: TransitionId,
        transition_id: TransitionId,
        observation: Any,
    ) -> None:
        if previous_id < 0 or not self.contains(previous_id):
            return
        previous_slot = previous_id % self.capacity
        transition_slot = transition_id % self.capacity
        if (
            self._terminated[previous_slot]
            or self._truncated[previous_slot]
            or self._episode_codes[previous_slot] != self._episode_codes[transition_slot]
        ):
            return
        self._next_ids[previous_slot] = transition_id
        self._previous_ids[transition_slot] = previous_id
        previous_next = self._next_overrides.get(previous_id)
        if (
            transition_id > previous_id
            and previous_next is not None
            and self._tree_equal(previous_next, observation)
        ):
            self._next_overrides.pop(previous_id)

    @staticmethod
    def _tree_equal(left: Any, right: Any) -> bool:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return tuple(left) == tuple(right) and all(
                InMemoryReplayStore._tree_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(
                InMemoryReplayStore._tree_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        if isinstance(left, torch.Tensor):
            left = left.detach().cpu().numpy()
        if isinstance(right, torch.Tensor):
            right = right.detach().cpu().numpy()
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))

    def get(self, transition_ids: list[TransitionId]) -> list[Transition]:
        with self._lock:
            missing = [
                transition_id
                for transition_id in transition_ids
                if not self.contains(transition_id)
            ]
            if missing:
                raise KeyError(f"Replay transitions no longer available: {missing[:3]}")
            return [self._transition(transition_id) for transition_id in transition_ids]

    def materialize_n_step(
        self, transition_ids: list[TransitionId], request: BatchRequest
    ) -> tuple[list[Transition], list[float]]:
        """Aggregate n-step targets while reading only their endpoint observations."""

        with self._lock:
            missing = [
                transition_id
                for transition_id in transition_ids
                if not self.contains(transition_id)
            ]
            if missing:
                raise KeyError(f"Replay transitions no longer available: {missing[:3]}")
            materialized = [
                self._materialize_n_step_locked(transition_id, request)
                for transition_id in transition_ids
            ]
        return [item[0] for item in materialized], [item[1] for item in materialized]

    def _materialize_n_step_locked(
        self, transition_id: TransitionId, request: BatchRequest
    ) -> tuple[Transition, float]:
        assert self._observations is not None
        assert self._actions is not None
        first_slot = transition_id % self.capacity
        first_episode_code = int(self._episode_codes[first_slot])
        first_step = int(self._steps[first_slot])
        current_id = transition_id
        final_id = transition_id
        reward = 0.0
        discount = 1.0
        effective_steps = 0
        terminated = False
        truncated = False
        for offset in range(request.n_step):
            if not self.contains(current_id):
                break
            current_slot = current_id % self.capacity
            if int(self._episode_codes[current_slot]) != first_episode_code:
                break
            current_step = int(self._steps[current_slot])
            if first_step >= 0 and current_step >= 0 and current_step != first_step + offset:
                break
            final_id = current_id
            reward += discount * float(self._rewards[current_slot])
            effective_steps += 1
            terminated = bool(self._terminated[current_slot])
            truncated = bool(self._truncated[current_slot])
            if terminated or truncated:
                break
            discount *= request.gamma
            current_id = int(self._next_ids[current_slot])
        if effective_steps == 0:
            raise RuntimeError(f"Transition {transition_id} is no longer available")
        final_slot = final_id % self.capacity
        transition = Transition(
            observation=tree_snapshot(self._observations.read(first_slot)),
            action=tree_snapshot(self._actions.read(first_slot)),
            reward=reward,
            next_observation=tree_snapshot(self._next_observation(final_id, final_slot)),
            terminated=terminated,
            truncated=truncated,
            info=self._info.get(transition_id, {}),
            episode_id=self._episode_names.get(first_episode_code),
            step=first_step if first_step >= 0 else None,
        )
        bootstrap_discount = 0.0 if terminated else request.gamma**effective_steps
        return transition, bootstrap_discount

    def _transition(self, transition_id: TransitionId) -> Transition:
        assert self._observations is not None
        assert self._actions is not None
        slot = transition_id % self.capacity
        episode_code = int(self._episode_codes[slot])
        step = int(self._steps[slot])
        return Transition(
            observation=self._observations.read(slot),
            action=self._actions.read(slot),
            reward=float(self._rewards[slot]),
            next_observation=self._next_observation(transition_id, slot),
            terminated=bool(self._terminated[slot]),
            truncated=bool(self._truncated[slot]),
            info=self._info.get(transition_id, {}),
            episode_id=self._episode_names.get(episode_code),
            step=step if step >= 0 else None,
        )

    def _next_observation(self, transition_id: TransitionId, slot: int) -> Any:
        assert self._observations is not None
        next_observation = self._next_overrides.get(transition_id)
        if next_observation is not None:
            return next_observation
        next_id = int(self._next_ids[slot])
        if not self.contains(next_id):
            raise RuntimeError(f"Transition {transition_id} has no next observation")
        return self._observations.read(next_id % self.capacity)

    def available_ids(self) -> list[TransitionId]:
        with self._lock:
            return list(range(self._next_index - self._size, self._next_index))

    def available_indices(self) -> list[TransitionId]:
        """Compatibility spelling; IDs are never reused after eviction."""

        return self.available_ids()

    def contains(self, transition_id: TransitionId) -> bool:
        if transition_id < 0:
            return False
        return bool(self._ids[transition_id % self.capacity] == transition_id)

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            empty = {
                "format": "columnar-v1",
                "capacity": self.capacity,
                "size": self._size,
                "next_index": self._next_index,
            }
            if self._observations is None or self._actions is None:
                return empty
            slots = self._snapshot_slots()
            return {
                **empty,
                "observations": self._observations.snapshot(slots),
                "actions": self._actions.snapshot(slots),
                "rewards": np.array(self._rewards[slots], copy=True, order="C"),
                "terminated": np.array(self._terminated[slots], copy=True, order="C"),
                "truncated": np.array(self._truncated[slots], copy=True, order="C"),
                "episode_codes": np.array(self._episode_codes[slots], copy=True, order="C"),
                "steps": np.array(self._steps[slots], copy=True, order="C"),
                "previous_ids": np.array(self._previous_ids[slots], copy=True, order="C"),
                "next_ids": np.array(self._next_ids[slots], copy=True, order="C"),
                "sampling_pace": np.array(self._sampling_pace[slots], copy=True, order="C"),
                "episode_names": dict(self._episode_names),
                "next_overrides": dict(self._next_overrides),
                "info": dict(self._info),
            }

    def _snapshot_slots(self) -> slice | np.ndarray[Any, np.dtype[np.int64]]:
        first_id = self._next_index - self._size
        first_slot = first_id % self.capacity
        if first_slot + self._size <= self.capacity:
            return slice(first_slot, first_slot + self._size)
        ids = np.arange(first_id, self._next_index, dtype=np.int64)
        return ids % self.capacity

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format") != "columnar-v1":
            self._load_legacy_state(state)
            return
        with self._lock:
            checkpoint_capacity = int(state["capacity"])
            # Growing the buffer on resume is safe because the stored dense ID
            # range spans at most the old capacity, so IDs stay collision-free
            # under the larger modulus; shrinking would evict live transitions.
            if checkpoint_capacity > self.capacity:
                raise ValueError(
                    f"Replay checkpoint capacity {checkpoint_capacity} exceeds "
                    f"configured capacity {self.capacity}"
                )
            self._reset_arrays()
            self._size = int(state["size"])
            self._next_index = int(state["next_index"])
            if self._size:
                ids = np.asarray(
                    state.get(
                        "ids",
                        np.arange(
                            self._next_index - self._size,
                            self._next_index,
                            dtype=np.int64,
                        ),
                    ),
                    dtype=np.int64,
                )
                slots = ids % self.capacity
                self._ids[slots] = ids
                self._observations = _TreeColumns.restore(
                    self.capacity, cast(Mapping[str, Any], state["observations"]), slots
                )
                self._actions = _TreeColumns.restore(
                    self.capacity, cast(Mapping[str, Any], state["actions"]), slots
                )
                for name, target in (
                    ("rewards", self._rewards),
                    ("terminated", self._terminated),
                    ("truncated", self._truncated),
                    ("episode_codes", self._episode_codes),
                    ("steps", self._steps),
                    ("previous_ids", self._previous_ids),
                    ("next_ids", self._next_ids),
                ):
                    target[slots] = state[name]
                self._sampling_pace[slots] = state.get(
                    "sampling_pace",
                    np.full(self._size, np.inf, dtype=np.float32),
                )
            self._episode_names = {
                int(code): str(name)
                for code, name in cast(Mapping[Any, Any], state.get("episode_names", {})).items()
            }
            self._episode_codes_by_name = {name: code for code, name in self._episode_names.items()}
            self._next_episode_code = max(self._episode_names, default=-1) + 1
            self._next_overrides = dict(state.get("next_overrides", {}))
            self._info = dict(state.get("info", {}))
            self._rebuild_episode_steps()
            self._rebuild_reference_state()
            self._revision += 1
            self._changes.clear()

    def _reset_arrays(self) -> None:
        self._ids.fill(-1)
        self._episode_codes.fill(-1)
        self._steps.fill(-1)
        self._previous_ids.fill(-1)
        self._next_ids.fill(-1)
        self._sampling_pace.fill(np.inf)
        self._demo_flags.fill(False)
        self._demo_count = 0
        self._observations = None
        self._actions = None
        self._next_overrides.clear()
        self._info.clear()
        self._episode_steps.clear()
        self._episode_terminal_steps.clear()
        self._episode_refcounts.clear()

    def _rebuild_reference_state(self) -> None:
        for transition_id in range(self._next_index - self._size, self._next_index):
            slot = transition_id % self.capacity
            code = int(self._episode_codes[slot])
            if code >= 0:
                self._episode_refcounts[code] = self._episode_refcounts.get(code, 0) + 1
            is_demo = _is_demo(self._info.get(transition_id, {}))
            self._demo_flags[slot] = is_demo
            self._demo_count += int(is_demo)

    def _rebuild_episode_steps(self) -> None:
        self._episode_steps.clear()
        self._episode_terminal_steps.clear()
        completed: set[int] = set()
        for transition_id in range(self._next_index - self._size, self._next_index):
            slot = transition_id % self.capacity
            code = int(self._episode_codes[slot])
            step = int(self._steps[slot])
            if code >= 0 and step >= 0 and (self._terminated[slot] or self._truncated[slot]):
                completed.add(code)
        for transition_id in range(self._next_index - self._size, self._next_index):
            slot = transition_id % self.capacity
            code = int(self._episode_codes[slot])
            step = int(self._steps[slot])
            if code >= 0 and step >= 0 and code not in completed:
                self._episode_steps.setdefault(code, {})[step] = transition_id

    def _load_legacy_state(self, state: Mapping[str, Any]) -> None:
        order = list(state["order"])
        items = cast(Mapping[TransitionId, Transition], state["items"])
        if len(order) > self.capacity:
            raise ValueError("Legacy replay checkpoint exceeds configured capacity")
        with self._lock:
            self._reset_arrays()
            self._size = 0
            self._next_index = 0
            self._episode_names.clear()
            self._episode_codes_by_name.clear()
        for transition_id in order:
            appended = self.append(items[transition_id])
            if appended != transition_id:
                raise ValueError("Legacy replay checkpoint transition IDs are not contiguous")

    def eligible_transition_ids(self, n_step: int) -> list[TransitionId]:
        """Return complete n-step starts without retaining a second full ID index."""

        with self._lock:
            return [
                transition_id
                for transition_id in range(self._next_index - self._size, self._next_index)
                if self._is_n_step_eligible_locked(transition_id, n_step)
            ]

    def sample_eligible_ids(
        self, n_step: int, batch_size: int, rng: random.Random
    ) -> list[TransitionId]:
        """Draw complete starts by bounded rejection from the dense ID interval."""

        with self._lock:
            if self._size < batch_size:
                raise RuntimeError(
                    f"Need {batch_size} complete n-step transitions, replay has {self._size}"
                )
            chosen: list[TransitionId] = []
            chosen_set: set[TransitionId] = set()
            attempts = 0
            lower = self._next_index - self._size
            while len(chosen) < batch_size and attempts < batch_size * 32:
                candidate = rng.randrange(lower, self._next_index)
                attempts += 1
                if candidate not in chosen_set and self._is_n_step_eligible_locked(
                    candidate, n_step
                ):
                    chosen.append(candidate)
                    chosen_set.add(candidate)
            if len(chosen) < batch_size:
                eligible = self.eligible_transition_ids(n_step)
                if len(eligible) < batch_size:
                    raise RuntimeError(
                        f"Need {batch_size} complete n-step transitions, replay has {len(eligible)}"
                    )
                return rng.sample(eligible, batch_size)
            return chosen

    def is_n_step_eligible(self, transition_id: TransitionId, n_step: int) -> bool:
        with self._lock:
            return self._is_n_step_eligible_locked(transition_id, n_step)

    def n_step_ids(self, transition_id: TransitionId, n_step: int) -> list[TransitionId]:
        """Resolve an episode-local horizon even when actors are interleaved."""

        with self._lock:
            if not self.contains(transition_id):
                return []
            result: list[TransitionId] = []
            candidate_id = transition_id
            for _ in range(n_step):
                if not self.contains(candidate_id):
                    break
                result.append(candidate_id)
                slot = candidate_id % self.capacity
                if self._terminated[slot] or self._truncated[slot]:
                    break
                candidate_id = int(self._next_ids[slot])
            return result

    def affected_n_step_starts(
        self, transition_id: TransitionId, n_step: int
    ) -> list[TransitionId]:
        """Return starts whose eligibility can change after this append."""

        with self._lock:
            if not self.contains(transition_id):
                return []
            return self._predecessor_ids_locked(transition_id, n_step)

    def history_ids(self, transition_id: TransitionId, sequence_length: int) -> list[TransitionId]:
        """Return actor-equivalent left-padded history ending at ``transition_id``."""

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        with self._lock:
            if not self.contains(transition_id):
                return []
            result: list[TransitionId] = []
            candidate = transition_id
            for _ in range(sequence_length):
                if not self.contains(candidate):
                    break
                result.append(candidate)
                slot = candidate % self.capacity
                candidate = int(self._previous_ids[slot])
            result.reverse()
            return [result[0]] * (sequence_length - len(result)) + result

    def next_history_observations(
        self, transition_id: TransitionId, n_step: int, sequence_length: int
    ) -> list[Any]:
        """Return recurrent history ending at the resolved n-step next state."""

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        with self._lock:
            horizon = self.n_step_ids(transition_id, n_step)
            if not horizon:
                return []
            final_id = horizon[-1]
            history = self.history_ids(final_id, max(1, sequence_length - 1))
            if sequence_length == 1:
                history = []
            observations = [self._transition(item).observation for item in history]
            observations.append(self._transition(final_id).next_observation)
            return observations

    def sampling_pace_s(self, transition_id: TransitionId) -> float:
        with self._lock:
            if not self.contains(transition_id):
                return float("inf")
            return float(self._sampling_pace[transition_id % self.capacity])

    def demo_flags(self, transition_ids: list[TransitionId]) -> list[bool]:
        with self._lock:
            return [
                self.contains(transition_id)
                and bool(self._demo_flags[transition_id % self.capacity])
                for transition_id in transition_ids
            ]

    def changes_since(
        self, revision: int | None
    ) -> tuple[int, list[tuple[TransitionId, TransitionId | None]] | None]:
        """Return append/eviction changes since a sampler's last observed revision."""

        with self._lock:
            if revision is None:
                return self._revision, None
            if revision == self._revision:
                return self._revision, []
            if not self._changes or revision < self._changes[0][0] - 1:
                return self._revision, None
            return self._revision, [
                (index, evicted)
                for change_revision, index, evicted in self._changes
                if change_revision > revision
            ]

    def _is_n_step_eligible_locked(self, transition_id: TransitionId, n_step: int) -> bool:
        if n_step < 1 or not self.contains(transition_id):
            return False
        candidate_id = transition_id
        for _ in range(n_step):
            if not self.contains(candidate_id):
                return False
            slot = candidate_id % self.capacity
            if self._terminated[slot] or self._truncated[slot]:
                return True
            candidate_id = int(self._next_ids[slot])
        return True

    def _predecessor_ids_locked(
        self, transition_id: TransitionId, n_step: int
    ) -> list[TransitionId]:
        result: list[TransitionId] = []
        candidate = transition_id
        for _ in range(n_step):
            if not self.contains(candidate):
                break
            result.append(candidate)
            candidate = int(self._previous_ids[candidate % self.capacity])
        return result
