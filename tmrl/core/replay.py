"""Reference replay storage and uniform sampling for projects and tests."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from math import ceil, floor, isfinite
from numbers import Number
from threading import RLock
from typing import Any, Protocol, TypeGuard, cast

import numpy as np
import torch

from tmrl.core.contracts import ReplayStore
from tmrl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, Transition, TransitionId
from tmrl.core.pytree import tree_collate, tree_map


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

    def changes_since(
        self, revision: int | None
    ) -> tuple[int, list[tuple[TransitionId, TransitionId | None]] | None]: ...


def _is_incremental_store(store: ReplayStore) -> TypeGuard[_IncrementalReplayStore]:
    return all(
        callable(getattr(store, name, None))
        for name in (
            "changes_since",
            "eligible_transition_ids",
            "is_n_step_eligible",
            "affected_n_step_starts",
        )
    ) and isinstance(getattr(store, "capacity", None), int)


class _IdPool:
    """Dense mutable ID set supporting O(1) membership changes and random draws."""

    def __init__(self) -> None:
        self.ids: list[TransitionId] = []
        self.positions: dict[TransitionId, int] = {}

    def add(self, transition_id: TransitionId) -> None:
        if transition_id not in self.positions:
            self.positions[transition_id] = len(self.ids)
            self.ids.append(transition_id)

    def remove(self, transition_id: TransitionId) -> None:
        position = self.positions.pop(transition_id, None)
        if position is None:
            return
        final = self.ids.pop()
        if position < len(self.ids):
            self.ids[position] = final
            self.positions[final] = position


class _FenwickTree:
    """Fixed-capacity prefix-sum tree for O(log N) proportional replay draws."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.values = np.zeros(size + 1, dtype=np.float64)
        self.leaves = np.zeros(size, dtype=np.float32)

    def set(self, index: int, value: float) -> None:
        delta = value - self.leaves[index]
        self.leaves[index] = value
        index += 1
        while index <= self.size:
            self.values[index] += delta
            index += index & -index

    @property
    def total(self) -> float:
        total = 0.0
        index = self.size
        while index:
            total += float(self.values[index])
            index -= index & -index
        return total

    def find(self, target: float) -> int:
        """Return the zero-based leaf containing a target in ``[0, total)``."""

        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= self.size and self.values[candidate] <= target:
                index = candidate
                target -= self.values[candidate]
            bit >>= 1
        return min(index, self.size - 1)


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
        self._observations: _TreeColumns | None = None
        self._actions: _TreeColumns | None = None
        self._next_overrides: dict[TransitionId, Any] = {}
        self._info: dict[TransitionId, Mapping[str, Any]] = {}
        self._episode_names: dict[int, str] = {}
        self._episode_codes_by_name: dict[str, int] = {}
        self._episode_steps: dict[int, dict[int, TransitionId]] = {}
        self._episode_terminal_steps: dict[int, int] = {}
        self._next_index = 0
        self._size = 0
        self._lock = RLock()
        self._revision = 0
        self._changes: deque[tuple[int, TransitionId, TransitionId | None]] = deque(
            maxlen=min(capacity, 65_536)
        )

    def append(self, transition: Transition) -> TransitionId:
        with self._lock:
            episode_code = self._episode_code(transition.episode_id)
            previous_id = self._previous_transition(transition, episode_code)
            transition_id = self._next_index
            slot = transition_id % self.capacity
            evicted = int(self._ids[slot]) if self._ids[slot] >= 0 else None
            if evicted is not None:
                self._info.pop(evicted, None)
                self._next_overrides.pop(evicted, None)
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
            self._next_overrides[transition_id] = transition.next_observation
            if transition.info:
                self._info[transition_id] = transition.info
            self._link_previous(previous_id, transition_id, transition.observation)
            if episode_code >= 0 and transition.step is not None:
                self._register_episode_step(episode_code, transition.step, transition_id)
            self._next_index += 1
            self._size = min(self.capacity, self._size + 1)
            self._revision += 1
            self._changes.append((self._revision, transition_id, evicted))
            return transition_id

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
        code = len(self._episode_names)
        self._episode_names[code] = episode_id
        self._episode_codes_by_name[episode_id] = code
        return code

    def _previous_transition(self, transition: Transition, episode_code: int) -> TransitionId:
        if episode_code < 0 or transition.step is None:
            candidate = self._next_index - 1
            return candidate if self.contains(candidate) else -1
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
        self._next_ids[previous_slot] = transition_id
        self._previous_ids[transition_id % self.capacity] = previous_id
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

    def _transition(self, transition_id: TransitionId) -> Transition:
        assert self._observations is not None
        assert self._actions is not None
        slot = transition_id % self.capacity
        next_observation = self._next_overrides.get(transition_id)
        if next_observation is None:
            next_id = int(self._next_ids[slot])
            if not self.contains(next_id):
                raise RuntimeError(f"Transition {transition_id} has no next observation")
            next_observation = self._observations.read(next_id % self.capacity)
        episode_code = int(self._episode_codes[slot])
        step = int(self._steps[slot])
        return Transition(
            observation=self._observations.read(slot),
            action=self._actions.read(slot),
            reward=float(self._rewards[slot]),
            next_observation=next_observation,
            terminated=bool(self._terminated[slot]),
            truncated=bool(self._truncated[slot]),
            info=self._info.get(transition_id, {}),
            episode_id=self._episode_names.get(episode_code),
            step=step if step >= 0 else None,
        )

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
            self._episode_names = {
                int(code): str(name)
                for code, name in cast(Mapping[Any, Any], state.get("episode_names", {})).items()
            }
            self._episode_codes_by_name = {name: code for code, name in self._episode_names.items()}
            self._next_overrides = dict(state.get("next_overrides", {}))
            self._info = dict(state.get("info", {}))
            self._rebuild_episode_steps()
            self._revision += 1
            self._changes.clear()

    def _reset_arrays(self) -> None:
        self._ids.fill(-1)
        self._episode_codes.fill(-1)
        self._steps.fill(-1)
        self._previous_ids.fill(-1)
        self._next_ids.fill(-1)
        self._observations = None
        self._actions = None
        self._next_overrides.clear()
        self._info.clear()
        self._episode_steps.clear()
        self._episode_terminal_steps.clear()

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


class UniformSampler:
    """Reference sampler suitable for custom project templates and smoke tests."""

    def __init__(self, pipeline: Any, seed: int = 0) -> None:
        self.pipeline = pipeline
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("UniformSampler supports sequence_length=1; use a sequence sampler")
        fast_sample = getattr(store, "sample_eligible_ids", None)
        if callable(fast_sample):
            transition_ids = fast_sample(request.n_step, request.batch_size, self._rng)
        else:
            transition_ids = _eligible_n_step_ids(store, request)
            if len(transition_ids) < request.batch_size:
                raise RuntimeError(
                    f"Need {request.batch_size} complete n-step transitions, replay has "
                    f"{len(transition_ids)}"
                )
            transition_ids = self._rng.sample(transition_ids, request.batch_size)
        return _make_batch(store, self.pipeline, transition_ids, request)

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update  # Uniform sampling intentionally ignores priority feedback.

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class PrioritizedSampler:
    """Array-backed proportional PER with normalized importance weights."""

    thread_safe_prefetch = True

    def __init__(
        self,
        pipeline: Any,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if alpha < 0.0 or beta < 0.0 or priority_epsilon <= 0.0:
            raise ValueError("alpha/beta must be non-negative and priority_epsilon positive")
        self.pipeline = pipeline
        self.alpha = alpha
        self.beta = beta
        self.priority_epsilon = priority_epsilon
        self._fallback_priorities: dict[int, float] = {}
        self._priorities = np.empty(0, dtype=np.float32)
        self._slot_ids = np.empty(0, dtype=np.int64)
        self._rng = random.Random(seed)
        self._active_count = 0
        self._tree: _FenwickTree | None = None
        self._replay_revision: int | None = None
        self._n_step: int | None = None
        self._maximum_priority = 1.0
        self._lock = RLock()

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("PrioritizedSampler supports sequence_length=1")
        if _is_incremental_store(store):
            return self._sample_incrementally(store, request)
        chosen, normalized, beta = self._sample_fallback(store, request)
        return _make_batch(
            store,
            self.pipeline,
            chosen,
            request,
            importance_weights=normalized,
            metadata={"sampling": "prioritized", "beta": beta},
        )

    def _sample_fallback(
        self, store: ReplayStore, request: BatchRequest
    ) -> tuple[list[TransitionId], tuple[float, ...], float]:
        with self._lock:
            transition_ids = _eligible_n_step_ids(store, request)
            if len(transition_ids) < request.batch_size:
                raise RuntimeError(
                    f"Need {request.batch_size} transitions, replay has {len(transition_ids)}"
                )
            self._synchronize_fallback(transition_ids)
            scaled = [
                self._fallback_priorities[transition_id] ** self.alpha
                for transition_id in transition_ids
            ]
            total = sum(scaled)
            probabilities = (
                [weight / total for weight in scaled]
                if total > 0.0
                else [1 / len(transition_ids)] * len(transition_ids)
            )
            chosen = self._rng.choices(transition_ids, weights=probabilities, k=request.batch_size)
            by_id = dict(zip(transition_ids, probabilities, strict=True))
            beta = self.beta if request.beta is None else request.beta
            weights = [
                (len(transition_ids) * by_id[transition_id]) ** (-beta) for transition_id in chosen
            ]
            maximum = max(weights)
            return chosen, tuple(weight / maximum for weight in weights), beta

    def update_priorities(self, update: PriorityUpdate) -> None:
        with self._lock:
            for transition_id, priority in zip(
                update.transition_ids, update.priorities, strict=True
            ):
                value = abs(float(priority)) + self.priority_epsilon
                if not isfinite(value):
                    raise ValueError("PER priorities must be finite")
                self._maximum_priority = max(self._maximum_priority, value)
                if self._tree is None:
                    if transition_id in self._fallback_priorities:
                        self._fallback_priorities[transition_id] = value
                    continue
                slot = transition_id % self._tree.size
                if self._slot_ids[slot] == transition_id and self._tree.leaves[slot] > 0.0:
                    self._priorities[slot] = value
                    self._tree.set(slot, value**self.alpha)

    def _synchronize_fallback(self, transition_ids: list[TransitionId]) -> None:
        active = set(transition_ids)
        self._fallback_priorities = {
            index: priority
            for index, priority in self._fallback_priorities.items()
            if index in active
        }
        maximum = max(self._fallback_priorities.values(), default=1.0)
        for index in active:
            self._fallback_priorities.setdefault(index, maximum)

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format": "array-per-v1",
                "priorities": self._priorities.copy(),
                "slot_ids": self._slot_ids.copy(),
                "fallback_priorities": dict(self._fallback_priorities),
                "maximum_priority": self._maximum_priority,
                "rng": self._rng.getstate(),
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format") == "array-per-v1":
            self._priorities = np.asarray(state["priorities"], dtype=np.float32)
            self._slot_ids = np.asarray(state["slot_ids"], dtype=np.int64)
            fallback = cast(Mapping[Any, Any], state.get("fallback_priorities", {}))
            self._fallback_priorities = {int(key): float(value) for key, value in fallback.items()}
            self._maximum_priority = float(state["maximum_priority"])
        else:
            legacy = cast(Mapping[Any, Any], state["priorities"])
            self._fallback_priorities = {int(key): float(value) for key, value in legacy.items()}
            self._maximum_priority = max(self._fallback_priorities.values(), default=1.0)
            self._priorities = np.empty(0, dtype=np.float32)
            self._slot_ids = np.empty(0, dtype=np.int64)
        self._active_count = 0
        self._tree = None
        self._replay_revision = None
        self._n_step = None
        self._rng.setstate(state["rng"])

    def _sample_incrementally(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> TrainingBatch:
        transition_ids, normalized, beta = self._sample_incremental_ids(store, request)
        return _make_batch(
            store,
            self.pipeline,
            transition_ids,
            request,
            importance_weights=normalized,
            metadata={"sampling": "prioritized", "beta": beta},
        )

    def _sample_incremental_ids(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> tuple[list[TransitionId], tuple[float, ...], float]:
        with self._lock:
            self._synchronize_incremental_store(store, request.n_step)
            if self._active_count < request.batch_size:
                raise RuntimeError(
                    f"Need {request.batch_size} transitions, replay has {self._active_count}"
                )
            assert self._tree is not None
            total = self._tree.total
            if total <= 0.0:
                raise RuntimeError("Prioritized replay has no positive sampling mass")
            transition_ids: list[TransitionId] = []
            probabilities: list[float] = []
            for _ in range(request.batch_size):
                slot = self._tree.find(self._rng.random() * total)
                transition_id = int(self._slot_ids[slot])
                if transition_id < 0:
                    raise RuntimeError(
                        "Prioritized replay tree is out of sync with active transitions"
                    )
                transition_ids.append(transition_id)
                probabilities.append(float(self._tree.leaves[slot]) / total)
            beta = self.beta if request.beta is None else request.beta
            weights = [
                (self._active_count * probability) ** (-beta) for probability in probabilities
            ]
            maximum = max(weights)
            return transition_ids, tuple(weight / maximum for weight in weights), beta

    def _synchronize_incremental_store(self, store: _IncrementalReplayStore, n_step: int) -> None:
        capacity = store.capacity
        if self._tree is None or self._n_step != n_step or self._tree.size != capacity:
            self._tree = _FenwickTree(capacity)
            if self._slot_ids.shape != (capacity,):
                self._slot_ids = np.full(capacity, -1, dtype=np.int64)
                self._priorities = np.zeros(capacity, dtype=np.float32)
            self._active_count = 0
            self._replay_revision = None
            self._n_step = n_step
        revision, changes = store.changes_since(self._replay_revision)
        if changes is None:
            self._active_count = 0
            self._tree = _FenwickTree(capacity)
            for transition_id in store.eligible_transition_ids(n_step):
                self._activate(transition_id)
        else:
            for appended, evicted in changes:
                if evicted is not None:
                    self._deactivate(evicted)
                for candidate in store.affected_n_step_starts(appended, n_step):
                    if store.contains(candidate) and store.is_n_step_eligible(candidate, n_step):
                        self._activate(candidate)
                    else:
                        self._deactivate(candidate)
        self._replay_revision = revision

    def _activate(self, transition_id: TransitionId) -> None:
        assert self._tree is not None
        slot = transition_id % self._tree.size
        if self._slot_ids[slot] == transition_id and self._tree.leaves[slot] > 0.0:
            return
        if self._tree.leaves[slot] > 0.0:
            self._active_count -= 1
        priority = (
            float(self._priorities[slot])
            if self._slot_ids[slot] == transition_id and self._priorities[slot] > 0.0
            else self._maximum_priority
        )
        self._priorities[slot] = priority
        self._slot_ids[slot] = transition_id
        self._tree.set(slot, priority**self.alpha)
        self._active_count += 1

    def _deactivate(self, transition_id: TransitionId) -> None:
        assert self._tree is not None
        slot = transition_id % self._tree.size
        if self._slot_ids[slot] == transition_id and self._tree.leaves[slot] > 0.0:
            self._active_count -= 1
            self._tree.set(slot, 0.0)


class SequenceSampler:
    """Samples only contiguous transitions from one identified episode."""

    def __init__(self, pipeline: Any, sequence_length: int, seed: int = 0) -> None:
        if sequence_length < 2:
            raise ValueError("SequenceSampler requires sequence_length >= 2")
        self.pipeline = pipeline
        self.sequence_length = sequence_length
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        length = request.sequence_length if request.sequence_length > 1 else self.sequence_length
        ordered = store.available_ids()
        if len(ordered) < length:
            raise RuntimeError(f"Need at least {length} transitions for sequence sampling")
        transitions = store.get(ordered)
        available = dict(zip(ordered, transitions, strict=True))
        windows: list[list[int]] = []
        for offset in range(len(ordered) - length + 1):
            indices = ordered[offset : offset + length]
            values = transitions[offset : offset + length]
            if _is_contiguous_episode(indices, values) and all(
                _has_complete_n_step(transition_id, available, request) for transition_id in indices
            ):
                windows.append(indices)
        if len(windows) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} valid sequences, replay has {len(windows)}"
            )
        selected = self._rng.sample(windows, request.batch_size)
        flattened = [transition_id for window in selected for transition_id in window]
        batch = _make_batch(
            store,
            self.pipeline,
            flattened,
            request,
            metadata={"sampling": "sequence", "sequence_length": length},
        )
        return replace(
            batch,
            data=_reshape_sequence_batch(batch.data, request.batch_size, length),
            observations=_reshape_sequence_batch(batch.observations, request.batch_size, length),
            actions=_reshape_sequence_batch(batch.actions, request.batch_size, length),
            rewards=_reshape_sequence_batch(batch.rewards, request.batch_size, length),
            next_observations=_reshape_sequence_batch(
                batch.next_observations, request.batch_size, length
            ),
            terminated=_reshape_sequence_batch(batch.terminated, request.batch_size, length),
            truncated=_reshape_sequence_batch(batch.truncated, request.batch_size, length),
            bootstrap_discounts=_reshape_sequence_batch(
                batch.bootstrap_discounts, request.batch_size, length
            ),
            masks=torch.ones((request.batch_size, length), dtype=torch.bool),
        )

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class DemoMixSampler:
    """Uniform sampler with explicit, bounded demonstration mixing."""

    def __init__(
        self,
        pipeline: Any,
        *,
        min_demo_fraction: float = 0.0,
        max_demo_fraction: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= min_demo_fraction <= max_demo_fraction <= 1.0:
            raise ValueError("demo fractions must satisfy 0 <= min <= max <= 1")
        self.pipeline = pipeline
        self.min_demo_fraction = min_demo_fraction
        self.max_demo_fraction = max_demo_fraction
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("DemoMixSampler supports sequence_length=1")
        transition_ids = _eligible_n_step_ids(store, request)
        if len(transition_ids) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} transitions, replay has {len(transition_ids)}"
            )
        items = dict(zip(transition_ids, store.get(transition_ids), strict=True))
        demos = [index for index, value in items.items() if _is_demo(value.info)]
        demo_indices = set(demos)
        online = [
            transition_id for transition_id in transition_ids if transition_id not in demo_indices
        ]
        minimum = ceil(self.min_demo_fraction * request.batch_size)
        maximum = floor(self.max_demo_fraction * request.batch_size)
        demo_count = min(maximum, len(demos))
        if demo_count < minimum:
            raise RuntimeError(
                f"Need {minimum} demo transitions for this batch, replay has {len(demos)}"
            )
        online_count = request.batch_size - demo_count
        if len(online) < online_count:
            raise RuntimeError(f"Need {online_count} online transitions, replay has {len(online)}")
        chosen = self._rng.sample(demos, demo_count) + self._rng.sample(online, online_count)
        self._rng.shuffle(chosen)
        return _make_batch(
            store,
            self.pipeline,
            chosen,
            request,
            metadata={"sampling": "demo_mix", "demo_fraction": demo_count / request.batch_size},
        )

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


def _is_contiguous_episode(indices: list[TransitionId], transitions: list[Transition]) -> bool:
    episode_id = transitions[0].episode_id
    if episode_id is None:
        return False
    for previous_index, current_index, previous, current in zip(
        indices[:-1], indices[1:], transitions[:-1], transitions[1:], strict=True
    ):
        if current.episode_id != episode_id or current_index != previous_index + 1:
            return False
        if previous.step is not None and current.step != previous.step + 1:
            return False
        if previous.terminated or previous.truncated:
            return False
    return True


def _eligible_n_step_ids(store: ReplayStore, request: BatchRequest) -> list[TransitionId]:
    """Return starts whose target is complete or ends at a real episode boundary."""

    incremental = getattr(store, "eligible_transition_ids", None)
    if callable(incremental):
        return cast(list[TransitionId], incremental(request.n_step))
    transition_ids = store.available_ids()
    available = dict(zip(transition_ids, store.get(transition_ids), strict=True))
    return [
        transition_id
        for transition_id in transition_ids
        if _has_complete_n_step(transition_id, available, request)
    ]


def _has_complete_n_step(
    transition_id: TransitionId,
    available: Mapping[TransitionId, Transition],
    request: BatchRequest,
) -> bool:
    """Reject live replay tails, whose future rewards have not arrived yet."""

    first = available[transition_id]
    for offset in range(request.n_step):
        candidate = available.get(transition_id + offset)
        if candidate is None or candidate.episode_id != first.episode_id:
            return False
        if (
            candidate.step is not None
            and first.step is not None
            and candidate.step != first.step + offset
        ):
            return False
        if candidate.terminated or candidate.truncated:
            return True
    return True


def _is_demo(info: Mapping[str, Any]) -> bool:
    return bool(info.get("is_demo", False) or info.get("source") == "demo")


def _reshape_sequence_batch(value: Any, batch_size: int, sequence_length: int) -> Any:
    """Restore ``(batch, time, ...)`` layout after replay gathers contiguous IDs."""

    flattened_size = batch_size * sequence_length

    def reshape(leaf: Any) -> Any:
        if (
            hasattr(leaf, "shape")
            and hasattr(leaf, "reshape")
            and leaf.shape[:1] == (flattened_size,)
        ):
            return leaf.reshape(batch_size, sequence_length, *leaf.shape[1:])
        return leaf

    return tree_map(reshape, value)


def _make_batch(
    store: ReplayStore,
    pipeline: Any,
    transition_ids: list[TransitionId],
    request: BatchRequest,
    *,
    importance_weights: tuple[float, ...] | None = None,
    masks: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingBatch:
    """Build a batch whose n-step returns are derived from replay order, not batch order."""

    requested_ids: list[TransitionId] = []
    horizons: dict[TransitionId, list[TransitionId]] = {}
    seen: set[TransitionId] = set()
    for transition_id in transition_ids:
        resolver = getattr(store, "n_step_ids", None)
        horizon = (
            cast(list[TransitionId], resolver(transition_id, request.n_step))
            if callable(resolver)
            else [transition_id + offset for offset in range(request.n_step)]
        )
        horizons[transition_id] = horizon
        for candidate in horizon:
            if candidate not in seen and store.contains(candidate):
                seen.add(candidate)
                requested_ids.append(candidate)
    available = dict(zip(requested_ids, store.get(requested_ids), strict=True))
    n_step = [
        _n_step_transition(
            transition_id,
            available,
            request,
            horizon=horizons[transition_id],
        )
        for transition_id in transition_ids
    ]
    transitions = [item[0] for item in n_step]
    discounts = [item[1] for item in n_step]
    data = pipeline.collate(transitions)
    standard = (
        data
        if isinstance(data, Mapping)
        and data.get("_tmrl_batch_collated") is True
        and {
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
        }.issubset(data)
        else None
    )
    batch_data = (
        {
            key: value
            for key, value in standard.items()
            if key
            not in {
                "_tmrl_batch_collated",
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminated",
                "truncated",
            }
        }
        if standard is not None
        else data
    )
    return TrainingBatch(
        data=batch_data,
        observations=standard["observations"]
        if standard is not None
        else tree_collate([item.observation for item in transitions]),
        actions=standard["actions"]
        if standard is not None
        else tree_collate([item.action for item in transitions]),
        rewards=standard["rewards"]
        if standard is not None
        else tree_collate([item.reward for item in transitions]),
        next_observations=standard["next_observations"]
        if standard is not None
        else tree_collate([item.next_observation for item in transitions]),
        terminated=standard["terminated"]
        if standard is not None
        else tree_collate([item.terminated for item in transitions]),
        truncated=standard["truncated"]
        if standard is not None
        else tree_collate([item.truncated for item in transitions]),
        bootstrap_discounts=tree_collate(discounts),
        transition_ids=transition_ids,
        importance_weights=tree_collate(importance_weights)
        if importance_weights is not None
        else None,
        masks=masks,
        metadata=dict(metadata or {}),
    )


def _n_step_transition(
    transition_id: TransitionId,
    available: Mapping[TransitionId, Transition],
    request: BatchRequest,
    *,
    horizon: list[TransitionId] | None = None,
) -> tuple[Transition, float]:
    """Return the episode-safe discounted n-step transition beginning at ``transition_id``."""

    first = available[transition_id]
    current = first
    reward = 0.0
    discount = 1.0
    effective_steps = 0
    terminated = False
    truncated = False
    ordered_ids = horizon or [transition_id + offset for offset in range(request.n_step)]
    for offset, current_id in enumerate(ordered_ids):
        candidate = available.get(current_id)
        if candidate is None or candidate.episode_id != first.episode_id:
            break
        if (
            candidate.step is not None
            and first.step is not None
            and candidate.step != first.step + offset
        ):
            break
        current = candidate
        reward += discount * candidate.reward
        effective_steps += 1
        terminated = candidate.terminated
        truncated = candidate.truncated
        if terminated or truncated:
            break
        discount *= request.gamma
    if effective_steps == 0:
        raise RuntimeError(f"Transition {transition_id} is no longer available")
    bootstrap_discount = 0.0 if terminated else request.gamma**effective_steps
    return (
        Transition(
            observation=first.observation,
            action=first.action,
            reward=reward,
            next_observation=current.next_observation,
            terminated=terminated,
            truncated=truncated,
            info=first.info,
            episode_id=first.episode_id,
            step=first.step,
        ),
        bootstrap_discount,
    )
