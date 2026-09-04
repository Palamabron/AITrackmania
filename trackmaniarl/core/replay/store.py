"""Columnar in-memory replay storage with stable transition IDs."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import isfinite
from threading import RLock
from typing import Any, cast

import numpy as np

import trackmaniarl.core.replay.store_index as store_index
import trackmaniarl.core.replay.store_materialization as store_materialization
import trackmaniarl.core.replay.store_pace as store_pace
import trackmaniarl.core.replay.store_queries as store_queries
import trackmaniarl.core.replay.store_state as store_state
from trackmaniarl.core.data import BatchRequest, Transition, TransitionId
from trackmaniarl.core.pytree import tree_snapshot
from trackmaniarl.core.replay.store_support import (
    _IncrementalReplayStore as _IncrementalReplayStore,
)
from trackmaniarl.core.replay.store_support import _is_demo as _is_demo
from trackmaniarl.core.replay.store_support import (
    _is_incremental_store as _is_incremental_store,
)
from trackmaniarl.core.replay.store_support import _ReplayChange, _TreeColumns


@dataclass(frozen=True, slots=True)
class _EvictedSlot:
    transition_id: TransitionId
    previous_id: TransitionId | None
    next_id: TransitionId | None
    resurrected: Transition | None


@dataclass(frozen=True, slots=True)
class _AppendContext:
    transition_id: TransitionId
    slot: int
    episode_code: int
    previous_id: TransitionId
    evicted: _EvictedSlot | None


class InMemoryReplayStore:
    """Preallocated columnar FIFO replay with stable transition IDs."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._initialize_columns()
        self._initialize_episode_index()
        self._initialize_runtime_state()

    def _initialize_columns(self) -> None:
        capacity = self.capacity
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

    def _initialize_episode_index(self) -> None:
        self._episode_names: dict[int, str] = {}
        self._episode_codes_by_name: dict[str, int] = {}
        self._episode_steps: dict[int, dict[int, TransitionId]] = {}
        self._episode_terminal_steps: dict[int, int] = {}
        self._episode_refcounts: dict[int, int] = {}
        self._episode_sampling_paces: dict[str, float] = {}
        self._next_episode_code = 0
        self._demo_flags = np.zeros(self.capacity, dtype=np.bool_)
        self._demo_count = 0

    def _initialize_runtime_state(self) -> None:
        self._next_index = 0
        self._size = 0
        self._lock = RLock()
        self._revision = 0
        self._changes: deque[tuple[int, _ReplayChange]] = deque(maxlen=min(self.capacity, 65_536))

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
        evicted = self._evict_slot(slot, episode_code)
        context = _AppendContext(transition_id, slot, episode_code, previous_id, evicted)
        self._allocate_columns(transition)
        self._write_transition(context, transition)
        self._write_transition_metadata(context, transition)
        self._complete_append(context, transition)
        resurrected = None if evicted is None else evicted.resurrected
        return transition_id, resurrected

    def _evict_slot(self, slot: int, retained_episode_code: int) -> _EvictedSlot | None:
        evicted = int(self._ids[slot]) if self._ids[slot] >= 0 else None
        if evicted is None:
            return None
        previous_id = _optional_transition_id(self._previous_ids[slot])
        next_id = _optional_transition_id(self._next_ids[slot])
        resurrected = self._protected_demo(evicted, slot)
        self._info.pop(evicted, None)
        self._next_overrides.pop(evicted, None)
        self._release_episode_reference(int(self._episode_codes[slot]), retained_episode_code)
        return _EvictedSlot(evicted, previous_id, next_id, resurrected)

    def _protected_demo(self, evicted: TransitionId, slot: int) -> Transition | None:
        if not self._demo_flags[slot]:
            return None
        if self._demo_count * 2 >= self.capacity:
            raise RuntimeError("replay capacity is too small to protect demonstration transitions")
        self._demo_count -= 1
        return self._resurrectable_transition(evicted, slot)

    def _write_transition(self, context: _AppendContext, transition: Transition) -> None:
        assert self._observations is not None
        assert self._actions is not None
        slot = context.slot
        self._observations.write(slot, transition.observation)
        self._actions.write(slot, transition.action)
        self._ids[slot] = context.transition_id
        self._rewards[slot] = transition.reward
        self._terminated[slot] = transition.terminated
        self._truncated[slot] = transition.truncated
        self._episode_codes[slot] = context.episode_code
        self._steps[slot] = transition.step if transition.step is not None else -1
        self._previous_ids[slot] = context.previous_id
        self._next_ids[slot] = -1

    def _write_transition_metadata(self, context: _AppendContext, transition: Transition) -> None:
        transition_info = cast(dict[str, Any], tree_snapshot(dict(transition.info)))
        self._sampling_pace[context.slot] = store_pace.transition_sampling_pace(
            self, transition.episode_id, transition_info
        )
        is_demo = _is_demo(transition_info)
        self._demo_flags[context.slot] = is_demo
        self._demo_count += int(is_demo)
        if context.episode_code >= 0:
            count = self._episode_refcounts.get(context.episode_code, 0)
            self._episode_refcounts[context.episode_code] = count + 1
        self._next_overrides[context.transition_id] = tree_snapshot(transition.next_observation)
        if transition_info:
            self._info[context.transition_id] = transition_info

    def _complete_append(self, context: _AppendContext, transition: Transition) -> None:
        self._link_previous(context.previous_id, context.transition_id, transition.observation)
        if context.episode_code >= 0 and transition.step is not None:
            self._register_episode_step(
                context.episode_code, transition.step, context.transition_id
            )
        self._next_index += 1
        self._size = min(self.capacity, self._size + 1)
        self._revision += 1
        evicted = context.evicted
        change = _ReplayChange(
            appended=context.transition_id,
            evicted=None if evicted is None else evicted.transition_id,
            evicted_previous=None if evicted is None else evicted.previous_id,
            evicted_next=None if evicted is None else evicted.next_id,
        )
        self._changes.append((self._revision, change))

    def _resurrectable_transition(self, transition_id: TransitionId, slot: int) -> Transition:
        resurrected = self._transition(transition_id)
        pace = float(self._sampling_pace[slot])
        if not isfinite(pace):
            return resurrected
        return replace(
            resurrected,
            info={**resurrected.info, "sampling/projected_lap_time_s": pace},
        )

    def _release_episode_reference(self, episode_code: int, retained_episode_code: int) -> None:
        if episode_code < 0:
            return
        remaining = self._episode_refcounts.get(episode_code, 0) - 1
        if remaining > 0:
            self._episode_refcounts[episode_code] = remaining
            return
        self._episode_refcounts.pop(episode_code, None)
        name = self._episode_names.get(episode_code)
        if name is not None and episode_code != retained_episode_code:
            self._episode_names.pop(episode_code)
            self._episode_codes_by_name.pop(name, None)
            self._episode_sampling_paces.pop(name, None)
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
        return store_index._episode_code(self, episode_id)

    def _previous_transition(self, transition: Transition, episode_code: int) -> TransitionId:
        return store_index._previous_transition(self, transition, episode_code)

    def _register_episode_step(
        self, episode_code: int, step: int, transition_id: TransitionId
    ) -> None:
        registration = store_index._EpisodeStepRegistration(episode_code, step, transition_id)
        store_index._register_episode_step(self, registration)

    def _episode_step(self, steps: dict[int, TransitionId], step: int) -> TransitionId:
        return store_index._episode_step(self, steps, step)

    def _release_completed_episode(self, episode_code: int) -> None:
        store_index._release_completed_episode(self, episode_code)

    def _link_previous(
        self,
        previous_id: TransitionId,
        transition_id: TransitionId,
        observation: Any,
    ) -> None:
        link = store_index._TransitionLink(previous_id, transition_id, observation)
        store_index._link_previous(self, link)

    @staticmethod
    def _tree_equal(left: Any, right: Any) -> bool:
        return store_index._tree_equal(left, right)

    def get(self, transition_ids: list[TransitionId]) -> list[Transition]:
        return store_materialization.get(self, transition_ids)

    def materialize_n_step(
        self, transition_ids: list[TransitionId], request: BatchRequest
    ) -> tuple[list[Transition], list[float]]:
        return store_materialization.materialize_n_step(self, transition_ids, request)

    def _materialize_n_step_locked(
        self, transition_id: TransitionId, request: BatchRequest
    ) -> tuple[Transition, float]:
        return store_materialization._materialize_n_step_locked(self, transition_id, request)

    def _transition(self, transition_id: TransitionId) -> Transition:
        return store_materialization._transition(self, transition_id)

    def _next_observation(self, transition_id: TransitionId, slot: int) -> Any:
        return store_materialization._next_observation(self, transition_id, slot)

    def available_ids(self) -> list[TransitionId]:
        with self._lock:
            return list(range(self._next_index - self._size, self._next_index))

    def contains(self, transition_id: TransitionId) -> bool:
        if transition_id < 0:
            return False
        return bool(self._ids[transition_id % self.capacity] == transition_id)

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def state_dict(self) -> dict[str, Any]:
        return store_state.state_dict(self)

    def _snapshot_slots(
        self,
    ) -> slice | np.ndarray[Any, np.dtype[np.int64]]:
        return store_state._snapshot_slots(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        store_state.load_state_dict(self, state)

    def _reset_arrays(self) -> None:
        store_state._reset_arrays(self)

    def _rebuild_reference_state(self) -> None:
        store_state._rebuild_reference_state(self)

    def _rebuild_episode_steps(self) -> None:
        store_state._rebuild_episode_steps(self)

    def eligible_transition_ids(self, n_step: int) -> list[TransitionId]:
        return store_queries.eligible_transition_ids(self, n_step)

    def sample_eligible_ids(
        self, n_step: int, batch_size: int, rng: random.Random
    ) -> list[TransitionId]:
        request = store_queries._EligibleSample(self, n_step, batch_size, rng)
        return store_queries.sample_eligible_ids(request)

    def is_n_step_eligible(self, transition_id: TransitionId, n_step: int) -> bool:
        return store_queries.is_n_step_eligible(self, transition_id, n_step)

    def n_step_ids(self, transition_id: TransitionId, n_step: int) -> list[TransitionId]:
        return store_queries.n_step_ids(self, transition_id, n_step)

    def affected_n_step_starts(
        self, transition_id: TransitionId, n_step: int
    ) -> list[TransitionId]:
        return store_queries.affected_n_step_starts(self, transition_id, n_step)

    def history_ids(self, transition_id: TransitionId, sequence_length: int) -> list[TransitionId]:
        return store_queries.history_ids(self, transition_id, sequence_length)

    def next_history_observations(
        self, transition_id: TransitionId, n_step: int, sequence_length: int
    ) -> list[Any]:
        request = store_queries._NextHistoryRequest(self, transition_id, n_step, sequence_length)
        return store_queries.next_history_observations(request)

    def sampling_pace_s(self, transition_id: TransitionId) -> float:
        return store_queries.sampling_pace_s(self, transition_id)

    def label_episode_sampling_pace(self, episode_id: str, finish_time_s: float) -> int:
        return store_pace.label_episode_sampling_pace(self, episode_id, finish_time_s)

    def demo_flags(self, transition_ids: list[TransitionId]) -> list[bool]:
        return store_queries.demo_flags(self, transition_ids)

    def changes_since(self, revision: int | None) -> tuple[int, list[_ReplayChange] | None]:
        return store_queries.changes_since(self, revision)

    def _is_n_step_eligible_locked(self, transition_id: TransitionId, n_step: int) -> bool:
        return store_queries._is_n_step_eligible_locked(self, transition_id, n_step)

    def _predecessor_ids_locked(
        self, transition_id: TransitionId, n_step: int
    ) -> list[TransitionId]:
        return store_queries._predecessor_ids_locked(self, transition_id, n_step)


def _optional_transition_id(value: np.int64) -> TransitionId | None:
    return int(value) if value >= 0 else None
