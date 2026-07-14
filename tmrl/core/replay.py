"""Reference replay storage and uniform sampling for projects and tests."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from math import ceil, floor, isfinite
from threading import RLock
from typing import Any, Protocol, TypeGuard, cast

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

    def changes_since(
        self, revision: int | None
    ) -> tuple[int, list[tuple[TransitionId, TransitionId | None]] | None]: ...


def _is_incremental_store(store: ReplayStore) -> TypeGuard[_IncrementalReplayStore]:
    return all(
        callable(getattr(store, name, None))
        for name in ("changes_since", "eligible_transition_ids", "is_n_step_eligible")
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
        self.values = [0.0] * (size + 1)
        self.leaves = [0.0] * size

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
            total += self.values[index]
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


class InMemoryReplayStore:
    """Bounded FIFO replay store; sampling policy is intentionally external."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._order: deque[TransitionId] = deque()
        self._items: dict[TransitionId, Transition] = {}
        self._next_index = 0
        self._lock = RLock()
        self._eligible: dict[int, _IdPool] = {}
        self._revision = 0
        self._changes: deque[tuple[int, TransitionId, TransitionId | None]] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> TransitionId:
        with self._lock:
            index = self._next_index
            self._next_index += 1
            evicted: TransitionId | None = None
            if len(self._order) == self.capacity:
                evicted = self._order.popleft()
                self._items.pop(evicted)
            self._order.append(index)
            self._items[index] = transition
            self._revision += 1
            self._changes.append((self._revision, index, evicted))
            for n_step, pool in self._eligible.items():
                if evicted is not None:
                    pool.remove(evicted)
                for candidate in range(index - n_step + 1, index + 1):
                    self._refresh_eligibility_locked(pool, candidate, n_step)
            return index

    def get(self, transition_ids: list[TransitionId]) -> list[Transition]:
        with self._lock:
            missing = [
                transition_id
                for transition_id in transition_ids
                if transition_id not in self._items
            ]
            if missing:
                raise KeyError(f"Replay transitions no longer available: {missing[:3]}")
            return [self._items[transition_id] for transition_id in transition_ids]

    def available_ids(self) -> list[TransitionId]:
        with self._lock:
            return list(self._order)

    def available_indices(self) -> list[TransitionId]:
        """Compatibility spelling; IDs are never reused after eviction."""

        return self.available_ids()

    def contains(self, transition_id: TransitionId) -> bool:
        with self._lock:
            return transition_id in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "order": list(self._order),
                "items": dict(self._items),
                "next_index": self._next_index,
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self._order = deque(state["order"])
            self._items = dict(state["items"])
            self._next_index = int(state["next_index"])
            self._eligible.clear()
            self._revision += 1
            self._changes.clear()

    def eligible_transition_ids(self, n_step: int) -> list[TransitionId]:
        """Return complete n-step starts, incrementally maintained on append."""

        with self._lock:
            return list(self._eligible_pool_locked(n_step).ids)

    def sample_eligible_ids(
        self, n_step: int, batch_size: int, rng: random.Random
    ) -> list[TransitionId]:
        """Sample complete n-step starts without scanning the replay buffer."""

        with self._lock:
            pool = self._eligible_pool_locked(n_step)
            if len(pool.ids) < batch_size:
                raise RuntimeError(
                    f"Need {batch_size} complete n-step transitions, replay has {len(pool.ids)}"
                )
            return rng.sample(pool.ids, batch_size)

    def is_n_step_eligible(self, transition_id: TransitionId, n_step: int) -> bool:
        with self._lock:
            return self._is_n_step_eligible_locked(transition_id, n_step)

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

    def _eligible_pool_locked(self, n_step: int) -> _IdPool:
        if n_step < 1:
            raise ValueError("n_step must be positive")
        pool = self._eligible.get(n_step)
        if pool is None:
            pool = _IdPool()
            for transition_id in self._order:
                self._refresh_eligibility_locked(pool, transition_id, n_step)
            self._eligible[n_step] = pool
        return pool

    def _refresh_eligibility_locked(
        self, pool: _IdPool, transition_id: TransitionId, n_step: int
    ) -> None:
        if self._is_n_step_eligible_locked(transition_id, n_step):
            pool.add(transition_id)
        else:
            pool.remove(transition_id)

    def _is_n_step_eligible_locked(self, transition_id: TransitionId, n_step: int) -> bool:
        first = self._items.get(transition_id)
        if first is None:
            return False
        for offset in range(n_step):
            candidate = self._items.get(transition_id + offset)
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
    """Proportional PER separated from storage, with normalized IS weights."""

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
        self._priorities: dict[int, float] = {}
        self._rng = random.Random(seed)
        self._active_ids: set[TransitionId] = set()
        self._slot_ids: list[TransitionId | None] = []
        self._tree: _FenwickTree | None = None
        self._replay_revision: int | None = None
        self._n_step: int | None = None
        self._maximum_priority = 1.0

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("PrioritizedSampler supports sequence_length=1")
        if _is_incremental_store(store):
            return self._sample_incrementally(store, request)
        transition_ids = _eligible_n_step_ids(store, request)
        if len(transition_ids) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} transitions, replay has {len(transition_ids)}"
            )
        self._synchronize(transition_ids)
        scaled = [self._priorities[transition_id] ** self.alpha for transition_id in transition_ids]
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
        normalized_weights = tuple(weight / maximum for weight in weights)
        return _make_batch(
            store,
            self.pipeline,
            chosen,
            request,
            importance_weights=normalized_weights,
            metadata={"sampling": "prioritized", "beta": beta},
        )

    def update_priorities(self, update: PriorityUpdate) -> None:
        for transition_id, priority in zip(update.transition_ids, update.priorities, strict=True):
            value = abs(float(priority)) + self.priority_epsilon
            if not isfinite(value):
                raise ValueError("PER priorities must be finite")
            if transition_id in self._priorities:
                self._priorities[transition_id] = value
                self._maximum_priority = max(self._maximum_priority, value)
                if self._tree is not None and transition_id in self._active_ids:
                    self._tree.set(transition_id % self._tree.size, value**self.alpha)

    def _synchronize(self, transition_ids: list[TransitionId]) -> None:
        active = set(transition_ids)
        self._priorities = {
            index: priority for index, priority in self._priorities.items() if index in active
        }
        maximum = max(self._priorities.values(), default=1.0)
        for index in active:
            self._priorities.setdefault(index, maximum)

    def state_dict(self) -> dict[str, Any]:
        return {"priorities": dict(self._priorities), "rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._priorities = {int(key): float(value) for key, value in state["priorities"].items()}
        self._maximum_priority = max(self._priorities.values(), default=1.0)
        self._active_ids.clear()
        self._slot_ids = []
        self._tree = None
        self._replay_revision = None
        self._n_step = None
        self._rng.setstate(state["rng"])

    def _sample_incrementally(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> TrainingBatch:
        self._synchronize_incremental_store(store, request.n_step)
        if len(self._active_ids) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} transitions, replay has {len(self._active_ids)}"
            )
        assert self._tree is not None
        total = self._tree.total
        if total <= 0.0:
            raise RuntimeError("Prioritized replay has no positive sampling mass")
        transition_ids: list[TransitionId] = []
        probabilities: list[float] = []
        for _ in range(request.batch_size):
            slot = self._tree.find(self._rng.random() * total)
            transition_id = self._slot_ids[slot]
            if transition_id is None:
                raise RuntimeError("Prioritized replay tree is out of sync with active transitions")
            transition_ids.append(transition_id)
            probabilities.append(self._tree.leaves[slot] / total)
        beta = self.beta if request.beta is None else request.beta
        weights = [
            (len(self._active_ids) * probability) ** (-beta) for probability in probabilities
        ]
        maximum = max(weights)
        return _make_batch(
            store,
            self.pipeline,
            transition_ids,
            request,
            importance_weights=tuple(weight / maximum for weight in weights),
            metadata={"sampling": "prioritized", "beta": beta},
        )

    def _synchronize_incremental_store(self, store: _IncrementalReplayStore, n_step: int) -> None:
        capacity = store.capacity
        if self._tree is None or self._n_step != n_step or self._tree.size != capacity:
            self._tree = _FenwickTree(capacity)
            self._slot_ids = [None] * capacity
            self._active_ids.clear()
            self._replay_revision = None
            self._n_step = n_step
        revision, changes = store.changes_since(self._replay_revision)
        if changes is None:
            self._active_ids.clear()
            self._tree = _FenwickTree(capacity)
            self._slot_ids = [None] * capacity
            for transition_id in store.eligible_transition_ids(n_step):
                self._activate(transition_id)
        else:
            for appended, evicted in changes:
                if evicted is not None:
                    self._deactivate(evicted)
                for candidate in range(appended - n_step + 1, appended + 1):
                    if store.contains(candidate) and store.is_n_step_eligible(candidate, n_step):
                        self._activate(candidate)
                    else:
                        self._deactivate(candidate)
        self._replay_revision = revision

    def _activate(self, transition_id: TransitionId) -> None:
        if transition_id in self._active_ids:
            return
        assert self._tree is not None
        slot = transition_id % self._tree.size
        replaced = self._slot_ids[slot]
        if replaced is not None:
            self._active_ids.discard(replaced)
            self._priorities.pop(replaced, None)
        priority = self._priorities.setdefault(transition_id, self._maximum_priority)
        self._active_ids.add(transition_id)
        self._slot_ids[slot] = transition_id
        self._tree.set(slot, priority**self.alpha)

    def _deactivate(self, transition_id: TransitionId) -> None:
        if transition_id not in self._active_ids:
            return
        assert self._tree is not None
        self._active_ids.remove(transition_id)
        self._priorities.pop(transition_id, None)
        slot = transition_id % self._tree.size
        if self._slot_ids[slot] == transition_id:
            self._slot_ids[slot] = None
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
    seen: set[TransitionId] = set()
    for transition_id in transition_ids:
        for offset in range(request.n_step):
            candidate = transition_id + offset
            if candidate not in seen and store.contains(candidate):
                seen.add(candidate)
                requested_ids.append(candidate)
    available = dict(zip(requested_ids, store.get(requested_ids), strict=True))
    n_step = [
        _n_step_transition(transition_id, available, request) for transition_id in transition_ids
    ]
    transitions = [item[0] for item in n_step]
    discounts = [item[1] for item in n_step]
    data = pipeline.collate(transitions)
    return TrainingBatch(
        data=data,
        observations=tree_collate([item.observation for item in transitions]),
        actions=tree_collate([item.action for item in transitions]),
        rewards=tree_collate([item.reward for item in transitions]),
        next_observations=tree_collate([item.next_observation for item in transitions]),
        terminated=tree_collate([item.terminated for item in transitions]),
        truncated=tree_collate([item.truncated for item in transitions]),
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
) -> tuple[Transition, float]:
    """Return the episode-safe discounted n-step transition beginning at ``transition_id``."""

    first = available[transition_id]
    current = first
    reward = 0.0
    discount = 1.0
    effective_steps = 0
    terminated = False
    truncated = False
    for offset in range(request.n_step):
        current_id = transition_id + offset
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
