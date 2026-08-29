"""Incremental index synchronization for prioritized replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from trackmaniarl.core.data import TransitionId
from trackmaniarl.core.replay.store import _IncrementalReplayStore
from trackmaniarl.core.replay.store_support import _ReplayChange
from trackmaniarl.core.replay.structures import _FenwickTree

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler


@dataclass(frozen=True, slots=True)
class _SynchronizationRequest:
    sampler: PrioritizedSampler
    store: _IncrementalReplayStore
    n_step: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class _SamplerTrees:
    all: _FenwickTree
    uniform: _FenwickTree
    expert: _FenwickTree
    expert_uniform: _FenwickTree
    non_expert: _FenwickTree
    non_expert_uniform: _FenwickTree


@dataclass(frozen=True, slots=True)
class _SlotClassification:
    elite: bool
    expert: bool


def _synchronize_incremental_store(request: _SynchronizationRequest) -> None:
    if _requires_index_reset(request):
        _initialize_index(request)
    revision, changes = request.store.changes_since(request.sampler._replay_revision)
    if changes is None:
        _rebuild_index(request)
    else:
        _apply_changes(request, changes)
    request.sampler._replay_revision = revision


def _requires_index_reset(request: _SynchronizationRequest) -> bool:
    tree = request.sampler._tree
    return (
        tree is None
        or request.sampler._n_step != request.n_step
        or request.sampler._sequence_length != request.sequence_length
        or tree.size != request.store.capacity
    )


def _initialize_index(request: _SynchronizationRequest) -> None:
    sampler = request.sampler
    _reset_trees(sampler, request.store.capacity)
    _resize_slot_arrays(sampler, request.store.capacity)
    _reset_counts(sampler)
    sampler._replay_revision = None
    sampler._n_step = request.n_step
    sampler._sequence_length = request.sequence_length


def _reset_trees(sampler: PrioritizedSampler, capacity: int) -> None:
    sampler._tree = _FenwickTree(capacity)
    sampler._uniform_tree = _FenwickTree(capacity)
    sampler._expert_tree = _FenwickTree(capacity)
    sampler._expert_uniform_tree = _FenwickTree(capacity)
    sampler._non_expert_tree = _FenwickTree(capacity)
    sampler._non_expert_uniform_tree = _FenwickTree(capacity)


def _resize_slot_arrays(sampler: PrioritizedSampler, capacity: int) -> None:
    if sampler._slot_ids.shape != (capacity,):
        sampler._slot_ids = np.full(capacity, -1, dtype=np.int64)
        sampler._priorities = np.zeros(capacity, dtype=np.float32)
    if sampler._elite_slots.shape != (capacity,):
        sampler._elite_slots = np.zeros(capacity, dtype=np.bool_)
    if sampler._expert_slots.shape != (capacity,):
        sampler._expert_slots = np.zeros(capacity, dtype=np.bool_)


def _reset_counts(sampler: PrioritizedSampler) -> None:
    sampler._active_count = 0
    sampler._elite_active_count = 0
    sampler._expert_active_count = 0


def _rebuild_index(request: _SynchronizationRequest) -> None:
    _reset_counts(request.sampler)
    _reset_trees(request.sampler, request.store.capacity)
    for transition_id in request.store.eligible_transition_ids(request.n_step):
        history = request.store.history_ids(transition_id, request.sequence_length)
        if len(set(history)) == request.sequence_length:
            _activate(request.sampler, request.store, transition_id)


def _apply_changes(
    request: _SynchronizationRequest,
    changes: list[_ReplayChange],
) -> None:
    candidates: set[TransitionId] = set()
    for change in changes:
        if change.evicted is not None:
            _deactivate(request.sampler, change.evicted)
        candidates.update(_affected_candidates(request, change))
    for candidate in candidates:
        _refresh_candidate(request, candidate)


def _affected_candidates(
    request: _SynchronizationRequest, change: _ReplayChange
) -> set[TransitionId]:
    store = request.store
    candidates = set(store.affected_n_step_starts(change.appended, request.n_step))
    candidates.update(store.n_step_ids(change.appended, request.sequence_length))
    if change.evicted_previous is not None and request.n_step > 1:
        candidates.update(store.affected_n_step_starts(change.evicted_previous, request.n_step - 1))
    if change.evicted_next is not None and request.sequence_length > 1:
        candidates.update(store.n_step_ids(change.evicted_next, request.sequence_length - 1))
    return candidates


def _refresh_candidate(request: _SynchronizationRequest, candidate: TransitionId) -> None:
    if _eligible_candidate(request, candidate):
        _activate(request.sampler, request.store, candidate)
    else:
        _deactivate(request.sampler, candidate)


def _eligible_candidate(request: _SynchronizationRequest, candidate: TransitionId) -> bool:
    store = request.store
    return (
        store.contains(candidate)
        and store.is_n_step_eligible(candidate, request.n_step)
        and len(set(store.history_ids(candidate, request.sequence_length)))
        == request.sequence_length
    )


def _activate(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    transition_id: TransitionId,
) -> None:
    trees = _trees(sampler)
    slot = transition_id % trees.all.size
    if sampler._slot_ids[slot] == transition_id and trees.all.leaves[slot] > 0.0:
        return
    if trees.all.leaves[slot] > 0.0:
        _decrement_counts(sampler, slot)
    priority = _slot_priority(sampler, slot, transition_id)
    classification = _classify_slot(sampler, store, transition_id)
    sampler._priorities[slot] = priority
    sampler._slot_ids[slot] = transition_id
    sampler._elite_slots[slot] = classification.elite
    sampler._expert_slots[slot] = classification.expert
    _set_slot_weight(sampler, slot)
    _increment_counts(sampler, classification)


def _trees(sampler: PrioritizedSampler) -> _SamplerTrees:
    return _SamplerTrees(
        _required_tree(sampler._tree),
        _required_tree(sampler._uniform_tree),
        _required_tree(sampler._expert_tree),
        _required_tree(sampler._expert_uniform_tree),
        _required_tree(sampler._non_expert_tree),
        _required_tree(sampler._non_expert_uniform_tree),
    )


def _required_tree(tree: _FenwickTree | None) -> _FenwickTree:
    if tree is None:
        raise RuntimeError("prioritized replay index is not initialized")
    return tree


def _decrement_counts(sampler: PrioritizedSampler, slot: int) -> None:
    sampler._active_count -= 1
    sampler._elite_active_count -= int(sampler._elite_slots[slot])
    sampler._expert_active_count -= int(sampler._expert_slots[slot])


def _increment_counts(sampler: PrioritizedSampler, classification: _SlotClassification) -> None:
    sampler._active_count += 1
    sampler._elite_active_count += int(classification.elite)
    sampler._expert_active_count += int(classification.expert)


def _slot_priority(sampler: PrioritizedSampler, slot: int, transition_id: TransitionId) -> float:
    existing = sampler._slot_ids[slot] == transition_id and sampler._priorities[slot] > 0.0
    return float(sampler._priorities[slot]) if existing else sampler._maximum_priority


def _classify_slot(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    transition_id: TransitionId,
) -> _SlotClassification:
    pace = store.sampling_pace_s(transition_id)
    elite = sampler.elite_time_s is not None and pace <= sampler.elite_time_s
    is_demo = store.demo_flags([transition_id])[0]
    expert = (
        is_demo and sampler.expert_demo_time_s is not None and pace <= sampler.expert_demo_time_s
    )
    return _SlotClassification(elite, expert)


def _set_slot_weight(sampler: PrioritizedSampler, slot: int) -> None:
    trees = _trees(sampler)
    boost = sampler.elite_priority_boost if sampler._elite_slots[slot] else 1.0
    weight = float(sampler._priorities[slot]) ** sampler.alpha * boost
    expert = bool(sampler._expert_slots[slot])
    trees.all.set(slot, weight)
    trees.uniform.set(slot, 1.0)
    trees.expert.set(slot, weight if expert else 0.0)
    trees.expert_uniform.set(slot, 1.0 if expert else 0.0)
    trees.non_expert.set(slot, 0.0 if expert else weight)
    trees.non_expert_uniform.set(slot, 0.0 if expert else 1.0)


def _deactivate(sampler: PrioritizedSampler, transition_id: TransitionId) -> None:
    trees = _trees(sampler)
    slot = transition_id % trees.all.size
    if sampler._slot_ids[slot] != transition_id or trees.all.leaves[slot] <= 0.0:
        return
    _decrement_counts(sampler, slot)
    for tree in (
        trees.all,
        trees.uniform,
        trees.expert,
        trees.expert_uniform,
        trees.non_expert,
        trees.non_expert_uniform,
    ):
        tree.set(slot, 0.0)
