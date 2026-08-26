"""Proportional prioritized replay sampling."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from math import isfinite
from statistics import fmean
from threading import RLock
from typing import Any, cast

import numpy as np

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, TransitionId
from trackmaniarl.core.pytree import tree_collate
from trackmaniarl.core.replay.batches import (
    _eligible_n_step_ids,
    _history_padding_masks,
    _make_batch,
    _reshape_batch_sequences,
    _reshape_sequence_batch,
    _sequence_target_observation_histories,
)
from trackmaniarl.core.replay.store import _IncrementalReplayStore, _is_demo, _is_incremental_store
from trackmaniarl.core.replay.structures import _FenwickTree


class PrioritizedSampler:
    """Array-backed proportional PER with normalized importance weights."""

    thread_safe_prefetch = True
    supports_sequence_sampling = True

    def __init__(
        self,
        pipeline: Any,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        priority_epsilon: float = 1e-6,
        elite_time_s: float | None = None,
        elite_priority_boost: float = 1.0,
        expert_demo_time_s: float | None = None,
        expert_fraction: float = 0.0,
        uniform_mix: float = 0.0,
        seed: int = 0,
    ) -> None:
        if (
            alpha < 0.0
            or beta < 0.0
            or priority_epsilon <= 0.0
            or (elite_time_s is not None and elite_time_s <= 0.0)
            or elite_priority_boost < 1.0
            or (expert_demo_time_s is not None and expert_demo_time_s <= 0.0)
            or not 0.0 <= expert_fraction <= 1.0
            or (expert_fraction > 0.0 and expert_demo_time_s is None)
            or not 0.0 <= uniform_mix <= 1.0
        ):
            raise ValueError("invalid prioritized replay parameters")
        self.pipeline = pipeline
        self.alpha = alpha
        self.beta = beta
        self.priority_epsilon = priority_epsilon
        self.elite_time_s = elite_time_s
        self.elite_priority_boost = elite_priority_boost
        self.expert_demo_time_s = expert_demo_time_s
        self.expert_fraction = expert_fraction
        self.uniform_mix = uniform_mix
        self._fallback_priorities: dict[int, float] = {}
        self._priorities = np.empty(0, dtype=np.float32)
        self._slot_ids = np.empty(0, dtype=np.int64)
        self._elite_slots = np.empty(0, dtype=np.bool_)
        self._expert_slots = np.empty(0, dtype=np.bool_)
        self._rng = random.Random(seed)
        self._active_count = 0
        self._elite_active_count = 0
        self._expert_active_count = 0
        self._tree: _FenwickTree | None = None
        self._uniform_tree: _FenwickTree | None = None
        self._expert_tree: _FenwickTree | None = None
        self._expert_uniform_tree: _FenwickTree | None = None
        self._non_expert_tree: _FenwickTree | None = None
        self._non_expert_uniform_tree: _FenwickTree | None = None
        self._replay_revision: int | None = None
        self._n_step: int | None = None
        self._sequence_length: int | None = None
        self._maximum_priority = 1.0
        self._lock = RLock()

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if _is_incremental_store(store):
            return self._sample_incrementally(store, request)
        if request.sequence_length != 1:
            raise ValueError("sequence PER requires InMemoryReplayStore")
        chosen, normalized, beta, sampling, demo_flags, expert_flags = self._sample_fallback(
            store, request
        )
        return _make_batch(
            store,
            self.pipeline,
            chosen,
            request,
            importance_weights=normalized,
            metadata={
                "sampling": "prioritized",
                "beta": beta,
                "demo_flags": demo_flags,
                **(
                    {"expert_demo_flags": expert_flags}
                    if self.expert_demo_time_s is not None
                    else {}
                ),
                **sampling,
            },
        )

    def _sample_fallback(
        self, store: ReplayStore, request: BatchRequest
    ) -> tuple[
        list[TransitionId],
        tuple[float, ...],
        float,
        dict[str, float],
        tuple[bool, ...],
        tuple[bool, ...],
    ]:
        with self._lock:
            transition_ids = _eligible_n_step_ids(store, request)
            if len(transition_ids) < request.batch_size:
                raise RuntimeError(
                    f"Need {request.batch_size} transitions, replay has {len(transition_ids)}"
                )
            self._synchronize_fallback(transition_ids)
            transitions = store.get(transition_ids)
            elite_by_id = {
                transition_id: (
                    self.elite_time_s is not None
                    and float(transition.info.get("sampling/projected_lap_time_s", float("inf")))
                    <= self.elite_time_s
                )
                for transition_id, transition in zip(transition_ids, transitions, strict=True)
            }
            scaled = [
                self._fallback_priorities[transition_id] ** self.alpha
                * (self.elite_priority_boost if elite_by_id[transition_id] else 1.0)
                for transition_id in transition_ids
            ]
            total = sum(scaled)
            probabilities = (
                [weight / total for weight in scaled]
                if total > 0.0
                else [1 / len(transition_ids)] * len(transition_ids)
            )
            if self.uniform_mix:
                uniform = 1.0 / len(transition_ids)
                probabilities = [
                    (1.0 - self.uniform_mix) * probability + self.uniform_mix * uniform
                    for probability in probabilities
                ]
            demo_by_id = {
                transition_id: _is_demo(transition.info)
                for transition_id, transition in zip(transition_ids, transitions, strict=True)
            }
            expert_by_id = {
                transition_id: (
                    demo_by_id[transition_id]
                    and self.expert_demo_time_s is not None
                    and float(transition.info.get("sampling/projected_lap_time_s", float("inf")))
                    <= self.expert_demo_time_s
                )
                for transition_id, transition in zip(transition_ids, transitions, strict=True)
            }
            chosen, chosen_probabilities = self._stratified_fallback_choices(
                transition_ids,
                probabilities,
                expert_by_id,
                request.batch_size,
            )
            beta = self.beta if request.beta is None else request.beta
            weights = [
                (len(transition_ids) * probability) ** (-beta)
                for probability in chosen_probabilities
            ]
            maximum = max(weights)
            demo_flags = tuple(demo_by_id[transition_id] for transition_id in chosen)
            expert_flags = tuple(expert_by_id[transition_id] for transition_id in chosen)
            elite_flags = tuple(elite_by_id[transition_id] for transition_id in chosen)
            metadata = {
                "replay/elite_active_fraction": sum(elite_by_id.values()) / len(transition_ids),
                "replay/elite_sample_fraction": sum(elite_flags) / len(chosen),
                "replay/demo_sample_fraction": sum(demo_flags) / len(chosen),
                "replay/expert_demo_sample_fraction": sum(expert_flags) / len(chosen),
            }
            return (
                chosen,
                tuple(weight / maximum for weight in weights),
                beta,
                metadata,
                demo_flags,
                expert_flags,
            )

    def _stratified_fallback_choices(
        self,
        transition_ids: list[TransitionId],
        probabilities: list[float],
        expert_by_id: Mapping[TransitionId, bool],
        batch_size: int,
    ) -> tuple[list[TransitionId], list[float]]:
        if self.expert_fraction == 0.0 or all(expert_by_id.values()):
            chosen = self._rng.choices(transition_ids, weights=probabilities, k=batch_size)
            by_id = dict(zip(transition_ids, probabilities, strict=True))
            return chosen, [by_id[transition_id] for transition_id in chosen]
        expert_count = round(self.expert_fraction * batch_size)
        return self._fallback_group_choices(
            transition_ids,
            probabilities,
            expert_by_id,
            expert_count,
            batch_size - expert_count,
        )

    def _fallback_group_choices(
        self,
        transition_ids: list[TransitionId],
        probabilities: list[float],
        expert_by_id: Mapping[TransitionId, bool],
        expert_count: int,
        non_expert_count: int,
    ) -> tuple[list[TransitionId], list[float]]:
        grouped: list[tuple[TransitionId, float]] = []
        for expert, count in ((True, expert_count), (False, non_expert_count)):
            if count == 0:
                continue
            candidates = [
                (transition_id, probability)
                for transition_id, probability in zip(transition_ids, probabilities, strict=True)
                if expert_by_id[transition_id] is expert
            ]
            if not candidates:
                label = "expert demonstration" if expert else "non-expert"
                raise RuntimeError(f"Prioritized replay has no {label} transitions")
            ids, weights = zip(*candidates, strict=True)
            total = sum(weights)
            conditional = [weight / total for weight in weights]
            group_fraction = count / (expert_count + non_expert_count)
            selected = self._rng.choices(ids, weights=conditional, k=count)
            by_id = dict(zip(ids, conditional, strict=True))
            grouped.extend(
                (transition_id, group_fraction * by_id[transition_id]) for transition_id in selected
            )
        self._rng.shuffle(grouped)
        return [item[0] for item in grouped], [item[1] for item in grouped]

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
                    self._set_slot_weight(slot)

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
        self._elite_active_count = 0
        self._expert_active_count = 0
        self._tree = None
        self._uniform_tree = None
        self._expert_tree = None
        self._expert_uniform_tree = None
        self._non_expert_tree = None
        self._non_expert_uniform_tree = None
        self._replay_revision = None
        self._n_step = None
        self._sequence_length = None
        self._rng.setstate(state["rng"])

    def _sample_incrementally(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> TrainingBatch:
        with store.sampling_transaction():
            return self._sample_incremental_snapshot(store, request)

    def _sample_incremental_snapshot(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> TrainingBatch:
        transition_ids, normalized, beta, sampling = self._sample_incremental_ids(store, request)
        demo_flags = tuple(store.demo_flags(transition_ids))
        assert self._tree is not None
        expert_flags = tuple(
            bool(self._expert_slots[transition_id % self._tree.size])
            for transition_id in transition_ids
        )
        if request.sequence_length > 1:
            histories = [
                store.history_ids(transition_id, request.sequence_length)
                for transition_id in transition_ids
            ]
            if any(len(history) != request.sequence_length for history in histories):
                raise RuntimeError("prioritized sequence index is out of sync with replay")
            flattened = [transition_id for history in histories for transition_id in history]
            horizon_ids = [
                store.n_step_ids(transition_id, request.n_step) for transition_id in transition_ids
            ]
            if any(not horizon for horizon in horizon_ids):
                raise RuntimeError("prioritized n-step index is out of sync with replay")
            history_transitions = [store.get(history) for history in histories]
            horizon_transitions = [store.get(horizon) for horizon in horizon_ids]
            next_histories = _sequence_target_observation_histories(
                history_transitions,
                horizon_transitions,
                request.sequence_length,
            )
            batch = _make_batch(
                store,
                self.pipeline,
                flattened,
                request,
                importance_weights=normalized,
                metadata={
                    "sampling": "prioritized_sequence",
                    "beta": beta,
                    "sequence_length": request.sequence_length,
                    "n_step": request.n_step,
                    "gamma": request.gamma,
                    "priority_transition_ids": tuple(transition_ids),
                    "demo_flags": demo_flags,
                    **(
                        {"expert_demo_flags": expert_flags}
                        if self.expert_demo_time_s is not None
                        else {}
                    ),
                    **sampling,
                },
                bootstrap_stride=request.sequence_length,
            )
            reshaped = _reshape_batch_sequences(
                batch,
                request.batch_size,
                request.sequence_length,
                masks=_history_padding_masks(histories),
            )
            flattened_next = [observation for history in next_histories for observation in history]
            return replace(
                reshaped,
                next_observations=_reshape_sequence_batch(
                    tree_collate(flattened_next),
                    request.batch_size,
                    request.sequence_length,
                ),
            )
        return _make_batch(
            store,
            self.pipeline,
            transition_ids,
            request,
            importance_weights=normalized,
            metadata={
                "sampling": "prioritized",
                "beta": beta,
                "demo_flags": demo_flags,
                **(
                    {"expert_demo_flags": expert_flags}
                    if self.expert_demo_time_s is not None
                    else {}
                ),
                **sampling,
            },
        )

    def _sample_incremental_ids(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> tuple[list[TransitionId], tuple[float, ...], float, dict[str, float]]:
        with self._lock:
            self._synchronize_incremental_store(
                store,
                request.n_step,
                request.sequence_length,
            )
            if self._active_count < request.batch_size:
                raise RuntimeError(
                    f"Need {request.batch_size} transitions, replay has {self._active_count}"
                )
            assert self._tree is not None
            assert self._uniform_tree is not None
            transition_ids, probabilities = self._incremental_choices(request.batch_size)
            beta = self.beta if request.beta is None else request.beta
            weights = [
                (self._active_count * probability) ** (-beta) for probability in probabilities
            ]
            maximum = max(weights)
            elite_samples = sum(
                int(self._elite_slots[transition_id % self._tree.size])
                for transition_id in transition_ids
            )
            demo_samples = sum(store.demo_flags(transition_ids))
            expert_samples = sum(
                int(self._expert_slots[transition_id % self._tree.size])
                for transition_id in transition_ids
            )
            metadata = {
                "replay/active_count": float(self._active_count),
                "replay/elite_active_fraction": self._elite_active_count / self._active_count,
                "replay/elite_sample_fraction": elite_samples / len(transition_ids),
                "replay/demo_sample_fraction": demo_samples / len(transition_ids),
                "replay/expert_demo_active_fraction": self._expert_active_count
                / self._active_count,
                "replay/expert_demo_sample_fraction": expert_samples / len(transition_ids),
                "replay/sampling_probability_mean": fmean(probabilities),
                "replay/sampling_probability_min": min(probabilities),
                "replay/sampling_probability_max": max(probabilities),
            }
            return (
                transition_ids,
                tuple(weight / maximum for weight in weights),
                beta,
                metadata,
            )

    def _incremental_choices(self, batch_size: int) -> tuple[list[TransitionId], list[float]]:
        assert self._tree is not None
        assert self._uniform_tree is not None
        if self.expert_fraction == 0.0:
            return self._draw_incremental_group(
                self._tree,
                self._uniform_tree,
                self._active_count,
                batch_size,
                1.0,
            )
        assert self._expert_tree is not None
        assert self._expert_uniform_tree is not None
        assert self._non_expert_tree is not None
        assert self._non_expert_uniform_tree is not None
        if self._expert_active_count == self._active_count:
            return self._draw_incremental_group(
                self._expert_tree,
                self._expert_uniform_tree,
                self._expert_active_count,
                batch_size,
                1.0,
            )
        expert_count = round(self.expert_fraction * batch_size)
        grouped: list[tuple[TransitionId, float]] = []
        for trees, active_count, count in (
            (
                (self._expert_tree, self._expert_uniform_tree),
                self._expert_active_count,
                expert_count,
            ),
            (
                (self._non_expert_tree, self._non_expert_uniform_tree),
                self._active_count - self._expert_active_count,
                batch_size - expert_count,
            ),
        ):
            if count == 0:
                continue
            ids, probabilities = self._draw_incremental_group(
                *trees,
                active_count,
                count,
                count / batch_size,
            )
            grouped.extend(zip(ids, probabilities, strict=True))
        self._rng.shuffle(grouped)
        return [item[0] for item in grouped], [item[1] for item in grouped]

    def _draw_incremental_group(
        self,
        tree: _FenwickTree,
        uniform_tree: _FenwickTree,
        active_count: int,
        count: int,
        group_fraction: float,
    ) -> tuple[list[TransitionId], list[float]]:
        if active_count < 1 or tree.total <= 0.0 or uniform_tree.total <= 0.0:
            raise RuntimeError("Prioritized replay cannot satisfy expert_fraction")
        transition_ids: list[TransitionId] = []
        probabilities: list[float] = []
        for _ in range(count):
            if self.uniform_mix and self._rng.random() < self.uniform_mix:
                slot = uniform_tree.find(self._rng.random() * uniform_tree.total)
            else:
                slot = tree.find(self._rng.random() * tree.total)
            transition_id = int(self._slot_ids[slot])
            if transition_id < 0:
                raise RuntimeError("Prioritized replay tree is out of sync")
            proportional = float(tree.leaves[slot]) / tree.total
            uniform = 1.0 / active_count
            probabilities.append(
                group_fraction
                * ((1.0 - self.uniform_mix) * proportional + self.uniform_mix * uniform)
            )
            transition_ids.append(transition_id)
        return transition_ids, probabilities

    def _synchronize_incremental_store(
        self,
        store: _IncrementalReplayStore,
        n_step: int,
        sequence_length: int = 1,
    ) -> None:
        capacity = store.capacity
        if (
            self._tree is None
            or self._n_step != n_step
            or self._sequence_length != sequence_length
            or self._tree.size != capacity
        ):
            self._tree = _FenwickTree(capacity)
            self._uniform_tree = _FenwickTree(capacity)
            self._expert_tree = _FenwickTree(capacity)
            self._expert_uniform_tree = _FenwickTree(capacity)
            self._non_expert_tree = _FenwickTree(capacity)
            self._non_expert_uniform_tree = _FenwickTree(capacity)
            if self._slot_ids.shape != (capacity,):
                self._slot_ids = np.full(capacity, -1, dtype=np.int64)
                self._priorities = np.zeros(capacity, dtype=np.float32)
            if self._elite_slots.shape != (capacity,):
                self._elite_slots = np.zeros(capacity, dtype=np.bool_)
            if self._expert_slots.shape != (capacity,):
                self._expert_slots = np.zeros(capacity, dtype=np.bool_)
            self._active_count = 0
            self._elite_active_count = 0
            self._expert_active_count = 0
            self._replay_revision = None
            self._n_step = n_step
            self._sequence_length = sequence_length
        revision, changes = store.changes_since(self._replay_revision)
        if changes is None:
            self._active_count = 0
            self._elite_active_count = 0
            self._expert_active_count = 0
            self._tree = _FenwickTree(capacity)
            self._uniform_tree = _FenwickTree(capacity)
            self._expert_tree = _FenwickTree(capacity)
            self._expert_uniform_tree = _FenwickTree(capacity)
            self._non_expert_tree = _FenwickTree(capacity)
            self._non_expert_uniform_tree = _FenwickTree(capacity)
            for transition_id in store.eligible_transition_ids(n_step):
                history = store.history_ids(transition_id, sequence_length)
                if len(set(history)) == sequence_length:
                    self._activate(store, transition_id)
        else:
            for appended, evicted in changes:
                if evicted is not None:
                    self._deactivate(evicted)
                for candidate in store.affected_n_step_starts(appended, n_step):
                    if (
                        store.contains(candidate)
                        and store.is_n_step_eligible(candidate, n_step)
                        and len(set(store.history_ids(candidate, sequence_length)))
                        == sequence_length
                    ):
                        self._activate(store, candidate)
                    else:
                        self._deactivate(candidate)
        self._replay_revision = revision

    def _activate(
        self,
        store: _IncrementalReplayStore,
        transition_id: TransitionId,
    ) -> None:
        assert self._tree is not None
        assert self._uniform_tree is not None
        assert self._expert_tree is not None
        assert self._expert_uniform_tree is not None
        assert self._non_expert_tree is not None
        assert self._non_expert_uniform_tree is not None
        slot = transition_id % self._tree.size
        if self._slot_ids[slot] == transition_id and self._tree.leaves[slot] > 0.0:
            return
        if self._tree.leaves[slot] > 0.0:
            self._active_count -= 1
            self._elite_active_count -= int(self._elite_slots[slot])
            self._expert_active_count -= int(self._expert_slots[slot])
        priority = (
            float(self._priorities[slot])
            if self._slot_ids[slot] == transition_id and self._priorities[slot] > 0.0
            else self._maximum_priority
        )
        self._priorities[slot] = priority
        self._slot_ids[slot] = transition_id
        pace = store.sampling_pace_s(transition_id)
        elite = self.elite_time_s is not None and pace <= self.elite_time_s
        is_demo = store.demo_flags([transition_id])[0]
        expert = is_demo and self.expert_demo_time_s is not None and pace <= self.expert_demo_time_s
        self._elite_slots[slot] = elite
        self._expert_slots[slot] = expert
        self._set_slot_weight(slot)
        self._active_count += 1
        self._elite_active_count += int(elite)
        self._expert_active_count += int(expert)

    def _set_slot_weight(self, slot: int) -> None:
        assert self._tree is not None
        assert self._uniform_tree is not None
        assert self._expert_tree is not None
        assert self._expert_uniform_tree is not None
        assert self._non_expert_tree is not None
        assert self._non_expert_uniform_tree is not None
        boost = self.elite_priority_boost if self._elite_slots[slot] else 1.0
        weight = float(self._priorities[slot]) ** self.alpha * boost
        expert = bool(self._expert_slots[slot])
        self._tree.set(slot, weight)
        self._uniform_tree.set(slot, 1.0)
        self._expert_tree.set(slot, weight if expert else 0.0)
        self._expert_uniform_tree.set(slot, 1.0 if expert else 0.0)
        self._non_expert_tree.set(slot, 0.0 if expert else weight)
        self._non_expert_uniform_tree.set(slot, 0.0 if expert else 1.0)

    def _deactivate(self, transition_id: TransitionId) -> None:
        assert self._tree is not None
        assert self._uniform_tree is not None
        assert self._expert_tree is not None
        assert self._expert_uniform_tree is not None
        assert self._non_expert_tree is not None
        assert self._non_expert_uniform_tree is not None
        slot = transition_id % self._tree.size
        if self._slot_ids[slot] == transition_id and self._tree.leaves[slot] > 0.0:
            self._active_count -= 1
            self._elite_active_count -= int(self._elite_slots[slot])
            self._expert_active_count -= int(self._expert_slots[slot])
            self._tree.set(slot, 0.0)
            self._uniform_tree.set(slot, 0.0)
            self._expert_tree.set(slot, 0.0)
            self._expert_uniform_tree.set(slot, 0.0)
            self._non_expert_tree.set(slot, 0.0)
            self._non_expert_uniform_tree.set(slot, 0.0)
