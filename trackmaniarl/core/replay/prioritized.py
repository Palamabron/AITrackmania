"""Proportional prioritized replay sampling."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, NotRequired, TypedDict, Unpack

import numpy as np

import trackmaniarl.core.replay.prioritized_fallback as prioritized_fallback
import trackmaniarl.core.replay.prioritized_incremental as prioritized_incremental
import trackmaniarl.core.replay.prioritized_state as prioritized_state
from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, TransitionId
from trackmaniarl.core.replay.batches import (
    _BatchBuild,
    _make_batch,
)
from trackmaniarl.core.replay.store import _IncrementalReplayStore, _is_incremental_store
from trackmaniarl.core.replay.structures import _FenwickTree


class _PrioritizedOptions(TypedDict):
    alpha: NotRequired[float]
    beta: NotRequired[float]
    priority_epsilon: NotRequired[float]
    elite_time_s: NotRequired[float | None]
    elite_priority_boost: NotRequired[float]
    expert_demo_time_s: NotRequired[float | None]
    expert_fraction: NotRequired[float]
    expert_fraction_final: NotRequired[float | None]
    expert_fraction_anneal_transitions: NotRequired[int | None]
    uniform_mix: NotRequired[float]
    seed: NotRequired[int]


@dataclass(frozen=True, slots=True)
class _PrioritizedConfig:
    alpha: float = 0.6
    beta: float = 0.4
    priority_epsilon: float = 1e-6
    elite_time_s: float | None = None
    elite_priority_boost: float = 1.0
    expert_demo_time_s: float | None = None
    expert_fraction: float = 0.0
    expert_fraction_final: float | None = None
    expert_fraction_anneal_transitions: int | None = None
    uniform_mix: float = 0.0
    seed: int = 0

    @classmethod
    def from_options(cls, options: _PrioritizedOptions) -> _PrioritizedConfig:
        return cls(**options)


def _invalid_expert_schedule(config: _PrioritizedConfig) -> bool:
    schedule = (config.expert_fraction_final, config.expert_fraction_anneal_transitions)
    return any(value is not None for value in schedule) and any(value is None for value in schedule)


def _invalid_expert_config(config: _PrioritizedConfig) -> bool:
    final_fraction = config.expert_fraction_final or 0.0
    return (
        not 0.0 <= config.expert_fraction <= 1.0
        or not 0.0 <= final_fraction <= 1.0
        or (
            config.expert_fraction_anneal_transitions is not None
            and config.expert_fraction_anneal_transitions < 1
        )
        or _invalid_expert_schedule(config)
        or (max(config.expert_fraction, final_fraction) > 0.0 and config.expert_demo_time_s is None)
    )


def _invalid_prioritized_config(config: _PrioritizedConfig) -> bool:
    return (
        config.alpha < 0.0
        or config.beta < 0.0
        or config.priority_epsilon <= 0.0
        or (config.elite_time_s is not None and config.elite_time_s <= 0.0)
        or config.elite_priority_boost < 1.0
        or (config.expert_demo_time_s is not None and config.expert_demo_time_s <= 0.0)
        or _invalid_expert_config(config)
        or not 0.0 <= config.uniform_mix <= 1.0
    )


def _validate_prioritized_config(config: _PrioritizedConfig) -> None:
    if _invalid_prioritized_config(config):
        raise ValueError("invalid prioritized replay parameters")


def _fallback_metadata(
    beta: float, sampling: Mapping[str, Any], demo_flags: tuple[bool, ...]
) -> dict[str, Any]:
    return {"sampling": "prioritized", "beta": beta, "demo_flags": demo_flags, **sampling}


class PrioritizedSampler:
    """Array-backed proportional PER with normalized importance weights."""

    thread_safe_prefetch = True
    supports_sequence_sampling = True

    def __init__(self, pipeline: Any, **options: Unpack[_PrioritizedOptions]) -> None:
        config = _PrioritizedConfig.from_options(options)
        _validate_prioritized_config(config)
        self.pipeline = pipeline
        self._apply_config(config)
        self._initialize_arrays()
        self._initialize_trees()
        self._initialize_tracking(config.seed)

    def _apply_config(self, config: _PrioritizedConfig) -> None:
        self.alpha = config.alpha
        self.beta = config.beta
        self.priority_epsilon = config.priority_epsilon
        self.elite_time_s = config.elite_time_s
        self.elite_priority_boost = config.elite_priority_boost
        self.expert_demo_time_s = config.expert_demo_time_s
        self.expert_fraction = config.expert_fraction
        self.expert_fraction_final = config.expert_fraction_final
        self.expert_fraction_anneal_transitions = config.expert_fraction_anneal_transitions
        self.uniform_mix = config.uniform_mix

    def _expert_fraction_at(self, transition_count: int) -> float:
        if self.expert_fraction_final is None:
            return self.expert_fraction
        assert self.expert_fraction_anneal_transitions is not None
        progress = min(1.0, transition_count / self.expert_fraction_anneal_transitions)
        return self.expert_fraction + progress * (self.expert_fraction_final - self.expert_fraction)

    def _initialize_arrays(self) -> None:
        self._fallback_priorities: dict[int, float] = {}
        self._priorities = np.empty(0, dtype=np.float32)
        self._slot_ids = np.empty(0, dtype=np.int64)
        self._elite_slots = np.empty(0, dtype=np.bool_)
        self._expert_slots = np.empty(0, dtype=np.bool_)

    def _initialize_trees(self) -> None:
        self._tree: _FenwickTree | None = None
        self._uniform_tree: _FenwickTree | None = None
        self._expert_tree: _FenwickTree | None = None
        self._expert_uniform_tree: _FenwickTree | None = None
        self._non_expert_tree: _FenwickTree | None = None
        self._non_expert_uniform_tree: _FenwickTree | None = None

    def _initialize_tracking(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._active_count = 0
        self._elite_active_count = 0
        self._expert_active_count = 0
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
        return self._sample_fallback_batch(store, request)

    def _sample_fallback_batch(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        chosen, normalized, beta, sampling, demo_flags, expert_flags = self._sample_fallback(
            store, request
        )
        metadata = _fallback_metadata(beta, sampling, demo_flags)
        if self.expert_demo_time_s is not None:
            metadata["expert_demo_flags"] = expert_flags
        return _make_batch(
            _BatchBuild(
                store,
                self.pipeline,
                chosen,
                request,
                importance_weights=normalized,
                metadata=metadata,
            )
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
        return prioritized_fallback._sample_fallback(self, store, request)

    def update_priorities(self, update: PriorityUpdate) -> None:
        prioritized_state.update_priorities(self, update)

    def _synchronize_fallback(self, transition_ids: list[TransitionId]) -> None:
        prioritized_state._synchronize_fallback(self, transition_ids)

    def state_dict(self) -> dict[str, Any]:
        return prioritized_state.state_dict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        prioritized_state.load_state_dict(self, state)

    def _sample_incrementally(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> TrainingBatch:
        return prioritized_incremental._sample_incrementally(self, store, request)

    def _sample_incremental_snapshot(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> TrainingBatch:
        return prioritized_incremental._sample_incremental_snapshot(self, store, request)

    def _sample_incremental_ids(
        self, store: _IncrementalReplayStore, request: BatchRequest
    ) -> tuple[list[TransitionId], tuple[float, ...], float, dict[str, float]]:
        return prioritized_incremental._sample_incremental_ids(self, store, request)

    def _incremental_choices(self, request: BatchRequest) -> tuple[list[TransitionId], list[float]]:
        return prioritized_incremental._incremental_choices(self, request)

    def _synchronize_incremental_store(
        self,
        store: _IncrementalReplayStore,
        n_step: int,
        sequence_length: int = 1,
    ) -> None:
        request = prioritized_incremental._SynchronizationRequest(
            self, store, n_step, sequence_length
        )
        prioritized_incremental._synchronize_incremental_store(request)

    def _activate(
        self,
        store: _IncrementalReplayStore,
        transition_id: TransitionId,
    ) -> None:
        prioritized_incremental._activate(self, store, transition_id)

    def _set_slot_weight(self, slot: int) -> None:
        prioritized_incremental._set_slot_weight(self, slot)

    def _deactivate(self, transition_id: TransitionId) -> None:
        prioritized_incremental._deactivate(self, transition_id)
