"""Incremental prioritized-replay orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trackmaniarl.core.data import BatchRequest, TrainingBatch, TransitionId
from trackmaniarl.core.replay import prioritized_draw, prioritized_index, prioritized_sampling
from trackmaniarl.core.replay.store import _IncrementalReplayStore

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler

_SynchronizationRequest = prioritized_index._SynchronizationRequest


def _sample_incrementally(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    request: BatchRequest,
) -> TrainingBatch:
    with store.sampling_transaction():
        return sampler._sample_incremental_snapshot(store, request)


def _sample_incremental_snapshot(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    request: BatchRequest,
) -> TrainingBatch:
    return prioritized_sampling._sample_incremental_snapshot(sampler, store, request)


def _sample_incremental_ids(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    request: BatchRequest,
) -> tuple[list[TransitionId], tuple[float, ...], float, dict[str, float]]:
    return prioritized_draw._sample_incremental_ids(sampler, store, request)


def _incremental_choices(
    sampler: PrioritizedSampler, batch_size: int
) -> tuple[list[TransitionId], list[float]]:
    return prioritized_draw._incremental_choices(sampler, batch_size)


def _synchronize_incremental_store(request: _SynchronizationRequest) -> None:
    prioritized_index._synchronize_incremental_store(request)


def _activate(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    transition_id: TransitionId,
) -> None:
    prioritized_index._activate(sampler, store, transition_id)


def _set_slot_weight(sampler: PrioritizedSampler, slot: int) -> None:
    prioritized_index._set_slot_weight(sampler, slot)


def _deactivate(sampler: PrioritizedSampler, transition_id: TransitionId) -> None:
    prioritized_index._deactivate(sampler, transition_id)
