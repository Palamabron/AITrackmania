"""Batch assembly for incremental prioritized replay selections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from trackmaniarl.core.data import BatchRequest, TrainingBatch, TransitionId
from trackmaniarl.core.pytree import tree_collate
from trackmaniarl.core.replay.batches import (
    _BatchBuild,
    _history_padding_masks,
    _make_batch,
    _reshape_batch_sequences,
    _reshape_sequence_batch,
    _sequence_target_observation_histories,
    _SequenceBatchLayout,
)
from trackmaniarl.core.replay.store import _IncrementalReplayStore

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler


@dataclass(frozen=True, slots=True)
class _IncrementalSample:
    sampler: PrioritizedSampler
    store: _IncrementalReplayStore
    request: BatchRequest
    transition_ids: list[TransitionId]
    weights: tuple[float, ...]
    beta: float
    sampling: dict[str, float]
    demo_flags: tuple[bool, ...]
    expert_flags: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _SequenceMaterialization:
    histories: list[list[TransitionId]]
    flattened: list[TransitionId]
    next_observations: list[object]


def _sample_incremental_snapshot(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    request: BatchRequest,
) -> TrainingBatch:
    transition_ids, weights, beta, sampling = sampler._sample_incremental_ids(store, request)
    sample = _IncrementalSample(
        sampler,
        store,
        request,
        transition_ids,
        weights,
        beta,
        sampling,
        tuple(store.demo_flags(transition_ids)),
        _expert_flags(sampler, transition_ids),
    )
    return _sequence_batch(sample) if request.sequence_length > 1 else _flat_batch(sample)


def _expert_flags(
    sampler: PrioritizedSampler, transition_ids: list[TransitionId]
) -> tuple[bool, ...]:
    if sampler._tree is None:
        raise RuntimeError("prioritized replay index is not initialized")
    return tuple(bool(sampler._expert_slots[item % sampler._tree.size]) for item in transition_ids)


def _sequence_batch(sample: _IncrementalSample) -> TrainingBatch:
    materialized = _sequence_materialization(sample)
    batch = _make_batch(
        _BatchBuild(
            sample.store,
            sample.sampler.pipeline,
            materialized.flattened,
            sample.request,
            importance_weights=sample.weights,
            metadata=_sample_metadata(sample, "prioritized_sequence"),
            bootstrap_stride=sample.request.sequence_length,
        )
    )
    return _reshape_sequence_output(sample, materialized, batch)


def _sequence_materialization(sample: _IncrementalSample) -> _SequenceMaterialization:
    length = sample.request.sequence_length
    histories = [sample.store.history_ids(item, length) for item in sample.transition_ids]
    if any(len(history) != length for history in histories):
        raise RuntimeError("prioritized sequence index is out of sync with replay")
    horizons = [
        sample.store.n_step_ids(item, sample.request.n_step) for item in sample.transition_ids
    ]
    if any(not horizon for horizon in horizons):
        raise RuntimeError("prioritized n-step index is out of sync with replay")
    history_values = [sample.store.get(history) for history in histories]
    horizon_values = [sample.store.get(horizon) for horizon in horizons]
    next_histories = _sequence_target_observation_histories(history_values, horizon_values, length)
    flattened = [item for history in histories for item in history]
    next_observations = [item for history in next_histories for item in history]
    return _SequenceMaterialization(histories, flattened, next_observations)


def _reshape_sequence_output(
    sample: _IncrementalSample,
    materialized: _SequenceMaterialization,
    batch: TrainingBatch,
) -> TrainingBatch:
    request = sample.request
    layout = _SequenceBatchLayout(
        request.batch_size,
        request.sequence_length,
        _history_padding_masks(materialized.histories),
    )
    reshaped = _reshape_batch_sequences(batch, layout)
    next_observations = _reshape_sequence_batch(
        tree_collate(materialized.next_observations),
        request.batch_size,
        request.sequence_length,
    )
    return replace(reshaped, next_observations=next_observations)


def _flat_batch(sample: _IncrementalSample) -> TrainingBatch:
    return _make_batch(
        _BatchBuild(
            sample.store,
            sample.sampler.pipeline,
            sample.transition_ids,
            sample.request,
            importance_weights=sample.weights,
            metadata=_sample_metadata(sample, "prioritized"),
        )
    )


def _sample_metadata(
    sample: _IncrementalSample, kind: Literal["prioritized", "prioritized_sequence"]
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "sampling": kind,
        "beta": sample.beta,
        "demo_flags": sample.demo_flags,
        **sample.sampling,
    }
    if kind == "prioritized_sequence":
        metadata.update(_sequence_metadata(sample.request, sample.transition_ids))
    if sample.sampler.expert_demo_time_s is not None:
        metadata["expert_demo_flags"] = sample.expert_flags
    return metadata


def _sequence_metadata(
    request: BatchRequest, transition_ids: list[TransitionId]
) -> dict[str, object]:
    return {
        "sequence_length": request.sequence_length,
        "n_step": request.n_step,
        "gamma": request.gamma,
        "priority_transition_ids": tuple(transition_ids),
    }
