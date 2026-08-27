"""Replay batch collation, sequence validation, and n-step return construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition, TransitionId
from trackmaniarl.core.pytree import tree_collate, tree_map
from trackmaniarl.core.replay.batch_metadata import _behavior_metadata
from trackmaniarl.core.replay.n_step import _n_step_transition, _NStepInput


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

    try:
        _n_step_horizon(transition_id, available, request)
    except RuntimeError:
        return False
    return True


def _n_step_horizon(
    transition_id: TransitionId,
    available: Mapping[TransitionId, Transition],
    request: BatchRequest,
) -> list[Transition]:
    """Resolve one complete episode-local horizon from a basic replay snapshot."""

    start = _HorizonStart(transition_id, available[transition_id])
    result: list[Transition] = []
    for offset in range(request.n_step):
        candidate = available.get(transition_id + offset)
        candidate = _validated_horizon_candidate(start, candidate, offset)
        result.append(candidate)
        if candidate.terminated or candidate.truncated:
            break
    return result


@dataclass(frozen=True, slots=True)
class _HorizonStart:
    transition_id: TransitionId
    transition: Transition


def _validated_horizon_candidate(
    start: _HorizonStart, candidate: Transition | None, offset: int
) -> Transition:
    if candidate is None or candidate.episode_id != start.transition.episode_id:
        raise RuntimeError(f"Transition {start.transition_id} has no complete n-step horizon")
    if candidate.step is None or start.transition.step is None:
        return candidate
    if candidate.step != start.transition.step + offset:
        raise RuntimeError(f"Transition {start.transition_id} has a discontinuous n-step horizon")
    return candidate


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


def _history_padding_masks(histories: list[list[TransitionId]]) -> torch.Tensor:
    """Mark left-padded history positions, which repeat the first real transition."""

    sequence_length = len(histories[0])
    masks = torch.ones((len(histories), sequence_length), dtype=torch.bool)
    for row, history in enumerate(histories):
        padding = sequence_length - len(set(history))
        if padding:
            masks[row, :padding] = False
    return masks


def _sequence_target_observation_histories(
    histories: list[list[Transition]],
    horizons: list[list[Transition]],
    sequence_length: int,
) -> list[list[Any]]:
    """Build actor-equivalent histories ending at each n-step bootstrap state."""

    if len(histories) != len(horizons):
        raise ValueError("sequence histories and bootstrap horizons must have equal length")
    return [
        _sequence_target_history(history, horizon, sequence_length)
        for history, horizon in zip(histories, horizons, strict=True)
    ]


def _sequence_target_history(
    history: list[Transition], horizon: list[Transition], sequence_length: int
) -> list[Any]:
    if len(history) != sequence_length or not horizon:
        raise ValueError("sequence target history requires full context and a bootstrap horizon")
    current = history[-1]
    first = horizon[0]
    if current.episode_id != first.episode_id or current.step != first.step:
        raise ValueError("bootstrap horizon must begin at the final context transition")
    observations = [transition.observation for transition in history]
    observations.extend(transition.next_observation for transition in horizon)
    return observations[-sequence_length:]


@dataclass(frozen=True, slots=True)
class _SequenceBatchLayout:
    batch_size: int
    sequence_length: int
    masks: torch.Tensor | None = None


def _reshape_batch_sequences(batch: TrainingBatch, layout: _SequenceBatchLayout) -> TrainingBatch:
    batch_size = layout.batch_size
    sequence_length = layout.sequence_length
    return replace(
        batch,
        data=_reshape_sequence_batch(batch.data, batch_size, sequence_length),
        observations=_reshape_sequence_batch(batch.observations, batch_size, sequence_length),
        actions=_reshape_sequence_batch(batch.actions, batch_size, sequence_length),
        rewards=_reshape_sequence_batch(batch.rewards, batch_size, sequence_length),
        next_observations=_reshape_sequence_batch(
            batch.next_observations, batch_size, sequence_length
        ),
        terminated=_reshape_sequence_batch(batch.terminated, batch_size, sequence_length),
        truncated=_reshape_sequence_batch(batch.truncated, batch_size, sequence_length),
        bootstrap_discounts=_reshape_sequence_batch(
            batch.bootstrap_discounts, batch_size, sequence_length
        ),
        masks=_sequence_masks(layout),
    )


def _sequence_masks(layout: _SequenceBatchLayout) -> torch.Tensor:
    if layout.masks is not None:
        return layout.masks
    return torch.ones((layout.batch_size, layout.sequence_length), dtype=torch.bool)


@dataclass(frozen=True, slots=True)
class _BatchBuild:
    store: ReplayStore
    pipeline: Any
    transition_ids: list[TransitionId]
    request: BatchRequest
    importance_weights: tuple[float, ...] | None = None
    masks: Any = None
    metadata: Mapping[str, Any] | None = None
    bootstrap_stride: int = 1


@dataclass(frozen=True, slots=True)
class _MaterializedBatch:
    transitions: list[Transition]
    discounts: list[float]


@dataclass(frozen=True, slots=True)
class _BatchAssembly:
    build: _BatchBuild
    materialized: _MaterializedBatch
    standard: Mapping[str, Any] | None
    data: Any


def _make_batch(build: _BatchBuild) -> TrainingBatch:
    return _build_batch(build)


def _build_batch(build: _BatchBuild) -> TrainingBatch:
    materialized = _materialize_batch(build)
    collated = build.pipeline.collate(materialized.transitions)
    standard = _standard_collation(collated)
    data = _batch_data(collated, standard)
    return _training_batch(_BatchAssembly(build, materialized, standard, data))


def _materialize_batch(build: _BatchBuild) -> _MaterializedBatch:
    materializer = getattr(build.store, "materialize_n_step", None)
    if build.bootstrap_stride == 1 and callable(materializer):
        transitions, discounts = cast(
            tuple[list[Transition], list[float]],
            materializer(build.transition_ids, build.request),
        )
        return _MaterializedBatch(transitions, discounts)
    return _materialize_from_snapshot(build)


def _materialize_from_snapshot(build: _BatchBuild) -> _MaterializedBatch:
    horizons, requested_ids = _requested_horizons(build)
    available = dict(zip(requested_ids, build.store.get(requested_ids), strict=True))
    n_step = [
        _n_step_transition(_NStepInput(item, available, build.request, horizon))
        for item, horizon in zip(build.transition_ids, horizons, strict=True)
    ]
    return _MaterializedBatch(
        [item[0] for item in n_step],
        [item[1] for item in n_step],
    )


def _requested_horizons(
    build: _BatchBuild,
) -> tuple[list[list[TransitionId]], list[TransitionId]]:
    horizons = [
        _selection_horizon(build, index, item) for index, item in enumerate(build.transition_ids)
    ]
    requested_ids: list[TransitionId] = []
    seen: set[TransitionId] = set()
    for horizon in horizons:
        for candidate in horizon:
            if candidate not in seen and build.store.contains(candidate):
                seen.add(candidate)
                requested_ids.append(candidate)
    return horizons, requested_ids


def _selection_horizon(
    build: _BatchBuild, index: int, transition_id: TransitionId
) -> list[TransitionId]:
    needs_return = (
        build.bootstrap_stride == 1 or index % build.bootstrap_stride == build.bootstrap_stride - 1
    )
    if not needs_return:
        return [transition_id]
    resolver = getattr(build.store, "n_step_ids", None)
    if callable(resolver):
        return cast(list[TransitionId], resolver(transition_id, build.request.n_step))
    return [transition_id + offset for offset in range(build.request.n_step)]


_STANDARD_BATCH_FIELDS = frozenset(
    {"observations", "actions", "rewards", "next_observations", "terminated", "truncated"}
)


def _standard_collation(data: Any) -> Mapping[str, Any] | None:
    if not isinstance(data, Mapping):
        return None
    if data.get("_trackmaniarl_batch_collated") is not True:
        return None
    return data if _STANDARD_BATCH_FIELDS.issubset(data) else None


def _batch_data(data: Any, standard: Mapping[str, Any] | None) -> Any:
    if standard is None:
        return data
    excluded = _STANDARD_BATCH_FIELDS | {"_trackmaniarl_batch_collated"}
    return {key: value for key, value in standard.items() if key not in excluded}


def _batch_field(standard: Mapping[str, Any] | None, name: str, values: list[Any]) -> Any:
    return standard[name] if standard is not None else tree_collate(values)


@dataclass(frozen=True, slots=True)
class _CollatedFields:
    observations: Any
    actions: Any
    rewards: Any
    next_observations: Any
    terminated: Any
    truncated: Any


def _collated_fields(
    standard: Mapping[str, Any] | None, transitions: list[Transition]
) -> _CollatedFields:
    return _CollatedFields(
        _batch_field(standard, "observations", [item.observation for item in transitions]),
        _batch_field(standard, "actions", [item.action for item in transitions]),
        _batch_field(standard, "rewards", [item.reward for item in transitions]),
        _batch_field(
            standard, "next_observations", [item.next_observation for item in transitions]
        ),
        _batch_field(standard, "terminated", [item.terminated for item in transitions]),
        _batch_field(standard, "truncated", [item.truncated for item in transitions]),
    )


def _training_batch(assembly: _BatchAssembly) -> TrainingBatch:
    build = assembly.build
    transitions = assembly.materialized.transitions
    fields = _collated_fields(assembly.standard, transitions)
    weights = tree_collate(build.importance_weights) if build.importance_weights else None
    return TrainingBatch(
        data=assembly.data,
        observations=fields.observations,
        actions=fields.actions,
        rewards=fields.rewards,
        next_observations=fields.next_observations,
        terminated=fields.terminated,
        truncated=fields.truncated,
        bootstrap_discounts=tree_collate(assembly.materialized.discounts),
        transition_ids=build.transition_ids,
        importance_weights=weights,
        masks=build.masks,
        metadata={**dict(build.metadata or {}), **_behavior_metadata(transitions)},
    )
