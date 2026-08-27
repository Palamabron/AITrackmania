"""Fallback implementation for replay stores without incremental hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, Transition, TransitionId
from trackmaniarl.core.replay.batches import _eligible_n_step_ids
from trackmaniarl.core.replay.store import _is_demo

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler

type _FallbackResult = tuple[
    list[TransitionId],
    tuple[float, ...],
    float,
    dict[str, float],
    tuple[bool, ...],
    tuple[bool, ...],
]


@dataclass(frozen=True, slots=True)
class _FallbackContext:
    sampler: PrioritizedSampler
    store: ReplayStore
    request: BatchRequest


@dataclass(frozen=True, slots=True)
class _FallbackFlags:
    elite: Mapping[TransitionId, bool]
    demo: Mapping[TransitionId, bool]
    expert: Mapping[TransitionId, bool]


@dataclass(frozen=True, slots=True)
class _FallbackSelection:
    sampler: PrioritizedSampler
    transition_ids: list[TransitionId]
    probabilities: list[float]
    expert_by_id: Mapping[TransitionId, bool]
    batch_size: int


@dataclass(frozen=True, slots=True)
class _GroupCounts:
    expert: int
    non_expert: int


@dataclass(frozen=True, slots=True)
class _GroupRequest:
    expert: bool
    count: int


@dataclass(frozen=True, slots=True)
class _FallbackDraw:
    transition_ids: list[TransitionId]
    probabilities: list[float]


@dataclass(frozen=True, slots=True)
class _SampledFlags:
    transition_ids: list[TransitionId]
    demo: tuple[bool, ...]
    expert: tuple[bool, ...]
    elite: tuple[bool, ...]


def _sample_fallback(
    sampler: PrioritizedSampler, store: ReplayStore, request: BatchRequest
) -> _FallbackResult:
    context = _FallbackContext(sampler, store, request)
    with sampler._lock:
        return _sample_locked(context)


def _sample_locked(context: _FallbackContext) -> _FallbackResult:
    transition_ids = _eligible_n_step_ids(context.store, context.request)
    _validate_replay_size(transition_ids, context.request.batch_size)
    context.sampler._synchronize_fallback(transition_ids)
    transitions = context.store.get(transition_ids)
    flags = _fallback_flags(context.sampler, transition_ids, transitions)
    probabilities = _fallback_probabilities(context.sampler, transition_ids, flags.elite)
    selection = _FallbackSelection(
        context.sampler,
        transition_ids,
        probabilities,
        flags.expert,
        context.request.batch_size,
    )
    chosen, chosen_probabilities = _stratified_fallback_choices(selection)
    return _fallback_result(context, flags, _FallbackDraw(chosen, chosen_probabilities))


def _validate_replay_size(transition_ids: list[TransitionId], batch_size: int) -> None:
    if len(transition_ids) < batch_size:
        raise RuntimeError(f"Need {batch_size} transitions, replay has {len(transition_ids)}")


def _fallback_flags(
    sampler: PrioritizedSampler,
    transition_ids: list[TransitionId],
    transitions: list[Transition],
) -> _FallbackFlags:
    pairs = tuple(zip(transition_ids, transitions, strict=True))
    elite = {item: _is_elite(sampler, transition) for item, transition in pairs}
    demo = {item: _is_demo(transition.info) for item, transition in pairs}
    expert = {item: demo[item] and _is_expert(sampler, transition) for item, transition in pairs}
    return _FallbackFlags(elite, demo, expert)


def _is_elite(sampler: PrioritizedSampler, transition: Transition) -> bool:
    lap_time = float(transition.info.get("sampling/projected_lap_time_s", float("inf")))
    return sampler.elite_time_s is not None and lap_time <= sampler.elite_time_s


def _is_expert(sampler: PrioritizedSampler, transition: Transition) -> bool:
    lap_time = float(transition.info.get("sampling/projected_lap_time_s", float("inf")))
    return sampler.expert_demo_time_s is not None and lap_time <= sampler.expert_demo_time_s


def _fallback_probabilities(
    sampler: PrioritizedSampler,
    transition_ids: list[TransitionId],
    elite_by_id: Mapping[TransitionId, bool],
) -> list[float]:
    scaled = [_scaled_priority(sampler, item, elite_by_id) for item in transition_ids]
    total = sum(scaled)
    probabilities = (
        [weight / total for weight in scaled]
        if total > 0.0
        else [1 / len(transition_ids)] * len(transition_ids)
    )
    return _mix_uniform(sampler, probabilities)


def _scaled_priority(
    sampler: PrioritizedSampler,
    transition_id: TransitionId,
    elite_by_id: Mapping[TransitionId, bool],
) -> float:
    boost = sampler.elite_priority_boost if elite_by_id[transition_id] else 1.0
    return float(sampler._fallback_priorities[transition_id] ** sampler.alpha * boost)


def _mix_uniform(sampler: PrioritizedSampler, probabilities: list[float]) -> list[float]:
    if not sampler.uniform_mix:
        return probabilities
    uniform = 1.0 / len(probabilities)
    return [
        (1.0 - sampler.uniform_mix) * probability + sampler.uniform_mix * uniform
        for probability in probabilities
    ]


def _stratified_fallback_choices(
    selection: _FallbackSelection,
) -> tuple[list[TransitionId], list[float]]:
    if selection.sampler.expert_fraction == 0.0 or all(selection.expert_by_id.values()):
        return _unstratified_choices(selection)
    expert_count = round(selection.sampler.expert_fraction * selection.batch_size)
    counts = _GroupCounts(expert_count, selection.batch_size - expert_count)
    return _fallback_group_choices(selection, counts)


def _unstratified_choices(
    selection: _FallbackSelection,
) -> tuple[list[TransitionId], list[float]]:
    chosen = selection.sampler._rng.choices(
        selection.transition_ids,
        weights=selection.probabilities,
        k=selection.batch_size,
    )
    by_id = dict(zip(selection.transition_ids, selection.probabilities, strict=True))
    return chosen, [by_id[transition_id] for transition_id in chosen]


def _fallback_group_choices(
    selection: _FallbackSelection, counts: _GroupCounts
) -> tuple[list[TransitionId], list[float]]:
    grouped: list[tuple[TransitionId, float]] = []
    requests = (_GroupRequest(True, counts.expert), _GroupRequest(False, counts.non_expert))
    for request in requests:
        grouped.extend(_sample_group(selection, counts, request))
    selection.sampler._rng.shuffle(grouped)
    return [item[0] for item in grouped], [item[1] for item in grouped]


def _sample_group(
    selection: _FallbackSelection, counts: _GroupCounts, request: _GroupRequest
) -> list[tuple[TransitionId, float]]:
    if request.count == 0:
        return []
    candidates = _group_candidates(selection, request)
    if not candidates:
        label = "expert demonstration" if request.expert else "non-expert"
        raise RuntimeError(f"Prioritized replay has no {label} transitions")
    ids, weights = zip(*candidates, strict=True)
    conditional = [weight / sum(weights) for weight in weights]
    selected = selection.sampler._rng.choices(ids, weights=conditional, k=request.count)
    by_id = dict(zip(ids, conditional, strict=True))
    fraction = request.count / (counts.expert + counts.non_expert)
    return [(transition_id, fraction * by_id[transition_id]) for transition_id in selected]


def _group_candidates(
    selection: _FallbackSelection, request: _GroupRequest
) -> list[tuple[TransitionId, float]]:
    pairs = zip(selection.transition_ids, selection.probabilities, strict=True)
    return [
        (transition_id, probability)
        for transition_id, probability in pairs
        if selection.expert_by_id[transition_id] is request.expert
    ]


def _fallback_result(
    context: _FallbackContext,
    flags: _FallbackFlags,
    draw: _FallbackDraw,
) -> _FallbackResult:
    beta = context.sampler.beta if context.request.beta is None else context.request.beta
    weights = _normalized_importance(len(flags.elite), draw.probabilities, beta)
    sampled = _sampled_flags(flags, draw.transition_ids)
    metadata = _fallback_metadata(flags, sampled)
    return draw.transition_ids, weights, beta, metadata, sampled.demo, sampled.expert


def _sampled_flags(flags: _FallbackFlags, chosen: list[TransitionId]) -> _SampledFlags:
    return _SampledFlags(
        chosen,
        tuple(flags.demo[item] for item in chosen),
        tuple(flags.expert[item] for item in chosen),
        tuple(flags.elite[item] for item in chosen),
    )


def _normalized_importance(
    population_size: int, probabilities: list[float], beta: float
) -> tuple[float, ...]:
    weights = [(population_size * probability) ** (-beta) for probability in probabilities]
    maximum = max(weights)
    return tuple(weight / maximum for weight in weights)


def _fallback_metadata(flags: _FallbackFlags, sampled: _SampledFlags) -> dict[str, float]:
    return {
        "replay/elite_active_fraction": sum(flags.elite.values()) / len(flags.elite),
        "replay/elite_sample_fraction": sum(sampled.elite) / len(sampled.transition_ids),
        "replay/demo_sample_fraction": sum(sampled.demo) / len(sampled.transition_ids),
        "replay/expert_demo_sample_fraction": sum(sampled.expert) / len(sampled.transition_ids),
    }
