"""Probability and tree draws for incremental prioritized replay."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from trackmaniarl.core.data import BatchRequest, TransitionId
from trackmaniarl.core.replay.store import _IncrementalReplayStore
from trackmaniarl.core.replay.structures import _FenwickTree

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler


@dataclass(frozen=True, slots=True)
class _GroupDraw:
    tree: _FenwickTree
    uniform_tree: _FenwickTree
    active_count: int
    count: int
    group_fraction: float


@dataclass(frozen=True, slots=True)
class _DrawResult:
    transition_ids: list[TransitionId]
    probabilities: list[float]
    beta: float
    transition_count: int


def _sample_incremental_ids(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    request: BatchRequest,
) -> tuple[list[TransitionId], tuple[float, ...], float, dict[str, float]]:
    with sampler._lock:
        sampler._synchronize_incremental_store(store, request.n_step, request.sequence_length)
        _validate_active_count(sampler, request.batch_size)
        transition_ids, probabilities = _incremental_choices(sampler, request)
        beta = sampler.beta if request.beta is None else request.beta
        result = _DrawResult(transition_ids, probabilities, beta, request.transition_count)
        weights = _normalized_weights(sampler._active_count, result)
        metadata = _incremental_metadata(sampler, store, result)
        return transition_ids, weights, beta, metadata


def _validate_active_count(sampler: PrioritizedSampler, batch_size: int) -> None:
    if sampler._active_count < batch_size:
        raise RuntimeError(f"Need {batch_size} transitions, replay has {sampler._active_count}")


def _normalized_weights(active_count: int, result: _DrawResult) -> tuple[float, ...]:
    weights = [
        (active_count * probability) ** (-result.beta) for probability in result.probabilities
    ]
    maximum = max(weights)
    return tuple(weight / maximum for weight in weights)


def _incremental_choices(
    sampler: PrioritizedSampler, request: BatchRequest
) -> tuple[list[TransitionId], list[float]]:
    tree = _required_tree(sampler._tree)
    uniform = _required_tree(sampler._uniform_tree)
    expert_fraction = sampler._expert_fraction_at(request.transition_count)
    if expert_fraction == 0.0:
        return _draw_incremental_group(
            sampler, _GroupDraw(tree, uniform, sampler._active_count, request.batch_size, 1.0)
        )
    expert = _expert_group(sampler, request.batch_size, 1.0)
    if sampler._expert_active_count == sampler._active_count:
        return _draw_incremental_group(sampler, expert)
    return _stratified_incremental_choices(sampler, request.batch_size, expert_fraction)


def _required_tree(tree: _FenwickTree | None) -> _FenwickTree:
    if tree is None:
        raise RuntimeError("prioritized replay index is not initialized")
    return tree


def _expert_group(sampler: PrioritizedSampler, count: int, group_fraction: float) -> _GroupDraw:
    return _GroupDraw(
        _required_tree(sampler._expert_tree),
        _required_tree(sampler._expert_uniform_tree),
        sampler._expert_active_count,
        count,
        group_fraction,
    )


def _stratified_incremental_choices(
    sampler: PrioritizedSampler, batch_size: int, expert_fraction: float
) -> tuple[list[TransitionId], list[float]]:
    expert_count = round(expert_fraction * batch_size)
    groups = (
        _expert_group(sampler, expert_count, expert_count / batch_size),
        _non_expert_group(sampler, batch_size - expert_count, batch_size),
    )
    selected: list[tuple[TransitionId, float]] = []
    for group in groups:
        if group.count:
            ids, probabilities = _draw_incremental_group(sampler, group)
            selected.extend(zip(ids, probabilities, strict=True))
    sampler._rng.shuffle(selected)
    return [item[0] for item in selected], [item[1] for item in selected]


def _non_expert_group(sampler: PrioritizedSampler, count: int, batch_size: int) -> _GroupDraw:
    return _GroupDraw(
        _required_tree(sampler._non_expert_tree),
        _required_tree(sampler._non_expert_uniform_tree),
        sampler._active_count - sampler._expert_active_count,
        count,
        count / batch_size,
    )


def _draw_incremental_group(
    sampler: PrioritizedSampler, group: _GroupDraw
) -> tuple[list[TransitionId], list[float]]:
    if group.active_count < 1 or group.tree.total <= 0.0 or group.uniform_tree.total <= 0.0:
        raise RuntimeError("Prioritized replay cannot satisfy expert_fraction")
    transition_ids: list[TransitionId] = []
    probabilities: list[float] = []
    for _ in range(group.count):
        slot = _draw_slot(sampler, group)
        transition_id = int(sampler._slot_ids[slot])
        if transition_id < 0:
            raise RuntimeError("Prioritized replay tree is out of sync")
        transition_ids.append(transition_id)
        probabilities.append(_slot_probability(sampler, group, slot))
    return transition_ids, probabilities


def _draw_slot(sampler: PrioritizedSampler, group: _GroupDraw) -> int:
    use_uniform = sampler.uniform_mix and sampler._rng.random() < sampler.uniform_mix
    tree = group.uniform_tree if use_uniform else group.tree
    return tree.find(sampler._rng.random() * tree.total)


def _slot_probability(sampler: PrioritizedSampler, group: _GroupDraw, slot: int) -> float:
    proportional = float(group.tree.leaves[slot]) / group.tree.total
    uniform = 1.0 / group.active_count
    mixed = (1.0 - sampler.uniform_mix) * proportional + sampler.uniform_mix * uniform
    return group.group_fraction * mixed


def _incremental_metadata(
    sampler: PrioritizedSampler, store: _IncrementalReplayStore, result: _DrawResult
) -> dict[str, float]:
    elite_samples, demo_samples, expert_samples = _sample_counts(sampler, store, result)
    sample_count = len(result.transition_ids)
    return {
        "replay/active_count": float(sampler._active_count),
        "replay/elite_active_fraction": sampler._elite_active_count / sampler._active_count,
        "replay/elite_sample_fraction": elite_samples / sample_count,
        "replay/demo_sample_fraction": demo_samples / sample_count,
        "replay/expert_demo_active_fraction": sampler._expert_active_count / sampler._active_count,
        "replay/expert_demo_sample_fraction": expert_samples / sample_count,
        "replay/expert_demo_target_fraction": sampler._expert_fraction_at(result.transition_count),
        "replay/sampling_probability_mean": fmean(result.probabilities),
        "replay/sampling_probability_min": min(result.probabilities),
        "replay/sampling_probability_max": max(result.probabilities),
    }


def _sample_counts(
    sampler: PrioritizedSampler, store: _IncrementalReplayStore, result: _DrawResult
) -> tuple[int, int, int]:
    tree = _required_tree(sampler._tree)
    elite = sum(int(sampler._elite_slots[item % tree.size]) for item in result.transition_ids)
    demo = sum(store.demo_flags(result.transition_ids))
    expert = sum(int(sampler._expert_slots[item % tree.size]) for item in result.transition_ids)
    return elite, demo, expert
