"""Eligibility and sequence queries for in-memory replay."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trackmaniarl.core.data import TransitionId
from trackmaniarl.core.replay.store_support import _ReplayChange

if TYPE_CHECKING:
    from trackmaniarl.core.replay.store import InMemoryReplayStore


@dataclass(frozen=True, slots=True)
class _EligibleSample:
    store: InMemoryReplayStore
    n_step: int
    batch_size: int
    rng: random.Random


@dataclass(frozen=True, slots=True)
class _NextHistoryRequest:
    store: InMemoryReplayStore
    transition_id: TransitionId
    n_step: int
    sequence_length: int


def eligible_transition_ids(store: InMemoryReplayStore, n_step: int) -> list[TransitionId]:
    """Return complete n-step starts without retaining a second full ID index."""

    with store._lock:
        return [
            transition_id
            for transition_id in range(store._next_index - store._size, store._next_index)
            if store._is_n_step_eligible_locked(transition_id, n_step)
        ]


def sample_eligible_ids(request: _EligibleSample) -> list[TransitionId]:
    """Draw complete starts by bounded rejection from the dense ID interval."""

    store = request.store
    with store._lock:
        if store._size < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} complete n-step transitions, replay has {store._size}"
            )
        chosen = _rejection_sample(request)
        return chosen if len(chosen) == request.batch_size else _fallback_sample(request)


def _rejection_sample(request: _EligibleSample) -> list[TransitionId]:
    chosen: list[TransitionId] = []
    chosen_set: set[TransitionId] = set()
    attempts = 0
    lower = request.store._next_index - request.store._size
    while len(chosen) < request.batch_size and attempts < request.batch_size * 32:
        candidate = request.rng.randrange(lower, request.store._next_index)
        attempts += 1
        if candidate not in chosen_set and request.store._is_n_step_eligible_locked(
            candidate, request.n_step
        ):
            chosen.append(candidate)
            chosen_set.add(candidate)
    return chosen


def _fallback_sample(request: _EligibleSample) -> list[TransitionId]:
    eligible = request.store.eligible_transition_ids(request.n_step)
    if len(eligible) < request.batch_size:
        raise RuntimeError(
            f"Need {request.batch_size} complete n-step transitions, replay has {len(eligible)}"
        )
    return request.rng.sample(eligible, request.batch_size)


def is_n_step_eligible(
    store: InMemoryReplayStore, transition_id: TransitionId, n_step: int
) -> bool:
    with store._lock:
        return store._is_n_step_eligible_locked(transition_id, n_step)


def n_step_ids(
    store: InMemoryReplayStore, transition_id: TransitionId, n_step: int
) -> list[TransitionId]:
    """Resolve an episode-local horizon even when actors are interleaved."""

    with store._lock:
        if not store.contains(transition_id):
            return []
        result: list[TransitionId] = []
        candidate_id = transition_id
        for _ in range(n_step):
            if not store.contains(candidate_id):
                break
            result.append(candidate_id)
            slot = candidate_id % store.capacity
            if store._terminated[slot] or store._truncated[slot]:
                break
            candidate_id = int(store._next_ids[slot])
        return result


def affected_n_step_starts(
    store: InMemoryReplayStore, transition_id: TransitionId, n_step: int
) -> list[TransitionId]:
    """Return starts whose eligibility can change after this append."""

    with store._lock:
        if not store.contains(transition_id):
            return []
        return store._predecessor_ids_locked(transition_id, n_step)


def history_ids(
    store: InMemoryReplayStore, transition_id: TransitionId, sequence_length: int
) -> list[TransitionId]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    with store._lock:
        if not store.contains(transition_id):
            return []
        result: list[TransitionId] = []
        candidate = transition_id
        for _ in range(sequence_length):
            if not store.contains(candidate):
                break
            result.append(candidate)
            slot = candidate % store.capacity
            candidate = int(store._previous_ids[slot])
        result.reverse()
        return [result[0]] * (sequence_length - len(result)) + result


def next_history_observations(request: _NextHistoryRequest) -> list[Any]:
    """Return recurrent history ending at the resolved n-step next state."""

    if request.sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    store = request.store
    with store._lock:
        horizon = store.n_step_ids(request.transition_id, request.n_step)
        if not horizon:
            return []
        final_id = horizon[-1]
        history = store.history_ids(final_id, max(1, request.sequence_length - 1))
        if request.sequence_length == 1:
            history = []
        observations = [store._transition(item).observation for item in history]
        observations.append(store._transition(final_id).next_observation)
        return observations


def sampling_pace_s(store: InMemoryReplayStore, transition_id: TransitionId) -> float:
    with store._lock:
        if not store.contains(transition_id):
            return float("inf")
        return float(store._sampling_pace[transition_id % store.capacity])


def demo_flags(store: InMemoryReplayStore, transition_ids: list[TransitionId]) -> list[bool]:
    with store._lock:
        return [
            store.contains(transition_id)
            and bool(store._demo_flags[transition_id % store.capacity])
            for transition_id in transition_ids
        ]


def changes_since(
    store: InMemoryReplayStore, revision: int | None
) -> tuple[int, list[_ReplayChange] | None]:
    """Return append/eviction changes since a sampler's last observed revision."""

    with store._lock:
        if revision is None:
            return store._revision, None
        if revision == store._revision:
            return store._revision, []
        if not store._changes or revision < store._changes[0][0] - 1:
            return store._revision, None
        return store._revision, [
            change for change_revision, change in store._changes if change_revision > revision
        ]


def _is_n_step_eligible_locked(
    store: InMemoryReplayStore, transition_id: TransitionId, n_step: int
) -> bool:
    if n_step < 1 or not store.contains(transition_id):
        return False
    candidate_id = transition_id
    for _ in range(n_step):
        if not store.contains(candidate_id):
            return False
        slot = candidate_id % store.capacity
        if store._terminated[slot] or store._truncated[slot]:
            return True
        candidate_id = int(store._next_ids[slot])
    return True


def _predecessor_ids_locked(
    store: InMemoryReplayStore, transition_id: TransitionId, n_step: int
) -> list[TransitionId]:
    result: list[TransitionId] = []
    candidate = transition_id
    for _ in range(n_step):
        if not store.contains(candidate):
            break
        result.append(candidate)
        candidate = int(store._previous_ids[candidate % store.capacity])
    return result
