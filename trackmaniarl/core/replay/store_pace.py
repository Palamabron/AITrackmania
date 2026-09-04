"""Completed-episode pace state for replay sampling."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

from trackmaniarl.core.data import TransitionId
from trackmaniarl.core.replay.store_support import _ReplayChange

if TYPE_CHECKING:
    from trackmaniarl.core.replay.store import InMemoryReplayStore


def transition_sampling_pace(
    store: InMemoryReplayStore,
    episode_id: str | None,
    transition_info: MutableMapping[str, Any],
) -> float:
    projected = float(transition_info.pop("sampling/projected_lap_time_s", np.inf))
    if episode_id is None:
        return projected
    return store._episode_sampling_paces.get(episode_id, projected)


def label_episode_sampling_pace(
    store: InMemoryReplayStore, episode_id: str, finish_time_s: float
) -> int:
    if not episode_id:
        raise ValueError("episode_id must be non-empty")
    stored_pace = validated_sampling_pace(finish_time_s)
    with store._lock:
        store._episode_sampling_paces[episode_id] = stored_pace
        transition_ids = _relabelled_transition_ids(store, episode_id, stored_pace)
        if not transition_ids:
            return 0
        _record_reclassification(store, transition_ids)
        return len(transition_ids)


def validated_sampling_pace(value: float) -> float:
    pace = float(value)
    if not isfinite(pace) or pace <= 0.0 or pace > float(np.finfo(np.float32).max):
        raise ValueError("finish_time_s must be a positive finite float32 value")
    stored_pace = float(np.float32(pace))
    if stored_pace <= 0.0:
        raise ValueError("finish_time_s must be a positive finite float32 value")
    return stored_pace


def _relabelled_transition_ids(
    store: InMemoryReplayStore, episode_id: str, stored_pace: float
) -> tuple[TransitionId, ...]:
    episode_code = store._episode_codes_by_name.get(episode_id)
    if episode_code is None:
        return ()
    matching = (store._ids >= 0) & (store._episode_codes == episode_code)
    changed_slots = np.flatnonzero(matching & (store._sampling_pace != stored_pace))
    store._sampling_pace[changed_slots] = stored_pace
    return tuple(int(value) for value in store._ids[changed_slots])


def _record_reclassification(
    store: InMemoryReplayStore, transition_ids: tuple[TransitionId, ...]
) -> None:
    store._revision += 1
    change = _ReplayChange(
        appended=None,
        evicted=None,
        evicted_previous=None,
        evicted_next=None,
        reclassified=transition_ids,
    )
    store._changes.append((store._revision, change))


def validate_episode_sampling_paces(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("episode_sampling_paces must be a mapping")
    for episode_id, pace in value.items():
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_sampling_paces keys must be non-empty strings")
        if type(pace) is not float:
            raise ValueError("episode_sampling_paces values must be floats")
        validated_sampling_pace(pace)
