"""Checkpoint persistence for in-memory replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from trackmaniarl.core.replay.store_pace import validate_episode_sampling_paces
from trackmaniarl.core.replay.store_support import _is_demo, _TreeColumns

if TYPE_CHECKING:
    from trackmaniarl.core.replay.store import InMemoryReplayStore


_STATE_FORMAT = "columnar-v2"
_LEGACY_STATE_FORMAT = "columnar-v1"
_BASE_STATE_FIELDS = frozenset(
    {
        "format",
        "capacity",
        "size",
        "next_index",
        "episode_names",
        "next_overrides",
        "info",
    }
)
_POPULATED_STATE_FIELDS = frozenset(
    {
        "observations",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "episode_codes",
        "steps",
        "previous_ids",
        "next_ids",
        "sampling_pace",
    }
)


@dataclass(frozen=True, slots=True)
class _TreeRestore:
    name: str
    slots: np.ndarray[Any, np.dtype[np.int64]]


def state_dict(store: InMemoryReplayStore) -> dict[str, Any]:
    with store._lock:
        empty = _empty_state(store)
        if store._observations is None or store._actions is None:
            return empty
        return _populated_state(store, empty)


def _empty_state(store: InMemoryReplayStore) -> dict[str, Any]:
    return {
        "format": _STATE_FORMAT,
        "capacity": store.capacity,
        "size": store._size,
        "next_index": store._next_index,
        "episode_names": dict(store._episode_names),
        "episode_sampling_paces": dict(store._episode_sampling_paces),
        "next_overrides": dict(store._next_overrides),
        "info": dict(store._info),
    }


def _populated_state(store: InMemoryReplayStore, empty: dict[str, Any]) -> dict[str, Any]:
    assert store._observations is not None
    assert store._actions is not None
    slots = store._snapshot_slots()
    arrays = _snapshot_arrays(store, slots)
    return {
        **empty,
        "observations": store._observations.snapshot(slots),
        "actions": store._actions.snapshot(slots),
        **arrays,
    }


def _snapshot_arrays(
    store: InMemoryReplayStore, slots: slice | np.ndarray[Any, np.dtype[np.int64]]
) -> dict[str, np.ndarray[Any, Any]]:
    names = (
        "rewards",
        "terminated",
        "truncated",
        "episode_codes",
        "steps",
        "previous_ids",
        "next_ids",
        "sampling_pace",
    )
    return {
        name: np.array(getattr(store, f"_{name}")[slots], copy=True, order="C") for name in names
    }


def _snapshot_slots(
    store: InMemoryReplayStore,
) -> slice | np.ndarray[Any, np.dtype[np.int64]]:
    first_id = store._next_index - store._size
    first_slot = first_id % store.capacity
    if first_slot + store._size <= store.capacity:
        return slice(first_slot, first_slot + store._size)
    ids = np.arange(first_id, store._next_index, dtype=np.int64)
    return ids % store.capacity


def load_state_dict(store: InMemoryReplayStore, state: Mapping[str, Any]) -> None:
    _validate_state(state)
    with store._lock:
        _validate_checkpoint_capacity(store, int(state["capacity"]))
        store._reset_arrays()
        store._size = int(state["size"])
        store._next_index = int(state["next_index"])
        if store._size:
            _restore_entries(store, state)
        _restore_metadata(store, state)
        store._rebuild_episode_steps()
        store._rebuild_reference_state()
        store._revision += 1
        store._changes.clear()


def _validate_state(state: Mapping[str, Any]) -> None:
    _require_fields(state, _BASE_STATE_FIELDS, "replay checkpoint")
    state_format = state["format"]
    if state_format not in {_STATE_FORMAT, _LEGACY_STATE_FORMAT}:
        raise ValueError(f"unsupported replay checkpoint format: {state['format']!r}")
    if state_format == _STATE_FORMAT:
        _require_fields(state, frozenset({"episode_sampling_paces"}), "replay checkpoint")
        validate_episode_sampling_paces(state["episode_sampling_paces"])
    if int(state["size"]):
        _require_fields(state, _POPULATED_STATE_FIELDS, "populated replay checkpoint")


def _require_fields(state: Mapping[str, Any], required: frozenset[str], label: str) -> None:
    missing = required.difference(state)
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(sorted(missing))}")


def _validate_checkpoint_capacity(store: InMemoryReplayStore, capacity: int) -> None:
    if capacity > store.capacity:
        raise ValueError(
            f"Replay checkpoint capacity {capacity} exceeds configured capacity {store.capacity}"
        )


def _restore_entries(store: InMemoryReplayStore, state: Mapping[str, Any]) -> None:
    ids = np.arange(
        store._next_index - store._size,
        store._next_index,
        dtype=np.int64,
    )
    slots = ids % store.capacity
    store._ids[slots] = ids
    store._observations = _restore_tree(store, state, _TreeRestore("observations", slots))
    store._actions = _restore_tree(store, state, _TreeRestore("actions", slots))
    _restore_numeric_columns(store, state, slots)
    store._sampling_pace[slots] = state["sampling_pace"]


def _restore_numeric_columns(
    store: InMemoryReplayStore, state: Mapping[str, Any], slots: np.ndarray[Any, Any]
) -> None:
    names = (
        "rewards",
        "terminated",
        "truncated",
        "episode_codes",
        "steps",
        "previous_ids",
        "next_ids",
    )
    for name in names:
        getattr(store, f"_{name}")[slots] = state[name]


def _restore_tree(
    store: InMemoryReplayStore,
    state: Mapping[str, Any],
    request: _TreeRestore,
) -> _TreeColumns:
    return _TreeColumns.restore(
        store.capacity,
        cast(Mapping[str, Any], state[request.name]),
        request.slots,
    )


def _restore_metadata(store: InMemoryReplayStore, state: Mapping[str, Any]) -> None:
    names = cast(Mapping[Any, Any], state["episode_names"])
    store._episode_names = {int(code): str(name) for code, name in names.items()}
    store._episode_codes_by_name = {name: code for code, name in store._episode_names.items()}
    store._next_episode_code = max(store._episode_names, default=-1) + 1
    raw_paces = state["episode_sampling_paces"] if state["format"] == _STATE_FORMAT else {}
    paces = cast(Mapping[str, float], raw_paces)
    store._episode_sampling_paces = dict(paces)
    store._next_overrides = dict(state["next_overrides"])
    store._info = dict(state["info"])


def _reset_arrays(store: InMemoryReplayStore) -> None:
    store._ids.fill(-1)
    store._episode_codes.fill(-1)
    store._steps.fill(-1)
    store._previous_ids.fill(-1)
    store._next_ids.fill(-1)
    store._sampling_pace.fill(np.inf)
    store._demo_flags.fill(False)
    store._demo_count = 0
    store._observations = None
    store._actions = None
    store._next_overrides.clear()
    store._info.clear()
    store._episode_steps.clear()
    store._episode_terminal_steps.clear()
    store._episode_refcounts.clear()
    store._episode_sampling_paces.clear()


def _rebuild_reference_state(store: InMemoryReplayStore) -> None:
    for transition_id in range(store._next_index - store._size, store._next_index):
        slot = transition_id % store.capacity
        code = int(store._episode_codes[slot])
        if code >= 0:
            store._episode_refcounts[code] = store._episode_refcounts.get(code, 0) + 1
        is_demo = _is_demo(store._info.get(transition_id, {}))
        store._demo_flags[slot] = is_demo
        store._demo_count += int(is_demo)


def _rebuild_episode_steps(store: InMemoryReplayStore) -> None:
    store._episode_steps.clear()
    store._episode_terminal_steps.clear()
    for transition_id in range(store._next_index - store._size, store._next_index):
        slot = transition_id % store.capacity
        code = int(store._episode_codes[slot])
        step = int(store._steps[slot])
        if code < 0 or step < 0:
            continue
        store._episode_steps.setdefault(code, {})[step] = transition_id
        if store._terminated[slot] or store._truncated[slot]:
            store._episode_terminal_steps[code] = step
    for code in tuple(store._episode_terminal_steps):
        store._release_completed_episode(code)
