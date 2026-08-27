"""Priority persistence and mutation for proportional replay."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from trackmaniarl.core.data import PriorityUpdate, TransitionId

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler


_STATE_FORMAT = "array-per-v1"
_STATE_FIELDS = frozenset(
    {
        "format",
        "priorities",
        "slot_ids",
        "fallback_priorities",
        "maximum_priority",
        "rng",
    }
)


def update_priorities(sampler: PrioritizedSampler, update: PriorityUpdate) -> None:
    with sampler._lock:
        for transition_id, priority in zip(update.transition_ids, update.priorities, strict=True):
            value = abs(float(priority)) + sampler.priority_epsilon
            if not isfinite(value):
                raise ValueError("PER priorities must be finite")
            sampler._maximum_priority = max(sampler._maximum_priority, value)
            if sampler._tree is None:
                if transition_id in sampler._fallback_priorities:
                    sampler._fallback_priorities[transition_id] = value
                continue
            slot = transition_id % sampler._tree.size
            if sampler._slot_ids[slot] == transition_id and sampler._tree.leaves[slot] > 0.0:
                sampler._priorities[slot] = value
                sampler._set_slot_weight(slot)


def _synchronize_fallback(sampler: PrioritizedSampler, transition_ids: list[TransitionId]) -> None:
    active = set(transition_ids)
    sampler._fallback_priorities = {
        index: priority
        for index, priority in sampler._fallback_priorities.items()
        if index in active
    }
    maximum = max(sampler._fallback_priorities.values(), default=1.0)
    for index in active:
        sampler._fallback_priorities.setdefault(index, maximum)


def state_dict(sampler: PrioritizedSampler) -> dict[str, Any]:
    with sampler._lock:
        return {
            "format": _STATE_FORMAT,
            "priorities": sampler._priorities.copy(),
            "slot_ids": sampler._slot_ids.copy(),
            "fallback_priorities": dict(sampler._fallback_priorities),
            "maximum_priority": sampler._maximum_priority,
            "rng": sampler._rng.getstate(),
        }


def load_state_dict(sampler: PrioritizedSampler, state: Mapping[str, Any]) -> None:
    _validate_state(state)
    _load_array_state(sampler, state)
    _reset_runtime_index(sampler)
    sampler._rng.setstate(state["rng"])


def _validate_state(state: Mapping[str, Any]) -> None:
    missing = _STATE_FIELDS.difference(state)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"prioritized replay checkpoint missing required fields: {fields}")
    if state["format"] != _STATE_FORMAT:
        raise ValueError(f"unsupported prioritized replay checkpoint format: {state['format']!r}")


def _load_array_state(sampler: PrioritizedSampler, state: Mapping[str, Any]) -> None:
    sampler._priorities = np.asarray(state["priorities"], dtype=np.float32)
    sampler._slot_ids = np.asarray(state["slot_ids"], dtype=np.int64)
    fallback = cast(Mapping[Any, Any], state["fallback_priorities"])
    sampler._fallback_priorities = {int(key): float(value) for key, value in fallback.items()}
    sampler._maximum_priority = float(state["maximum_priority"])


def _reset_runtime_index(sampler: PrioritizedSampler) -> None:
    sampler._active_count = 0
    sampler._elite_active_count = 0
    sampler._expert_active_count = 0
    sampler._tree = None
    sampler._uniform_tree = None
    sampler._expert_tree = None
    sampler._expert_uniform_tree = None
    sampler._non_expert_tree = None
    sampler._non_expert_uniform_tree = None
    sampler._replay_revision = None
    sampler._n_step = None
    sampler._sequence_length = None
