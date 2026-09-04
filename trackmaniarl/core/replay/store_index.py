"""Episode indexing and transition linkage for in-memory replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from trackmaniarl.core.data import Transition, TransitionId

if TYPE_CHECKING:
    from trackmaniarl.core.replay.store import InMemoryReplayStore


@dataclass(frozen=True, slots=True)
class _EpisodeStepRegistration:
    episode_code: int
    step: int
    transition_id: TransitionId


@dataclass(frozen=True, slots=True)
class _TransitionLink:
    previous_id: TransitionId
    transition_id: TransitionId
    observation: Any


def _episode_code(store: InMemoryReplayStore, episode_id: str | None) -> int:
    if episode_id is None:
        return -1
    existing = store._episode_codes_by_name.get(episode_id)
    if existing is not None:
        return existing
    code = store._next_episode_code
    store._next_episode_code += 1
    store._episode_names[code] = episode_id
    store._episode_codes_by_name[episode_id] = code
    return code


def _previous_transition(
    store: InMemoryReplayStore, transition: Transition, episode_code: int
) -> TransitionId:
    if episode_code < 0 or transition.step is None:
        return _previous_unindexed(store, episode_code)
    steps = store._episode_steps.setdefault(episode_code, {})
    existing = store._episode_step(steps, transition.step)
    if existing >= 0:
        episode_id = store._episode_names[episode_code]
        raise ValueError(
            f"duplicate replay episode step: episode={episode_id!r}, step={transition.step}"
        )
    return store._episode_step(steps, transition.step - 1)


def _previous_unindexed(store: InMemoryReplayStore, episode_code: int) -> TransitionId:
    candidate = store._next_index - 1
    if not store.contains(candidate):
        return -1
    slot = candidate % store.capacity
    if store._terminated[slot] or store._truncated[slot]:
        return -1
    return candidate if int(store._episode_codes[slot]) == episode_code else -1


def _register_episode_step(
    store: InMemoryReplayStore, registration: _EpisodeStepRegistration
) -> None:
    steps = store._episode_steps.setdefault(registration.episode_code, {})
    steps[registration.step] = registration.transition_id
    successor = store._episode_step(steps, registration.step + 1)
    if successor >= 0:
        assert store._observations is not None
        store._link_previous(
            registration.transition_id,
            successor,
            store._observations.read(successor % store.capacity),
        )
    slot = registration.transition_id % store.capacity
    if store._terminated[slot] or store._truncated[slot]:
        store._episode_terminal_steps[registration.episode_code] = registration.step
    store._release_completed_episode(registration.episode_code)


def _episode_step(
    store: InMemoryReplayStore, steps: dict[int, TransitionId], step: int
) -> TransitionId:
    transition_id = steps.get(step, -1)
    if transition_id >= 0 and not store.contains(transition_id):
        steps.pop(step, None)
        return -1
    return transition_id


def _release_completed_episode(store: InMemoryReplayStore, episode_code: int) -> None:
    terminal_step = store._episode_terminal_steps.get(episode_code)
    if terminal_step is None:
        return
    steps = store._episode_steps[episode_code]
    if len(steps) < terminal_step + 1:
        return
    if all(store._episode_step(steps, step) >= 0 for step in range(terminal_step + 1)):
        store._episode_steps.pop(episode_code)
        store._episode_terminal_steps.pop(episode_code)


def _link_previous(store: InMemoryReplayStore, link: _TransitionLink) -> None:
    if link.previous_id < 0 or not store.contains(link.previous_id):
        return
    previous_slot = link.previous_id % store.capacity
    transition_slot = link.transition_id % store.capacity
    if not _can_link(store, previous_slot, transition_slot):
        return
    store._next_ids[previous_slot] = link.transition_id
    store._previous_ids[transition_slot] = link.previous_id
    previous_next = store._next_overrides.get(link.previous_id)
    if (
        link.transition_id > link.previous_id
        and previous_next is not None
        and store._tree_equal(previous_next, link.observation)
    ):
        store._next_overrides.pop(link.previous_id)


def _can_link(store: InMemoryReplayStore, previous_slot: int, transition_slot: int) -> bool:
    return not (
        store._terminated[previous_slot]
        or store._truncated[previous_slot]
        or store._episode_codes[previous_slot] != store._episode_codes[transition_slot]
    )


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))
