"""Contiguity validation for recurrent replay selections."""

from __future__ import annotations

from dataclasses import dataclass

from trackmaniarl.core.data import Transition, TransitionId


@dataclass(frozen=True, slots=True)
class _RolloutEdge:
    previous_id: TransitionId
    current_id: TransitionId
    previous: Transition
    current: Transition


def _is_contiguous_rollout(indices: list[TransitionId], transitions: list[Transition]) -> bool:
    pairs = zip(indices[:-1], indices[1:], transitions[:-1], transitions[1:], strict=True)
    return all(
        _extends_rollout(_RolloutEdge(previous_id, current_id, previous, current))
        for previous_id, current_id, previous, current in pairs
    )


def _extends_rollout(edge: _RolloutEdge) -> bool:
    if edge.current_id != edge.previous_id + 1:
        return False
    same_episode = edge.current.episode_id == edge.previous.episode_id
    if same_episode:
        return _extends_episode(edge.previous, edge.current)
    return (edge.previous.terminated or edge.previous.truncated) and edge.current.step in {0, None}


def _extends_episode(previous: Transition, current: Transition) -> bool:
    if previous.terminated or previous.truncated:
        return False
    return previous.step is None or current.step == previous.step + 1
