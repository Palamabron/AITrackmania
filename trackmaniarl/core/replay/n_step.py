"""Episode-safe n-step transition materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from trackmaniarl.core.data import BatchRequest, Transition, TransitionId


@dataclass(frozen=True, slots=True)
class _NStepInput:
    transition_id: TransitionId
    available: Mapping[TransitionId, Transition]
    request: BatchRequest
    horizon: list[TransitionId] | None = None


@dataclass(slots=True)
class _NStepAccumulator:
    first: Transition
    current: Transition
    reward: float = 0.0
    discount: float = 1.0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False

    def add(self, candidate: Transition, gamma: float) -> None:
        self.current = candidate
        self.reward += self.discount * candidate.reward
        self.steps += 1
        self.terminated = candidate.terminated
        self.truncated = candidate.truncated
        if not (self.terminated or self.truncated):
            self.discount *= gamma

    def result(self, gamma: float) -> tuple[Transition, float]:
        transition = replace(
            self.first,
            reward=self.reward,
            next_observation=self.current.next_observation,
            terminated=self.terminated,
            truncated=self.truncated,
        )
        bootstrap_discount = 0.0 if self.terminated else gamma**self.steps
        return transition, bootstrap_discount


def _n_step_transition(item: _NStepInput) -> tuple[Transition, float]:
    return _build_n_step(item)


def _build_n_step(item: _NStepInput) -> tuple[Transition, float]:
    accumulator = _NStepAccumulator(
        item.available[item.transition_id], item.available[item.transition_id]
    )
    ordered_ids = item.horizon or [
        item.transition_id + offset for offset in range(item.request.n_step)
    ]
    for offset, current_id in enumerate(ordered_ids):
        candidate = _n_step_candidate(item, current_id, offset)
        if candidate is None:
            break
        accumulator.add(candidate, item.request.gamma)
        if accumulator.terminated or accumulator.truncated:
            break
    if accumulator.steps == 0:
        raise RuntimeError(f"Transition {item.transition_id} is no longer available")
    return accumulator.result(item.request.gamma)


def _n_step_candidate(
    item: _NStepInput, current_id: TransitionId, offset: int
) -> Transition | None:
    candidate = item.available.get(current_id)
    first = item.available[item.transition_id]
    if candidate is None or candidate.episode_id != first.episode_id:
        return None
    if candidate.step is None or first.step is None:
        return candidate
    return candidate if candidate.step == first.step + offset else None
