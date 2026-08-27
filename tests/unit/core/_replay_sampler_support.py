"""Shared fixtures for replay sampler contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from trackmaniarl.core.data import Transition
from trackmaniarl.core.replay import InMemoryReplayStore


class _ReplayOrigin(Enum):
    DEMONSTRATION = "demonstration"
    ONLINE = "online"
    UNMARKED = "unmarked"


@dataclass(frozen=True, slots=True)
class _EpisodeSpec:
    episode_id: str
    steps: int
    pace_s: float
    origin: _ReplayOrigin = _ReplayOrigin.UNMARKED


class _BasicReplayStore:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def append(self, transition: Transition) -> int:
        self.transitions.append(transition)
        return len(self.transitions) - 1

    def get(self, transition_ids: list[int]) -> list[Transition]:
        return [self.transitions[transition_id] for transition_id in transition_ids]

    def available_ids(self) -> list[int]:
        return list(range(len(self.transitions)))

    def contains(self, transition_id: int) -> bool:
        return 0 <= transition_id < len(self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)


class _CountingSequenceStore(InMemoryReplayStore):
    def __init__(self) -> None:
        super().__init__()
        self.available_ids_calls = 0

    def available_ids(self) -> list[int]:
        self.available_ids_calls += 1
        return super().available_ids()


def _store(*, episodes: int = 2, steps: int = 4, demos: int = 0) -> InMemoryReplayStore:
    store = InMemoryReplayStore()
    for episode in range(episodes):
        for step in range(steps):
            store.append(
                Transition(
                    observation=float(step),
                    action=0.0,
                    reward=1.0,
                    next_observation=float(step + 1),
                    terminated=step == steps - 1,
                    truncated=False,
                    episode_id=f"episode-{episode}",
                    step=step,
                    info={"is_demo": episode * steps + step < demos},
                )
            )
    return store


def _paced_transition(spec: _EpisodeSpec, step: int) -> Transition:
    return Transition(
        observation=float(step),
        action=0.0,
        reward=1.0,
        next_observation=float(step + 1),
        terminated=step == spec.steps - 1,
        truncated=False,
        episode_id=spec.episode_id,
        step=step,
        info=_paced_info(spec),
    )


def _paced_info(spec: _EpisodeSpec) -> dict[str, float | bool]:
    info: dict[str, float | bool] = {"sampling/projected_lap_time_s": spec.pace_s}
    if spec.origin is not _ReplayOrigin.UNMARKED:
        info["is_demo"] = spec.origin is _ReplayOrigin.DEMONSTRATION
    return info


def _paced_store(specs: Sequence[_EpisodeSpec], capacity: int = 64) -> InMemoryReplayStore:
    store = InMemoryReplayStore(capacity=capacity)
    for spec in specs:
        _append_paced_episode(store, spec)
    return store


def _append_paced_episode(store: InMemoryReplayStore, spec: _EpisodeSpec) -> None:
    for step in range(spec.steps):
        store.append(_paced_transition(spec, step))


def _fallback_pace_store(size: int = 100) -> _BasicReplayStore:
    store = _BasicReplayStore()
    for transition_id in range(size):
        pace = 40.0 if transition_id < size // 2 else 60.0
        store.append(
            Transition(
                observation=float(transition_id),
                action=0,
                reward=0.0,
                next_observation=float(transition_id + 1),
                terminated=True,
                truncated=False,
                info={"sampling/projected_lap_time_s": pace},
            )
        )
    return store


def _basic_n_step_store() -> _BasicReplayStore:
    store = _BasicReplayStore()
    for step in range(5):
        store.append(
            Transition(
                observation=float(step),
                action=step % 3,
                reward=float(step + 1),
                next_observation=float(step + 1),
                terminated=step == 4,
                truncated=False,
                episode_id="episode-0",
                step=step,
            )
        )
    return store


def _behavior_store() -> InMemoryReplayStore:
    store = InMemoryReplayStore()
    for step in range(3):
        store.append(_behavior_transition(step))
    return store


def _behavior_transition(step: int) -> Transition:
    return Transition(
        observation=float(step),
        action=0.0,
        reward=1.0,
        next_observation=float(step + 1),
        terminated=step == 2,
        truncated=False,
        episode_id="episode-0",
        step=step,
        info={
            "_trackmaniarl_behavior_log_probability": -float(step),
            "_trackmaniarl_behavior_value": float(step) + 0.5,
            "_trackmaniarl_behavior_latent_action": [float(step), -float(step)],
        },
    )
