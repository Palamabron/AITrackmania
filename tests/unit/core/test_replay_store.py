"""Regression tests for the current replay and priority-update contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import pytest

from trackmaniarl.core.data import BatchRequest, PriorityUpdate, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, _n_step_transition
from trackmaniarl.core.replay.n_step import _NStepInput


class _Boundary(Enum):
    NONE = "none"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class _ArrayTransitionSpec:
    observation: tuple[float, float]
    action: int
    reward: float
    next_observation: tuple[float, float]
    boundary: _Boundary
    tag: str
    episode: str
    step: int


@dataclass(frozen=True, slots=True)
class _RandomStep:
    rng: np.random.Generator
    episode: str
    step: int
    last_step: int


def _transition(step: int, boundary: _Boundary = _Boundary.NONE) -> Transition:
    return Transition(
        observation=float(step),
        action=step,
        reward=float(step + 1),
        next_observation=float(step + 1),
        terminated=boundary is _Boundary.TERMINATED,
        truncated=boundary is _Boundary.TRUNCATED,
        episode_id="episode",
        step=step,
    )


def _reference_n_step(
    store: InMemoryReplayStore, transition_id: int, request: BatchRequest
) -> tuple[Transition, float]:
    horizon = store.n_step_ids(transition_id, request.n_step)
    available = dict(zip(horizon, store.get(horizon), strict=True))
    return _n_step_transition(_NStepInput(transition_id, available, request, horizon))


def _assert_transition_equal(actual: Transition, expected: Transition) -> None:
    np.testing.assert_array_equal(actual.observation, expected.observation)
    np.testing.assert_array_equal(actual.action, expected.action)
    assert actual.reward == pytest.approx(expected.reward)
    np.testing.assert_array_equal(actual.next_observation, expected.next_observation)
    assert actual.terminated is expected.terminated
    assert actual.truncated is expected.truncated
    assert actual.info == expected.info
    assert actual.episode_id == expected.episode_id
    assert actual.step == expected.step


def _anonymous_transition(value: float, boundary: _Boundary) -> Transition:
    return Transition(
        observation=value,
        action=0,
        reward=0.0,
        next_observation=value + 1.0,
        terminated=boundary is _Boundary.TERMINATED,
        truncated=boundary is _Boundary.TRUNCATED,
    )


def _anonymous_boundary_store(boundary: _Boundary) -> InMemoryReplayStore:
    store = InMemoryReplayStore()
    store.append(_anonymous_transition(0.0, boundary))
    store.append(_anonymous_transition(1.0, _Boundary.NONE))
    return store


def _array_transition(spec: _ArrayTransitionSpec) -> Transition:
    return Transition(
        observation=np.asarray(spec.observation, dtype=np.float32),
        action=np.asarray([spec.action], dtype=np.int64),
        reward=spec.reward,
        next_observation=np.asarray(spec.next_observation, dtype=np.float32),
        terminated=spec.boundary is _Boundary.TERMINATED,
        truncated=spec.boundary is _Boundary.TRUNCATED,
        info={"tag": spec.tag, "latent": [float(spec.action), float(spec.action + 1)]},
        episode_id=spec.episode,
        step=spec.step,
    )


def _interleaved_transitions() -> tuple[Transition, ...]:
    specs = (
        _ArrayTransitionSpec((0.0, 0.0), 0, 1.0, (0.0, 1.0), _Boundary.NONE, "a-0", "a", 0),
        _ArrayTransitionSpec((1.0, 0.0), 10, 10.0, (1.0, 1.0), _Boundary.NONE, "b-0", "b", 0),
        _ArrayTransitionSpec((0.0, 1.0), 1, 2.0, (99.0, 99.0), _Boundary.NONE, "a-1", "a", 1),
        _ArrayTransitionSpec(
            (1.0, 1.0), 11, 20.0, (88.0, 88.0), _Boundary.TRUNCATED, "b-1", "b", 1
        ),
        _ArrayTransitionSpec((0.0, 2.0), 2, 3.0, (77.0, 77.0), _Boundary.TERMINATED, "a-2", "a", 2),
    )
    return tuple(_array_transition(spec) for spec in specs)


def _interleaved_store() -> InMemoryReplayStore:
    store = InMemoryReplayStore()
    for transition in _interleaved_transitions():
        store.append(transition)
    return store


def _assert_n_step_parity(store: InMemoryReplayStore, request: BatchRequest) -> None:
    transition_ids = store.available_ids()
    materialized, discounts = store.materialize_n_step(transition_ids, request)
    for transition_id, actual, discount in zip(
        transition_ids, materialized, discounts, strict=True
    ):
        expected, expected_discount = _reference_n_step(store, transition_id, request)
        _assert_transition_equal(actual, expected)
        assert discount == pytest.approx(expected_discount)


def _ring_transition(step: int) -> Transition:
    natural_next = np.asarray([float(step + 1)], dtype=np.float32)
    next_observation = np.asarray([1234.0], dtype=np.float32) if step == 8 else natural_next
    return Transition(
        observation=np.asarray([float(step)], dtype=np.float32),
        action=np.asarray([step % 3], dtype=np.int64),
        reward=float(step - 3),
        next_observation=next_observation,
        terminated=step == 9,
        truncated=False,
        info={"source_step": step},
        episode_id="ring",
        step=step,
    )


def _ring_store() -> InMemoryReplayStore:
    store = InMemoryReplayStore(capacity=7)
    for step in range(10):
        store.append(_ring_transition(step))
    return store


def _assert_ring_parity(store: InMemoryReplayStore) -> None:
    for n_step in (1, 3, 7):
        request = BatchRequest(batch_size=len(store), n_step=n_step, gamma=0.73)
        _assert_n_step_parity(store, request)


def _assert_ring_override_semantics(store: InMemoryReplayStore) -> None:
    materialized, _ = store.materialize_n_step(
        [7, 3, 7], BatchRequest(batch_size=3, n_step=2, gamma=0.73)
    )
    np.testing.assert_array_equal(materialized[0].next_observation, [1234.0])
    assert materialized[0].info == materialized[2].info == {"source_step": 7}
    assert materialized[1].info == {"source_step": 3}


def _pytree_transition(sign: int, episode: str) -> Transition:
    return Transition(
        observation={"array": np.asarray([sign, 2 * sign]), "nested": [3 * sign]},
        action=(np.asarray([4 * sign], dtype=np.int64), 5 * sign),
        reward=float(6 * sign),
        next_observation={"array": np.asarray([7 * sign, 8 * sign]), "nested": [9 * sign]},
        terminated=True,
        truncated=False,
        info={"tag": episode},
        episode_id=episode,
        step=0,
    )


def _assert_first_pytree_owned(transition: Transition) -> None:
    np.testing.assert_array_equal(transition.observation["array"], [1.0, 2.0])
    assert transition.observation["nested"] == [3.0]
    np.testing.assert_array_equal(transition.action[0], [4])
    assert transition.action[1] == 5
    np.testing.assert_array_equal(transition.next_observation["array"], [7.0, 8.0])
    assert transition.next_observation["nested"] == [9.0]


def _random_observations(step: _RandomStep) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    episode_number = float(int(step.episode.split("-")[-1]))
    observation = np.asarray([episode_number, step.step], dtype=np.float32)
    natural_next = np.asarray([episode_number, step.step + 1], dtype=np.float32)
    if step.rng.random() < 0.2:
        return observation, natural_next + 1000.0
    return observation, natural_next


def _random_boundary(step: _RandomStep) -> _Boundary:
    if step.step != step.last_step:
        return _Boundary.NONE
    if bool(step.rng.integers(0, 2)):
        return _Boundary.TRUNCATED
    return _Boundary.TERMINATED


def _random_transition(step: _RandomStep) -> Transition:
    boundary = _random_boundary(step)
    observation, next_observation = _random_observations(step)
    return Transition(
        observation=observation,
        action=np.asarray([int(step.rng.integers(0, 78))], dtype=np.int64),
        reward=float(step.rng.normal()),
        next_observation=next_observation,
        terminated=boundary is _Boundary.TERMINATED,
        truncated=boundary is _Boundary.TRUNCATED,
        info={"episode": step.episode, "source_step": step.step},
        episode_id=step.episode,
        step=step.step,
    )


def _randomized_store() -> InMemoryReplayStore:
    rng = np.random.default_rng(8241)
    lengths = {f"episode-{index}": int(rng.integers(3, 10)) for index in range(12)}
    next_steps = dict.fromkeys(lengths, 0)
    store = InMemoryReplayStore(capacity=37)
    while pending := [episode for episode in lengths if next_steps[episode] < lengths[episode]]:
        episode = str(rng.choice(pending))
        step = next_steps[episode]
        store.append(_random_transition(_RandomStep(rng, episode, step, lengths[episode] - 1)))
        next_steps[episode] += 1
    return store


def _mutable_observation() -> dict[str, Any]:
    return {
        "array": np.asarray([1.0, 2.0], dtype=np.float32),
        "nested": [np.asarray([3.0], dtype=np.float32)],
    }


def _transition_with_next(next_observation: dict[str, Any]) -> Transition:
    return Transition(
        observation=np.zeros(2, dtype=np.float32),
        action=0,
        reward=0.0,
        next_observation=next_observation,
        terminated=False,
        truncated=False,
        episode_id="owned",
        step=0,
    )


def _mutate_observation(observation: dict[str, Any]) -> None:
    observation["array"][:] = 99.0
    observation["nested"][0][:] = 99.0


def test_priority_updates_require_finite_scalar_values() -> None:
    with pytest.raises(TypeError, match="scalar real numbers"):
        PriorityUpdate([0], cast(Any, [[1.0]]))
    for priority in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="priorities must be finite"):
            PriorityUpdate([0], [priority])


def test_truncation_stops_reward_accumulation_but_keeps_bootstrap() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0))
    store.append(_transition(1, _Boundary.TRUNCATED))
    transition, discount = _reference_n_step(
        store, 0, BatchRequest(batch_size=1, n_step=3, gamma=0.5)
    )
    assert transition.reward == 2.0
    assert discount == 0.25


def test_anonymous_history_never_crosses_episode_boundary() -> None:
    for boundary in (_Boundary.TERMINATED, _Boundary.TRUNCATED):
        store = _anonymous_boundary_store(boundary)
        assert store.history_ids(1, 2) == [1, 1]


def test_history_without_steps_never_crosses_named_episodes() -> None:
    store = InMemoryReplayStore()
    for value, episode in enumerate(("a", "b")):
        store.append(
            Transition(
                observation=float(value),
                action=0,
                reward=0.0,
                next_observation=float(value),
                terminated=False,
                truncated=False,
                episode_id=episode,
            )
        )

    assert store.history_ids(1, 2) == [1, 1]


def test_columnar_n_step_matches_reference_for_interleaved_episode_boundaries() -> None:
    store = _interleaved_store()
    request = BatchRequest(batch_size=len(store), n_step=5, gamma=0.5)

    materialized, discounts = store.materialize_n_step(store.available_ids(), request)

    _assert_n_step_parity(store, request)
    assert discounts[0] == 0.0
    assert discounts[1] == pytest.approx(0.25)
    np.testing.assert_array_equal(materialized[1].next_observation, [88.0, 88.0])
    assert materialized[0].info == {"tag": "a-0", "latent": [0.0, 1.0]}


def test_columnar_n_step_preserves_ring_and_next_override_semantics() -> None:
    store = _ring_store()

    _assert_ring_parity(store)
    with pytest.raises(KeyError, match="no longer available"):
        store.materialize_n_step([0], BatchRequest(batch_size=1, n_step=1))
    _assert_ring_override_semantics(store)


def test_columnar_n_step_owns_endpoint_pytrees_after_unlock() -> None:
    store = InMemoryReplayStore(capacity=1)
    store.append(_pytree_transition(1, "first"))
    materialized, _ = store.materialize_n_step(
        [0], BatchRequest(batch_size=1, n_step=1, gamma=0.9994)
    )

    store.append(_pytree_transition(-1, "second"))

    _assert_first_pytree_owned(materialized[0])


def test_columnar_n_step_randomized_parity_after_interleaved_ring_writes() -> None:
    store = _randomized_store()
    transition_ids = store.available_ids()
    for n_step, gamma in ((1, 0.0), (2, 0.5), (5, 0.9994), (11, 1.0)):
        request = BatchRequest(batch_size=len(transition_ids), n_step=n_step, gamma=gamma)
        _assert_n_step_parity(store, request)


def test_duplicate_episode_step_does_not_mutate_replay() -> None:
    store = InMemoryReplayStore(capacity=2)
    store.append(_transition(0))

    with pytest.raises(ValueError, match="duplicate replay episode step"):
        store.append(_transition(0))

    assert store.available_ids() == [0]
    assert store.get([0])[0].reward == 1.0


def test_replay_snapshot_is_immutable_after_ring_overwrite() -> None:
    store = InMemoryReplayStore(capacity=2)
    store.append(_transition(0))
    store.append(_transition(1))

    state = store.state_dict()
    store.append(_transition(2))
    restored = InMemoryReplayStore(capacity=2)
    restored.load_state_dict(state)

    assert [item.reward for item in restored.get(restored.available_ids())] == [1.0, 2.0]


def test_replay_takes_ownership_of_mutable_next_observation() -> None:
    store = InMemoryReplayStore(capacity=2)
    next_observation = _mutable_observation()
    store.append(_transition_with_next(next_observation))

    _mutate_observation(next_observation)
    restored = store.get([0])[0].next_observation

    np.testing.assert_array_equal(restored["array"], [1.0, 2.0])
    np.testing.assert_array_equal(restored["nested"][0], [3.0])
