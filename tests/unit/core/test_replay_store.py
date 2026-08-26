"""Regression tests for the current replay and priority-update contracts."""

from __future__ import annotations

import numpy as np
import pytest

from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, _n_step_transition


def _transition(step: int, *, terminated: bool = False, truncated: bool = False) -> Transition:
    return Transition(
        observation=float(step),
        action=step,
        reward=float(step + 1),
        next_observation=float(step + 1),
        terminated=terminated,
        truncated=truncated,
        episode_id="episode",
        step=step,
    )


def _reference_n_step(
    store: InMemoryReplayStore, transition_id: int, request: BatchRequest
) -> tuple[Transition, float]:
    horizon = store.n_step_ids(transition_id, request.n_step)
    available = dict(zip(horizon, store.get(horizon), strict=True))
    return _n_step_transition(transition_id, available, request, horizon=horizon)


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


def test_truncation_stops_reward_accumulation_but_keeps_bootstrap() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0))
    store.append(_transition(1, truncated=True))
    transition, discount = _n_step_transition(
        0,
        dict(zip(store.available_ids(), store.get(store.available_ids()), strict=True)),
        BatchRequest(batch_size=1, n_step=3, gamma=0.5),
    )
    assert transition.reward == 2.0
    assert discount == 0.25


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_anonymous_history_never_crosses_episode_boundary(boundary: str) -> None:
    store = InMemoryReplayStore()
    store.append(
        Transition(
            observation=0.0,
            action=0,
            reward=0.0,
            next_observation=1.0,
            terminated=boundary == "terminated",
            truncated=boundary == "truncated",
        )
    )
    store.append(
        Transition(
            observation=1.0,
            action=0,
            reward=0.0,
            next_observation=2.0,
            terminated=False,
            truncated=False,
        )
    )

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
    store = InMemoryReplayStore()
    transitions = (
        Transition(
            observation=np.asarray([0.0, 0.0], dtype=np.float32),
            action=np.asarray([0], dtype=np.int64),
            reward=1.0,
            next_observation=np.asarray([0.0, 1.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            info={"tag": "a-0", "latent": [0.0, 1.0]},
            episode_id="a",
            step=0,
        ),
        Transition(
            observation=np.asarray([1.0, 0.0], dtype=np.float32),
            action=np.asarray([10], dtype=np.int64),
            reward=10.0,
            next_observation=np.asarray([1.0, 1.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            info={"tag": "b-0", "latent": [10.0, 11.0]},
            episode_id="b",
            step=0,
        ),
        Transition(
            observation=np.asarray([0.0, 1.0], dtype=np.float32),
            action=np.asarray([1], dtype=np.int64),
            reward=2.0,
            next_observation=np.asarray([99.0, 99.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            info={"tag": "a-1", "latent": [1.0, 2.0]},
            episode_id="a",
            step=1,
        ),
        Transition(
            observation=np.asarray([1.0, 1.0], dtype=np.float32),
            action=np.asarray([11], dtype=np.int64),
            reward=20.0,
            next_observation=np.asarray([88.0, 88.0], dtype=np.float32),
            terminated=False,
            truncated=True,
            info={"tag": "b-1", "latent": [11.0, 12.0]},
            episode_id="b",
            step=1,
        ),
        Transition(
            observation=np.asarray([0.0, 2.0], dtype=np.float32),
            action=np.asarray([2], dtype=np.int64),
            reward=3.0,
            next_observation=np.asarray([77.0, 77.0], dtype=np.float32),
            terminated=True,
            truncated=False,
            info={"tag": "a-2", "latent": [2.0, 3.0]},
            episode_id="a",
            step=2,
        ),
    )
    for transition in transitions:
        store.append(transition)
    request = BatchRequest(batch_size=len(transitions), n_step=5, gamma=0.5)

    materialized, discounts = store.materialize_n_step(store.available_ids(), request)

    for transition_id, actual, discount in zip(
        store.available_ids(), materialized, discounts, strict=True
    ):
        expected, expected_discount = _reference_n_step(store, transition_id, request)
        _assert_transition_equal(actual, expected)
        assert discount == pytest.approx(expected_discount)
    assert discounts[0] == 0.0
    assert discounts[1] == pytest.approx(0.25)
    np.testing.assert_array_equal(materialized[1].next_observation, [88.0, 88.0])
    assert materialized[0].info == {"tag": "a-0", "latent": [0.0, 1.0]}


def test_columnar_n_step_preserves_ring_and_next_override_semantics() -> None:
    store = InMemoryReplayStore(capacity=7)
    for step in range(10):
        natural_next = np.asarray([float(step + 1)], dtype=np.float32)
        next_observation = np.asarray([1234.0], dtype=np.float32) if step == 8 else natural_next
        store.append(
            Transition(
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
        )

    for n_step in (1, 3, 7):
        request = BatchRequest(batch_size=len(store), n_step=n_step, gamma=0.73)
        transition_ids = store.available_ids()
        materialized, discounts = store.materialize_n_step(transition_ids, request)
        for transition_id, actual, discount in zip(
            transition_ids, materialized, discounts, strict=True
        ):
            expected, expected_discount = _reference_n_step(store, transition_id, request)
            _assert_transition_equal(actual, expected)
            assert discount == pytest.approx(expected_discount)

    with pytest.raises(KeyError, match="no longer available"):
        store.materialize_n_step([0], BatchRequest(batch_size=1, n_step=1))

    materialized, _ = store.materialize_n_step(
        [7, 3, 7], BatchRequest(batch_size=3, n_step=2, gamma=0.73)
    )
    np.testing.assert_array_equal(materialized[0].next_observation, [1234.0])
    assert materialized[0].info == materialized[2].info == {"source_step": 7}
    assert materialized[1].info == {"source_step": 3}


def test_columnar_n_step_owns_endpoint_pytrees_after_unlock() -> None:
    store = InMemoryReplayStore(capacity=1)
    store.append(
        Transition(
            observation={"array": np.asarray([1.0, 2.0]), "nested": [3.0]},
            action=(np.asarray([4], dtype=np.int64), 5),
            reward=6.0,
            next_observation={"array": np.asarray([7.0, 8.0]), "nested": [9.0]},
            terminated=True,
            truncated=False,
            info={"tag": "first"},
            episode_id="first",
            step=0,
        )
    )
    materialized, _ = store.materialize_n_step(
        [0], BatchRequest(batch_size=1, n_step=1, gamma=0.9994)
    )

    store.append(
        Transition(
            observation={"array": np.asarray([-1.0, -2.0]), "nested": [-3.0]},
            action=(np.asarray([-4], dtype=np.int64), -5),
            reward=-6.0,
            next_observation={"array": np.asarray([-7.0, -8.0]), "nested": [-9.0]},
            terminated=True,
            truncated=False,
            episode_id="second",
            step=0,
        )
    )

    np.testing.assert_array_equal(materialized[0].observation["array"], [1.0, 2.0])
    assert materialized[0].observation["nested"] == [3.0]
    np.testing.assert_array_equal(materialized[0].action[0], [4])
    assert materialized[0].action[1] == 5
    np.testing.assert_array_equal(materialized[0].next_observation["array"], [7.0, 8.0])
    assert materialized[0].next_observation["nested"] == [9.0]


def test_columnar_n_step_randomized_parity_after_interleaved_ring_writes() -> None:
    rng = np.random.default_rng(8241)
    lengths = {f"episode-{index}": int(rng.integers(3, 10)) for index in range(12)}
    next_steps = dict.fromkeys(lengths, 0)
    store = InMemoryReplayStore(capacity=37)
    while pending := [episode for episode in lengths if next_steps[episode] < lengths[episode]]:
        episode = str(rng.choice(pending))
        step = next_steps[episode]
        final = step == lengths[episode] - 1
        truncated = final and bool(rng.integers(0, 2))
        observation = np.asarray([float(int(episode.split("-")[-1])), step], dtype=np.float32)
        natural_next = np.asarray([observation[0], step + 1], dtype=np.float32)
        next_observation = natural_next + 1000.0 if rng.random() < 0.2 else natural_next
        store.append(
            Transition(
                observation=observation,
                action=np.asarray([int(rng.integers(0, 78))], dtype=np.int64),
                reward=float(rng.normal()),
                next_observation=next_observation,
                terminated=final and not truncated,
                truncated=truncated,
                info={"episode": episode, "source_step": step},
                episode_id=episode,
                step=step,
            )
        )
        next_steps[episode] += 1

    transition_ids = store.available_ids()
    for n_step, gamma in ((1, 0.0), (2, 0.5), (5, 0.9994), (11, 1.0)):
        request = BatchRequest(batch_size=len(transition_ids), n_step=n_step, gamma=gamma)
        materialized, discounts = store.materialize_n_step(transition_ids, request)
        for transition_id, actual, discount in zip(
            transition_ids, materialized, discounts, strict=True
        ):
            expected, expected_discount = _reference_n_step(store, transition_id, request)
            _assert_transition_equal(actual, expected)
            assert discount == pytest.approx(expected_discount)


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
    next_observation = {
        "array": np.asarray([1.0, 2.0], dtype=np.float32),
        "nested": [np.asarray([3.0], dtype=np.float32)],
    }
    store.append(
        Transition(
            observation=np.zeros(2, dtype=np.float32),
            action=0,
            reward=0.0,
            next_observation=next_observation,
            terminated=False,
            truncated=False,
            episode_id="owned",
            step=0,
        )
    )

    next_observation["array"][:] = 99.0
    next_observation["nested"][0][:] = 99.0
    restored = store.get([0])[0].next_observation

    np.testing.assert_array_equal(restored["array"], [1.0, 2.0])
    np.testing.assert_array_equal(restored["nested"][0], [3.0])
