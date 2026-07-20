"""Regression tests for the current replay and priority-update contracts."""

from __future__ import annotations

import pytest
import torch
from tmrl.core.builtins import IdentityFeaturePipeline
from tmrl.core.data import BatchRequest, PriorityUpdate, Transition
from tmrl.core.replay import InMemoryReplayStore, PrioritizedSampler, _n_step_transition


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


def test_n_step_return_stops_at_a_terminal_transition() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0))
    store.append(_transition(1, terminated=True))
    store.append(_transition(2))
    transition, discount = _n_step_transition(
        0,
        dict(zip(store.available_ids(), store.get(store.available_ids()), strict=True)),
        BatchRequest(batch_size=1, n_step=3, gamma=0.9),
    )
    assert transition.reward == 2.8
    assert discount == 0.0


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


def test_prioritized_sampler_accepts_learner_priority_feedback() -> None:
    store = InMemoryReplayStore()
    for step in range(8):
        store.append(_transition(step, terminated=step == 7))
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=0)
    batch = sampler.sample(store, BatchRequest(batch_size=4, n_step=1))
    sampler.update_priorities(PriorityUpdate(batch.transition_ids, [2.0] * len(batch.indices)))
    replayed = sampler.sample(store, BatchRequest(batch_size=4, n_step=1))
    assert torch.isfinite(replayed.importance_weights).all()


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
