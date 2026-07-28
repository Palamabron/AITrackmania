"""N-step replay must preserve episode boundaries and bootstrap semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tmrl.core.builtins import IdentityFeaturePipeline
from tmrl.core.data import BatchRequest, PriorityUpdate, Transition
from tmrl.core.replay import (
    InMemoryReplayStore,
    PrioritizedSampler,
    UniformSampler,
    _n_step_transition,
)


def _transition(step: int, *, terminated: bool = False, truncated: bool = False) -> Transition:
    return Transition(
        observation=float(step),
        action=float(step),
        reward=float(step + 1),
        next_observation=float(step + 1),
        terminated=terminated,
        truncated=truncated,
        episode_id="episode",
        step=step,
    )


def test_n_step_return_stops_on_termination_and_does_not_bootstrap() -> None:
    store = InMemoryReplayStore()
    for step in range(3):
        store.append(_transition(step, terminated=step == 1))
    transition, discount = _n_step_transition(
        0,
        dict(zip(store.available_ids(), store.get(store.available_ids()), strict=True)),
        BatchRequest(batch_size=1, n_step=3, gamma=0.5),
    )
    assert transition.reward == 2.0
    assert discount == 0.0


def test_truncation_keeps_the_bootstrap_discount() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0, truncated=True))
    batch = UniformSampler(IdentityFeaturePipeline(), seed=0).sample(
        store, BatchRequest(batch_size=1, n_step=3, gamma=0.9)
    )
    assert bool(batch.truncated.item())
    assert torch.isclose(batch.bootstrap_discounts, torch.tensor([0.9]))


def test_n_step_sampler_waits_for_a_live_transition_horizon() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0))
    store.append(_transition(1))
    sampler = UniformSampler(IdentityFeaturePipeline(), seed=0)
    request = BatchRequest(batch_size=1, n_step=3, gamma=0.9)
    with pytest.raises(RuntimeError, match="complete n-step"):
        sampler.sample(store, request)

    store.append(_transition(2, terminated=True))
    batch = sampler.sample(store, request)
    assert len(batch.transition_ids) == 1
    assert torch.isfinite(batch.rewards).all()


def test_late_priority_update_for_evicted_id_is_ignored() -> None:
    store = InMemoryReplayStore(capacity=1)
    old_id = store.append(_transition(0))
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=0)
    sampler.sample(store, BatchRequest(batch_size=1))
    store.append(_transition(1))
    sampler.update_priorities(PriorityUpdate(transition_ids=[old_id], priorities=[999.0]))
    batch = sampler.sample(store, BatchRequest(batch_size=1))
    assert batch.transition_ids == [1]


def test_replay_keeps_next_observation_for_an_out_of_order_episode_link() -> None:
    store = InMemoryReplayStore(capacity=2)
    later = store.append(_transition(1))
    earlier = store.append(_transition(0))
    store.append(
        Transition(
            observation=10.0,
            action=0.0,
            reward=1.0,
            next_observation=11.0,
            terminated=False,
            truncated=False,
            episode_id="other-episode",
            step=0,
        )
    )

    assert not store.contains(later)
    transition = store.get([earlier])[0]
    assert transition.next_observation == 1.0


def test_replay_checkpoint_restores_into_a_larger_capacity_store() -> None:
    store = InMemoryReplayStore(capacity=4)
    for step in range(6):
        store.append(_transition(step, terminated=step == 5))
    state = store.state_dict()

    grown = InMemoryReplayStore(capacity=8)
    grown.load_state_dict(state)

    assert len(grown) == 4
    assert grown.available_ids() == [2, 3, 4, 5]
    assert [item.reward for item in grown.get([2, 3, 4, 5])] == [3.0, 4.0, 5.0, 6.0]
    assert grown.n_step_ids(2, 3) == [2, 3, 4]
    assert grown.append(_transition(6)) == 6
    assert len(grown) == 5

    with pytest.raises(ValueError, match="exceeds"):
        InMemoryReplayStore(capacity=2).load_state_dict(state)


def test_mark_episode_demo_promotes_only_the_named_episode() -> None:
    store = InMemoryReplayStore(capacity=8)
    for step in range(3):
        store.append(replace(_transition(step, terminated=step == 2), episode_id="lap-a"))
    for step in range(2):
        store.append(replace(_transition(step, terminated=step == 1), episode_id="lap-b"))

    assert store.mark_episode_demo("lap-a") == 3
    assert store.mark_episode_demo("lap-a") == 0
    assert store.mark_episode_demo("missing") == 0
    assert store.demo_flags([0, 1, 2, 3, 4]) == [True, True, True, False, False]
    assert store.demo_fraction() == pytest.approx(0.6)
