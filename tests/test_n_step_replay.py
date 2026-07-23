"""N-step replay must preserve episode boundaries and bootstrap semantics."""

from __future__ import annotations

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
