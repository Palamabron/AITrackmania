"""N-step replay must preserve episode boundaries and bootstrap semantics."""

from __future__ import annotations

from enum import Enum

import pytest
import torch

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, Transition
from trackmaniarl.core.replay import (
    InMemoryReplayStore,
    PrioritizedSampler,
    UniformSampler,
    _n_step_transition,
)
from trackmaniarl.core.replay.n_step import _NStepInput


class _Boundary(Enum):
    NONE = "none"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"


def _transition(step: int, boundary: _Boundary = _Boundary.NONE) -> Transition:
    return Transition(
        observation=float(step),
        action=float(step),
        reward=float(step + 1),
        next_observation=float(step + 1),
        terminated=boundary is _Boundary.TERMINATED,
        truncated=boundary is _Boundary.TRUNCATED,
        episode_id="episode",
        step=step,
    )


def test_n_step_return_stops_on_termination_and_does_not_bootstrap() -> None:
    store = InMemoryReplayStore()
    for step in range(3):
        boundary = _Boundary.TERMINATED if step == 1 else _Boundary.NONE
        store.append(_transition(step, boundary))
    transition, discount = _n_step_transition(
        _NStepInput(
            0,
            dict(zip(store.available_ids(), store.get(store.available_ids()), strict=True)),
            BatchRequest(batch_size=1, n_step=3, gamma=0.5),
        )
    )
    assert transition.reward == 2.0
    assert discount == 0.0


def test_truncation_keeps_the_bootstrap_discount() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0, _Boundary.TRUNCATED))
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

    store.append(_transition(2, _Boundary.TERMINATED))
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


def _out_of_order_store() -> tuple[InMemoryReplayStore, int, int]:
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
    return store, earlier, later


def test_replay_keeps_next_observation_for_an_out_of_order_episode_link() -> None:
    store, earlier, later = _out_of_order_store()

    assert not store.contains(later)
    transition = store.get([earlier])[0]
    assert transition.next_observation == 1.0


def test_replay_checkpoint_restores_into_a_larger_capacity_store() -> None:
    store = InMemoryReplayStore(capacity=4)
    for step in range(6):
        boundary = _Boundary.TERMINATED if step == 5 else _Boundary.NONE
        store.append(_transition(step, boundary))
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


def test_replay_checkpoint_rejects_a_non_columnar_format() -> None:
    state = InMemoryReplayStore().state_dict()
    state["format"] = "object-map-v0"

    with pytest.raises(ValueError, match="unsupported replay checkpoint format"):
        InMemoryReplayStore().load_state_dict(state)


_REPLAY_FIELDS = (
    "format",
    "capacity",
    "size",
    "next_index",
    "episode_names",
    "next_overrides",
    "info",
    "observations",
    "actions",
    "rewards",
    "terminated",
    "truncated",
    "episode_codes",
    "steps",
    "previous_ids",
    "next_ids",
    "sampling_pace",
)


def test_replay_checkpoint_requires_current_schema_fields() -> None:
    store = InMemoryReplayStore()
    store.append(_transition(0))
    for field in _REPLAY_FIELDS:
        state = store.state_dict()
        state.pop(field)
        with pytest.raises(ValueError, match="missing required fields"):
            InMemoryReplayStore().load_state_dict(state)
