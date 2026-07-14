"""Deterministic contract tests for interchangeable 1.0 replay samplers."""

from __future__ import annotations

import pytest
import torch
from tmrl.core.builtins import IdentityFeaturePipeline
from tmrl.core.data import BatchRequest, PriorityUpdate, Transition
from tmrl.core.replay import (
    DemoMixSampler,
    InMemoryReplayStore,
    PrioritizedSampler,
    SequenceSampler,
    UniformSampler,
)


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


def test_prioritized_sampler_normalizes_weights_and_accepts_priority_feedback() -> None:
    store = _store()
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), alpha=0.6, beta=0.5, seed=1)
    initial = sampler.sample(store, BatchRequest(batch_size=4))
    sampler.update_priorities(
        PriorityUpdate(transition_ids=initial.transition_ids, priorities=[100.0] * 4)
    )
    batch = sampler.sample(store, BatchRequest(batch_size=4))
    assert len(batch.indices) == 4
    assert batch.weights is not None
    assert max(batch.weights) == 1.0
    assert min(batch.weights) > 0.0


def test_sequence_sampler_never_crosses_episode_or_terminal_boundary() -> None:
    store = _store(episodes=2, steps=3)
    batch = SequenceSampler(IdentityFeaturePipeline(), sequence_length=2, seed=1).sample(
        store, BatchRequest(batch_size=2)
    )
    assert batch.metadata["sequence_length"] == 2
    assert torch.equal(batch.masks, torch.ones((2, 2), dtype=torch.bool))
    assert batch.observations.shape == (2, 2)
    assert batch.actions.shape == (2, 2)
    assert batch.rewards.shape == (2, 2)
    windows = [batch.transition_ids[index : index + 2] for index in range(0, 4, 2)]
    assert all(window[0] in {0, 1, 3, 4} and window[1] == window[0] + 1 for window in windows)


def test_demo_mix_uses_ceil_minimum_and_floor_maximum() -> None:
    store = _store(episodes=2, steps=5, demos=2)
    sampler = DemoMixSampler(
        IdentityFeaturePipeline(), min_demo_fraction=0.4, max_demo_fraction=0.4, seed=1
    )
    batch = sampler.sample(store, BatchRequest(batch_size=5))
    assert batch.metadata["demo_fraction"] == 0.4


def test_demo_mix_rejects_unfulfillable_minimum_instead_of_silently_violating_policy() -> None:
    sampler = DemoMixSampler(
        IdentityFeaturePipeline(), min_demo_fraction=0.6, max_demo_fraction=0.8
    )
    with pytest.raises(RuntimeError, match="Need 3 demo"):
        sampler.sample(_store(episodes=1, steps=5, demos=2), BatchRequest(batch_size=5))


def test_standard_samplers_do_not_materialize_the_full_replay_per_update() -> None:
    class NoFullScanStore(InMemoryReplayStore):
        def available_ids(self) -> list[int]:
            raise AssertionError("standard replay sampling must use the incremental index")

    store = NoFullScanStore()
    for step in range(16):
        store.append(
            Transition(
                observation=float(step),
                action=0.0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=step == 15,
                truncated=False,
                episode_id="episode",
                step=step,
            )
        )
    request = BatchRequest(batch_size=4, n_step=3)
    assert (
        len(UniformSampler(IdentityFeaturePipeline(), seed=0).sample(store, request).indices) == 4
    )
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=0)
    first = sampler.sample(store, request)
    sampler.update_priorities(PriorityUpdate(first.indices, [2.0] * len(first.indices)))
    assert len(sampler.sample(store, request).indices) == 4
