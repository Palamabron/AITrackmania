"""Deterministic contract tests for interchangeable 1.0 replay samplers."""

from __future__ import annotations

import threading

import pytest
import torch

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, Transition
from trackmaniarl.core.replay import (
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


def test_prioritized_sampler_uses_demo_flags_without_an_expert_threshold() -> None:
    store = _store(demos=4)
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=1)

    batch = sampler.sample(store, BatchRequest(batch_size=4))

    assert "expert_demo_flags" not in batch.metadata
    assert any(batch.metadata["demo_flags"])


def test_prioritized_prefetch_blocks_fifo_eviction_until_batch_is_materialized() -> None:
    class BlockingStore(InMemoryReplayStore):
        def __init__(self) -> None:
            super().__init__(capacity=1)
            self.block_sampling = False
            self.sampling_started = threading.Event()
            self.release_sampling = threading.Event()

        def n_step_ids(self, transition_id: int, n_step: int) -> list[int]:
            result = super().n_step_ids(transition_id, n_step)
            if self.block_sampling:
                self.sampling_started.set()
                assert self.release_sampling.wait(timeout=2.0)
            return result

    store = BlockingStore()
    store.append(
        Transition(
            observation=0.0,
            action=0.0,
            reward=1.0,
            next_observation=1.0,
            terminated=True,
            truncated=False,
            episode_id="episode-0",
            step=0,
        )
    )
    sampler = PrioritizedSampler(IdentityFeaturePipeline())
    sampler.sample(store, BatchRequest(batch_size=1))
    store.block_sampling = True
    sampled: list[object] = []
    appended = threading.Event()

    sample_thread = threading.Thread(
        target=lambda: sampled.append(sampler.sample(store, BatchRequest(batch_size=1)))
    )
    append_thread = threading.Thread(
        target=lambda: (
            store.append(
                Transition(
                    observation=1.0,
                    action=0.0,
                    reward=1.0,
                    next_observation=2.0,
                    terminated=True,
                    truncated=False,
                    episode_id="episode-1",
                    step=0,
                )
            ),
            appended.set(),
        )
    )
    sample_thread.start()
    assert store.sampling_started.wait(timeout=2.0)
    append_thread.start()

    assert not appended.wait(timeout=0.05)
    store.release_sampling.set()
    sample_thread.join(timeout=2.0)
    append_thread.join(timeout=2.0)

    assert sampler.thread_safe_prefetch
    assert len(sampled) == 1
    assert appended.is_set()


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


def test_prioritized_sampler_returns_episode_local_sequences_and_target_priorities() -> None:
    store = _store(episodes=2, steps=5)
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), alpha=0.4, seed=3)
    batch = sampler.sample(
        store,
        BatchRequest(batch_size=3, sequence_length=3, n_step=1),
    )

    assert batch.observations.shape == (3, 3)
    assert batch.importance_weights is not None
    assert len(batch.importance_weights) == 3
    priority_ids = batch.metadata["priority_transition_ids"]
    assert len(priority_ids) == 3
    assert all(
        list(batch.transition_ids[offset : offset + 3])[-1] == priority_ids[offset // 3]
        for offset in range(0, 9, 3)
    )
    sampler.update_priorities(PriorityUpdate(priority_ids, [2.0, 3.0, 4.0]))


def test_prioritized_sequence_builds_full_n_step_return_only_on_the_final_step() -> None:
    store = _store(episodes=1, steps=6)
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=2)

    batch = sampler.sample(
        store, BatchRequest(batch_size=2, sequence_length=3, n_step=2, gamma=0.5)
    )

    assert torch.equal(batch.rewards[:, :-1], torch.ones((2, 2)))
    priority_ids = batch.metadata["priority_transition_ids"]
    expected = [1.0 if transition_id == 5 else 1.5 for transition_id in priority_ids]
    assert batch.rewards[:, -1].tolist() == pytest.approx(expected)


def test_recurrent_history_left_pads_episode_start_like_the_actor() -> None:
    store = _store(episodes=1, steps=5)

    assert store.history_ids(0, 4) == [0, 0, 0, 0]
    assert store.history_ids(2, 4) == [0, 0, 1, 2]
    assert store.next_history_observations(0, 2, 4) == [0.0, 0.0, 1.0, 2.0]

    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=0)
    sampler.sample(store, BatchRequest(batch_size=5, sequence_length=4))
    assert sampler._active_count == 5


def test_prioritized_uniform_mix_preserves_low_priority_recovery_coverage() -> None:
    store = _store(episodes=4, steps=1)
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        alpha=1.0,
        uniform_mix=0.25,
        seed=4,
    )
    sampler.sample(store, BatchRequest(batch_size=4))
    sampler.update_priorities(PriorityUpdate([0, 1, 2, 3], [1.0e12, 1.0, 1.0, 1.0]))

    sampled = [
        transition_id
        for _ in range(500)
        for transition_id in sampler.sample(store, BatchRequest(batch_size=4)).transition_ids
    ]

    assert sum(transition_id != 0 for transition_id in sampled) > 250


def test_prioritized_sampler_boosts_elite_pace_without_retaining_sampling_metadata() -> None:
    store = InMemoryReplayStore(capacity=8)
    for step, pace in enumerate((42.0, 42.0, 50.0, 50.0)):
        store.append(
            Transition(
                observation=float(step),
                action=0.0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=step in {1, 3},
                truncated=False,
                episode_id=f"episode-{step // 2}",
                step=step % 2,
                info={"sampling/projected_lap_time_s": pace},
            )
        )
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        elite_time_s=44.0,
        elite_priority_boost=4.0,
        seed=1,
    )
    sampler.sample(store, BatchRequest(batch_size=1))

    assert sampler._tree is not None
    assert sampler._tree.leaves[0] == pytest.approx(4.0)
    assert sampler._tree.leaves[2] == pytest.approx(1.0)
    assert store.get([0])[0].info == {}


def test_prioritized_sequence_sampler_enforces_exact_expert_demo_fraction() -> None:
    store = InMemoryReplayStore(capacity=64)
    for episode, (pace, demo) in enumerate(
        ((36.0, True), (37.4, True), (42.0, True), (39.0, False))
    ):
        for step in range(6):
            store.append(
                Transition(
                    observation=float(step),
                    action=0.0,
                    reward=1.0,
                    next_observation=float(step + 1),
                    terminated=step == 5,
                    truncated=False,
                    episode_id=f"episode-{episode}",
                    step=step,
                    info={
                        "is_demo": demo,
                        "sampling/projected_lap_time_s": pace,
                    },
                )
            )
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=0.25,
        seed=7,
    )

    batch = sampler.sample(
        store,
        BatchRequest(batch_size=8, sequence_length=3, n_step=1),
    )

    assert sum(batch.metadata["expert_demo_flags"]) == 2
    assert batch.metadata["replay/expert_demo_sample_fraction"] == 0.25
    assert batch.metadata["replay/demo_sample_fraction"] >= 0.25
    assert batch.importance_weights is not None
    assert max(batch.importance_weights) == pytest.approx(1.0)
    priority_ids = batch.metadata["priority_transition_ids"]
    for row, expert in enumerate(batch.metadata["expert_demo_flags"]):
        if expert:
            history = batch.transition_ids[row * 3 : (row + 1) * 3]
            assert all(store.demo_flags(list(history)))
            assert store.sampling_pace_s(priority_ids[row]) <= 37.5


def test_prioritized_sampler_bootstraps_from_an_expert_only_replay() -> None:
    store = InMemoryReplayStore(capacity=64)
    for episode in range(2):
        for step in range(6):
            store.append(
                Transition(
                    observation=float(step),
                    action=0.0,
                    reward=1.0,
                    next_observation=float(step + 1),
                    terminated=step == 5,
                    truncated=False,
                    episode_id=f"expert-{episode}",
                    step=step,
                    info={
                        "is_demo": True,
                        "sampling/projected_lap_time_s": 36.5,
                    },
                )
            )
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=0.5,
        seed=7,
    )
    request = BatchRequest(batch_size=8, sequence_length=3, n_step=1)

    bootstrap = sampler.sample(store, request)

    assert all(bootstrap.metadata["expert_demo_flags"])
    assert bootstrap.metadata["replay/expert_demo_sample_fraction"] == 1.0
    for step in range(6):
        store.append(
            Transition(
                observation=float(step),
                action=0.0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=step == 5,
                truncated=False,
                episode_id="online",
                step=step,
                info={"sampling/projected_lap_time_s": 40.0},
            )
        )

    mixed = sampler.sample(store, request)

    assert sum(mixed.metadata["expert_demo_flags"]) == 4
    assert mixed.metadata["replay/expert_demo_sample_fraction"] == 0.5


def test_prioritized_expert_bootstrap_still_rejects_an_undersized_replay() -> None:
    store = InMemoryReplayStore(capacity=8)
    store.append(
        Transition(
            observation=0.0,
            action=0.0,
            reward=1.0,
            next_observation=1.0,
            terminated=True,
            truncated=False,
            episode_id="expert",
            step=0,
            info={
                "is_demo": True,
                "sampling/projected_lap_time_s": 36.5,
            },
        )
    )
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=0.5,
    )

    with pytest.raises(RuntimeError, match="Need 2 transitions"):
        sampler.sample(store, BatchRequest(batch_size=2))


def test_prioritized_sampler_full_rebuild_resets_expert_count() -> None:
    store = InMemoryReplayStore(capacity=64)
    for episode, demo in (("expert", True), ("online", False)):
        for step in range(6):
            store.append(
                Transition(
                    observation=float(step),
                    action=0.0,
                    reward=1.0,
                    next_observation=float(step + 1),
                    terminated=step == 5,
                    truncated=False,
                    episode_id=episode,
                    step=step,
                    info={
                        "is_demo": demo,
                        "sampling/projected_lap_time_s": 36.0 if demo else 40.0,
                    },
                )
            )
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=0.25,
        seed=7,
    )
    request = BatchRequest(batch_size=4, sequence_length=3, n_step=1)
    sampler.sample(store, request)
    expert_count = sampler._expert_active_count
    sampler._replay_revision = None

    sampler.sample(store, request)

    assert sampler._expert_active_count == expert_count


def test_prioritized_sampler_rejects_missing_expert_demo_pool() -> None:
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=0.25,
    )

    with pytest.raises(RuntimeError, match="expert_fraction"):
        sampler.sample(_store(episodes=2, steps=4), BatchRequest(batch_size=4))


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
