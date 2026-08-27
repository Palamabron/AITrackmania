"""Deterministic contract tests for interchangeable replay samplers."""

from __future__ import annotations

import threading

import pytest
import torch

from tests.unit.core._replay_sampler_support import (
    _basic_n_step_store,
    _behavior_store,
    _CountingSequenceStore,
    _store,
)
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, Transition
from trackmaniarl.core.replay import (
    InMemoryReplayStore,
    PrioritizedSampler,
    SequenceSampler,
    UniformSampler,
)


class _BlockingStore(InMemoryReplayStore):
    def __init__(self) -> None:
        super().__init__(capacity=1)
        self.block_sampling = False
        self.sampling_started = threading.Event()
        self.release_sampling = threading.Event()

    def materialize_n_step(
        self, transition_ids: list[int], request: BatchRequest
    ) -> tuple[list[Transition], list[float]]:
        if self.block_sampling:
            self.sampling_started.set()
            assert self.release_sampling.wait(timeout=2.0)
        return super().materialize_n_step(transition_ids, request)


class _CountingMaterializationStore(InMemoryReplayStore):
    def __init__(self) -> None:
        super().__init__()
        self.materialize_calls = 0

    def materialize_n_step(
        self, transition_ids: list[int], request: BatchRequest
    ) -> tuple[list[Transition], list[float]]:
        self.materialize_calls += 1
        return super().materialize_n_step(transition_ids, request)


def _terminal_transition(value: float, episode_id: str) -> Transition:
    return Transition(
        observation=value,
        action=0.0,
        reward=1.0,
        next_observation=value + 1.0,
        terminated=True,
        truncated=False,
        episode_id=episode_id,
        step=0,
    )


def _sample_once(
    sampled: list[object], sampler: PrioritizedSampler, store: InMemoryReplayStore
) -> None:
    sampled.append(sampler.sample(store, BatchRequest(batch_size=1)))


def _append_and_signal(
    store: InMemoryReplayStore, transition: Transition, appended: threading.Event
) -> None:
    store.append(transition)
    appended.set()


def _start_blocked_operations(
    store: _BlockingStore, sampler: PrioritizedSampler
) -> tuple[threading.Thread, threading.Thread, list[object], threading.Event]:
    sampled: list[object] = []
    appended = threading.Event()
    sample_thread = threading.Thread(target=_sample_once, args=(sampled, sampler, store))
    append_thread = threading.Thread(
        target=_append_and_signal,
        args=(store, _terminal_transition(1.0, "episode-1"), appended),
    )
    sample_thread.start()
    assert store.sampling_started.wait(timeout=2.0)
    append_thread.start()
    return sample_thread, append_thread, sampled, appended


def _finish_blocked_operations(
    store: _BlockingStore, sample_thread: threading.Thread, append_thread: threading.Thread
) -> None:
    store.release_sampling.set()
    sample_thread.join(timeout=2.0)
    append_thread.join(timeout=2.0)


def _expected_sequence_target(window: list[int]) -> tuple[list[int], float, float, list[float]]:
    final_id = window[-1]
    horizon = list(range(final_id, min(final_id + 2, 5)))
    reward = sum(0.5**offset * (step + 1) for offset, step in enumerate(horizon))
    discount = 0.0 if horizon[-1] == 4 else 0.25
    history = ([float(step) for step in window] + [float(step + 1) for step in horizon])[-3:]
    return horizon, reward, discount, history


def _assert_sequence_target(batch: TrainingBatch, row: int) -> None:
    window = list(batch.transition_ids[row * 3 : (row + 1) * 3])
    _, reward, discount, history = _expected_sequence_target(window)
    assert batch.rewards[row, :2].tolist() == pytest.approx(
        [float(step + 1) for step in window[:2]]
    )
    assert float(batch.rewards[row, -1]) == pytest.approx(reward)
    assert batch.bootstrap_discounts[row, :2].tolist() == pytest.approx([0.5, 0.5])
    assert float(batch.bootstrap_discounts[row, -1]) == pytest.approx(discount)
    assert batch.next_observations[row].tolist() == pytest.approx(history)


def _copy_store(source: InMemoryReplayStore, target: InMemoryReplayStore) -> None:
    for transition in source.get(source.available_ids()):
        target.append(transition)


def test_prioritized_sampler_normalizes_weights_and_accepts_priority_feedback() -> None:
    store = _store()
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), alpha=0.6, beta=0.5, seed=1)
    initial = sampler.sample(store, BatchRequest(batch_size=4))
    sampler.update_priorities(
        PriorityUpdate(transition_ids=initial.transition_ids, priorities=[100.0] * 4)
    )
    batch = sampler.sample(store, BatchRequest(batch_size=4))
    assert len(batch.transition_ids) == 4
    assert batch.importance_weights is not None
    assert max(batch.importance_weights) == 1.0
    assert min(batch.importance_weights) > 0.0


def test_prioritized_sampler_uses_demo_flags_without_an_expert_threshold() -> None:
    store = _store(demos=4)
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=1)

    batch = sampler.sample(store, BatchRequest(batch_size=4))

    assert "expert_demo_flags" not in batch.metadata
    assert any(batch.metadata["demo_flags"])


def test_prioritized_sampler_state_round_trips_current_schema() -> None:
    store = _store()
    source = PrioritizedSampler(IdentityFeaturePipeline(), seed=4)
    source.sample(store, BatchRequest(batch_size=4))
    state = source.state_dict()
    expected = source.sample(store, BatchRequest(batch_size=4)).transition_ids
    restored = PrioritizedSampler(IdentityFeaturePipeline(), seed=999)

    restored.load_state_dict(state)

    assert restored.sample(store, BatchRequest(batch_size=4)).transition_ids == expected


def test_prioritized_sampler_rejects_a_non_array_checkpoint() -> None:
    state = PrioritizedSampler(IdentityFeaturePipeline()).state_dict()
    state["format"] = "mapping-per-v0"

    with pytest.raises(ValueError, match="unsupported prioritized replay checkpoint format"):
        PrioritizedSampler(IdentityFeaturePipeline()).load_state_dict(state)


def test_prioritized_sampler_requires_current_schema_fields() -> None:
    fields = ("format", "priorities", "slot_ids", "fallback_priorities", "maximum_priority", "rng")
    for field in fields:
        state = PrioritizedSampler(IdentityFeaturePipeline()).state_dict()
        state.pop(field)
        with pytest.raises(ValueError, match="missing required fields"):
            PrioritizedSampler(IdentityFeaturePipeline()).load_state_dict(state)


def test_prioritized_prefetch_blocks_fifo_eviction_until_batch_is_materialized() -> None:
    store = _BlockingStore()
    store.append(_terminal_transition(0.0, "episode-0"))
    sampler = PrioritizedSampler(IdentityFeaturePipeline())
    sampler.sample(store, BatchRequest(batch_size=1))
    store.block_sampling = True
    sample_thread, append_thread, sampled, appended = _start_blocked_operations(store, sampler)

    assert not appended.wait(timeout=0.05)
    _finish_blocked_operations(store, sample_thread, append_thread)

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


def test_basic_sequence_sampler_builds_only_the_final_n_step_target() -> None:
    store = _basic_n_step_store()

    batch = SequenceSampler(IdentityFeaturePipeline(), sequence_length=3, seed=4).sample(
        store,
        BatchRequest(batch_size=3, sequence_length=3, n_step=2, gamma=0.5),
    )

    assert batch.metadata["gamma"] == 0.5
    assert batch.metadata["n_step"] == 2
    assert batch.metadata["priority_transition_ids"] == tuple(
        batch.transition_ids[index] for index in range(2, 9, 3)
    )
    for row in range(3):
        _assert_sequence_target(batch, row)


def test_columnar_n_step_fast_path_is_limited_to_non_sequence_batches() -> None:
    source = _store(episodes=2, steps=6)
    store = _CountingMaterializationStore()
    _copy_store(source, store)

    UniformSampler(IdentityFeaturePipeline(), seed=0).sample(
        store, BatchRequest(batch_size=4, n_step=3)
    )
    assert store.materialize_calls == 1

    SequenceSampler(IdentityFeaturePipeline(), sequence_length=3, seed=0).sample(
        store, BatchRequest(batch_size=2, sequence_length=3, n_step=2)
    )
    assert store.materialize_calls == 1


def test_sequence_sampler_reuses_its_window_index_until_replay_changes() -> None:
    source = _store(episodes=1, steps=8)
    store = _CountingSequenceStore()
    _copy_store(source, store)
    sampler = SequenceSampler(IdentityFeaturePipeline(), sequence_length=3, seed=4)
    request = BatchRequest(batch_size=2, sequence_length=3)

    sampler.sample(store, request)
    sampler.sample(store, request)

    assert store.available_ids_calls == 1
    store.append(_terminal_transition(0.0, "episode-1"))
    sampler.sample(store, request)
    assert store.available_ids_calls == 2


def test_sequence_sampler_rng_resume_is_independent_of_derived_window_cache() -> None:
    store = _store(episodes=2, steps=8)
    sampler = SequenceSampler(IdentityFeaturePipeline(), sequence_length=3, seed=4)
    request = BatchRequest(batch_size=3, sequence_length=3)
    sampler.sample(store, request)
    state = sampler.state_dict()

    expected = sampler.sample(store, request).transition_ids
    restored = SequenceSampler(IdentityFeaturePipeline(), sequence_length=3, seed=999)
    restored.load_state_dict(state)

    assert restored.sample(store, request).transition_ids == expected


def test_sequence_sampler_preserves_ppo_behavior_statistics() -> None:
    store = _behavior_store()

    batch = SequenceSampler(IdentityFeaturePipeline(), sequence_length=3).sample(
        store, BatchRequest(batch_size=1, sequence_length=3)
    )

    assert torch.equal(
        batch.metadata["behavior_log_probabilities"], torch.tensor([[0.0, -1.0, -2.0]])
    )
    assert torch.equal(batch.metadata["behavior_values"], torch.tensor([[0.5, 1.5, 2.5]]))
    assert torch.equal(
        batch.metadata["behavior_latent_actions"],
        torch.tensor([[[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]]),
    )
