"""Deterministic contract tests for interchangeable replay samplers."""

from __future__ import annotations

import pytest
import torch

import trackmaniarl.core.replay.prioritized_index as prioritized_index
from tests.unit.core._replay_sampler_support import (
    _append_paced_episode,
    _EpisodeSpec,
    _fallback_pace_store,
    _paced_store,
    _paced_transition,
    _ReplayOrigin,
    _store,
)
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch
from trackmaniarl.core.replay import (
    InMemoryReplayStore,
    OnPolicySequenceSampler,
    PrioritizedSampler,
)


def _assert_final_n_step_targets(store: InMemoryReplayStore, batch: TrainingBatch) -> None:
    priority_ids = batch.metadata["priority_transition_ids"]
    for row, transition_id in enumerate(priority_ids):
        horizon = store.n_step_ids(transition_id, 2)
        expected_discount = 0.0 if horizon[-1] == 5 else 0.25
        histories = store.history_ids(transition_id, 3)
        expected_history = (
            [item.observation for item in store.get(histories)]
            + [item.next_observation for item in store.get(horizon)]
        )[-3:]
        assert float(batch.bootstrap_discounts[row, -1]) == pytest.approx(expected_discount)
        assert batch.next_observations[row].tolist() == pytest.approx(expected_history)


def _assert_expert_sequences(store: InMemoryReplayStore, batch: TrainingBatch) -> None:
    priority_ids = batch.metadata["priority_transition_ids"]
    for row, expert in enumerate(batch.metadata["expert_demo_flags"]):
        if expert:
            history = batch.transition_ids[row * 3 : (row + 1) * 3]
            assert all(store.demo_flags(list(history)))
            assert store.sampling_pace_s(priority_ids[row]) <= 37.5


def _expert_sampler(fraction: float) -> PrioritizedSampler:
    return PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=fraction,
        seed=7,
    )


def _annealed_expert_sampler() -> PrioritizedSampler:
    return PrioritizedSampler(
        IdentityFeaturePipeline(),
        expert_demo_time_s=37.5,
        expert_fraction=0.5,
        expert_fraction_final=0.0,
        expert_fraction_anneal_transitions=16,
        seed=7,
    )


def _scheduled_batch(
    sampler: PrioritizedSampler, store: InMemoryReplayStore, transition_count: int
) -> TrainingBatch:
    request = BatchRequest(batch_size=8, sequence_length=3, transition_count=transition_count)
    return sampler.sample(store, request)


def _mixed_demo_store() -> InMemoryReplayStore:
    specs = (
        _EpisodeSpec("episode-0", 6, 36.0, _ReplayOrigin.DEMONSTRATION),
        _EpisodeSpec("episode-1", 6, 37.4, _ReplayOrigin.DEMONSTRATION),
        _EpisodeSpec("episode-2", 6, 42.0, _ReplayOrigin.DEMONSTRATION),
        _EpisodeSpec("episode-3", 6, 39.0, _ReplayOrigin.ONLINE),
    )
    return _paced_store(specs)


def _expert_only_store() -> InMemoryReplayStore:
    specs = (
        _EpisodeSpec("expert-0", 6, 36.5, _ReplayOrigin.DEMONSTRATION),
        _EpisodeSpec("expert-1", 6, 36.5, _ReplayOrigin.DEMONSTRATION),
    )
    return _paced_store(specs)


def _interleaved_store() -> InMemoryReplayStore:
    first = _EpisodeSpec("episode-0", 32, 40.0)
    second = _EpisodeSpec("episode-1", 32, 40.0)
    store = InMemoryReplayStore(capacity=64)
    for step in range(32):
        store.append(_paced_transition(first, step))
        store.append(_paced_transition(second, step))
    return store


def _capture_candidate_refreshes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    refreshed: list[int] = []
    original_refresh = prioritized_index._refresh_candidate

    def count_refresh(request: prioritized_index._SynchronizationRequest, candidate: int) -> None:
        refreshed.append(candidate)
        original_refresh(request, candidate)

    monkeypatch.setattr(prioritized_index, "_refresh_candidate", count_refresh)
    return refreshed


def _assert_initial_interleaved_histories(store: InMemoryReplayStore) -> None:
    assert store.history_ids(0, 4) == [0, 0, 0, 0]
    assert store.history_ids(4, 4) == [0, 0, 2, 4]
    assert store.next_history_observations(0, 2, 4) == [0.0, 0.0, 1.0, 2.0]


def _assert_full_histories(store: InMemoryReplayStore, batch: TrainingBatch) -> None:
    assert all(
        len(set(store.history_ids(transition_id, 4))) == 4
        for transition_id in batch.metadata["priority_transition_ids"]
    )


def test_on_policy_sampler_uses_latest_fixed_rollout() -> None:
    store = _store(episodes=2, steps=3)

    batch = OnPolicySequenceSampler(IdentityFeaturePipeline()).sample(
        store, BatchRequest(batch_size=1, sequence_length=3)
    )

    assert batch.transition_ids == [3, 4, 5]
    assert batch.metadata["sampling"] == "on_policy"
    assert batch.rewards.shape == (1, 3)


def test_on_policy_sampler_allows_episode_boundary_inside_rollout() -> None:
    store = _store(episodes=2, steps=3)

    batch = OnPolicySequenceSampler(IdentityFeaturePipeline()).sample(
        store, BatchRequest(batch_size=1, sequence_length=4)
    )

    assert batch.transition_ids == [2, 3, 4, 5]
    assert torch.equal(batch.terminated, torch.tensor([[True, False, False, True]]))


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
    assert batch.metadata["gamma"] == 0.5
    assert batch.metadata["n_step"] == 2
    torch.testing.assert_close(
        batch.bootstrap_discounts[:, :-1],
        torch.full((2, 2), 0.5),
    )
    _assert_final_n_step_targets(store, batch)


def test_prioritized_sequences_only_activate_actor_equivalent_full_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _interleaved_store()
    _assert_initial_interleaved_histories(store)
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=0)
    batch = sampler.sample(store, BatchRequest(batch_size=2, sequence_length=4))
    assert sampler._active_count == 58
    _assert_full_histories(store, batch)
    refreshed = _capture_candidate_refreshes(monkeypatch)
    _append_paced_episode(store, _EpisodeSpec("episode-2", 1, 40.0))
    batch = sampler.sample(store, BatchRequest(batch_size=1, sequence_length=4))
    assert store.history_ids(6, 4) == [2, 2, 4, 6]
    assert store.history_ids(7, 4) == [1, 3, 5, 7]
    assert sampler._active_count == 57
    _assert_full_histories(store, batch)
    assert len(refreshed) <= 4


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
    specs = (_EpisodeSpec("episode-0", 2, 42.0), _EpisodeSpec("episode-1", 2, 50.0))
    store = _paced_store(specs, capacity=8)
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


def test_prioritized_fallback_applies_elite_pace_boost() -> None:
    store = _fallback_pace_store()
    sampler = PrioritizedSampler(
        IdentityFeaturePipeline(),
        alpha=1.0,
        elite_time_s=45.0,
        elite_priority_boost=100.0,
        seed=3,
    )

    batch = sampler.sample(store, BatchRequest(batch_size=100))

    assert batch.metadata["replay/elite_active_fraction"] == 0.5
    assert batch.metadata["replay/elite_sample_fraction"] > 0.95


def test_prioritized_sequence_sampler_enforces_exact_expert_demo_fraction() -> None:
    store = _mixed_demo_store()
    sampler = _expert_sampler(0.25)

    batch = sampler.sample(
        store,
        BatchRequest(batch_size=8, sequence_length=3, n_step=1),
    )

    assert sum(batch.metadata["expert_demo_flags"]) == 2
    assert batch.metadata["replay/expert_demo_sample_fraction"] == 0.25
    assert batch.metadata["replay/demo_sample_fraction"] >= 0.25
    assert batch.importance_weights is not None
    assert max(batch.importance_weights) == pytest.approx(1.0)
    _assert_expert_sequences(store, batch)


def test_prioritized_sampler_linearly_removes_the_forced_expert_quota() -> None:
    store = _mixed_demo_store()
    sampler = _annealed_expert_sampler()
    initial = _scheduled_batch(sampler, store, 0)
    midpoint = _scheduled_batch(sampler, store, 8)
    final = _scheduled_batch(sampler, store, 16)

    assert initial.metadata["replay/expert_demo_target_fraction"] == 0.5
    assert initial.metadata["replay/expert_demo_sample_fraction"] == 0.5
    assert midpoint.metadata["replay/expert_demo_target_fraction"] == 0.25
    assert midpoint.metadata["replay/expert_demo_sample_fraction"] == 0.25
    assert final.metadata["replay/expert_demo_target_fraction"] == 0.0


def test_prioritized_sampler_rejects_an_incomplete_expert_fraction_schedule() -> None:
    invalid_options = (
        {"expert_fraction_final": 0.0},
        {"expert_fraction_anneal_transitions": 10},
        {"expert_fraction_final": 1.1, "expert_fraction_anneal_transitions": 10},
        {"expert_fraction_final": 0.0, "expert_fraction_anneal_transitions": 0},
    )
    for options in invalid_options:
        with pytest.raises(ValueError, match="invalid prioritized replay parameters"):
            PrioritizedSampler(
                IdentityFeaturePipeline(),
                expert_demo_time_s=37.5,
                expert_fraction=0.5,
                **options,
            )


def test_prioritized_sampler_bootstraps_from_an_expert_only_replay() -> None:
    store = _expert_only_store()
    sampler = _expert_sampler(0.5)
    request = BatchRequest(batch_size=8, sequence_length=3, n_step=1)

    bootstrap = sampler.sample(store, request)

    assert all(bootstrap.metadata["expert_demo_flags"])
    assert bootstrap.metadata["replay/expert_demo_sample_fraction"] == 1.0
    online = _EpisodeSpec("online", 6, 40.0, _ReplayOrigin.ONLINE)
    _append_paced_episode(store, online)

    mixed = sampler.sample(store, request)

    assert sum(mixed.metadata["expert_demo_flags"]) == 4
    assert mixed.metadata["replay/expert_demo_sample_fraction"] == 0.5


def test_prioritized_expert_bootstrap_still_rejects_an_undersized_replay() -> None:
    spec = _EpisodeSpec("expert", 1, 36.5, _ReplayOrigin.DEMONSTRATION)
    store = _paced_store((spec,), capacity=8)
    sampler = _expert_sampler(0.5)

    with pytest.raises(RuntimeError, match="Need 2 transitions"):
        sampler.sample(store, BatchRequest(batch_size=2))


def test_prioritized_sampler_full_rebuild_resets_expert_count() -> None:
    specs = (
        _EpisodeSpec("expert", 6, 36.0, _ReplayOrigin.DEMONSTRATION),
        _EpisodeSpec("online", 6, 40.0, _ReplayOrigin.ONLINE),
    )
    store = _paced_store(specs)
    sampler = _expert_sampler(0.25)
    request = BatchRequest(batch_size=4, sequence_length=3, n_step=1)
    sampler.sample(store, request)
    expert_count = sampler._expert_active_count
    sampler._replay_revision = None

    sampler.sample(store, request)

    assert sampler._expert_active_count == expert_count
