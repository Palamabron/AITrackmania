"""Episode-level pace relabeling contracts for online replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import trackmaniarl.core.replay.prioritized_index as prioritized_index
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, PrioritizedSampler


@dataclass(frozen=True, slots=True)
class _EpisodeSpec:
    episode_id: str
    steps: int
    pace_s: float | None = None


def _transition(spec: _EpisodeSpec, step: int) -> Transition:
    info = {} if spec.pace_s is None else {"sampling/projected_lap_time_s": spec.pace_s}
    return Transition(
        observation=float(step),
        action=step,
        reward=float(step + 1),
        next_observation=float(step + 1),
        terminated=step == spec.steps - 1,
        truncated=False,
        info=info,
        episode_id=spec.episode_id,
        step=step,
    )


def _append_episode(store: InMemoryReplayStore, spec: _EpisodeSpec) -> None:
    for step in range(spec.steps):
        store.append(_transition(spec, step))


def _two_episode_store(capacity: int = 8) -> InMemoryReplayStore:
    store = InMemoryReplayStore(capacity=capacity)
    _append_episode(store, _EpisodeSpec("episode-0", 2))
    _append_episode(store, _EpisodeSpec("episode-1", 2))
    return store


def _elite_sampler(seed: int, boost: float = 100.0) -> PrioritizedSampler:
    return PrioritizedSampler(
        IdentityFeaturePipeline(),
        alpha=1.0,
        elite_time_s=37.0,
        elite_priority_boost=boost,
        seed=seed,
    )


def _initialized_sampler(
    store: InMemoryReplayStore, seed: int
) -> tuple[PrioritizedSampler, BatchRequest]:
    sampler = _elite_sampler(seed)
    request = BatchRequest(batch_size=4)
    sampler.sample(store, request)
    sampler.update_priorities(PriorityUpdate([0, 1, 2, 3], [2.0, 3.0, 5.0, 7.0]))
    return sampler, request


def _checkpoint_source() -> tuple[InMemoryReplayStore, PrioritizedSampler, BatchRequest]:
    store = _two_episode_store()
    store.label_episode_sampling_pace("episode-0", 36.5)
    sampler, request = _initialized_sampler(store, 4)
    return store, sampler, request


def _restore_checkpoint(
    source_store: InMemoryReplayStore, source_sampler: PrioritizedSampler
) -> tuple[InMemoryReplayStore, PrioritizedSampler]:
    store = InMemoryReplayStore(capacity=8)
    store.load_state_dict(source_store.state_dict())
    sampler = _elite_sampler(99)
    sampler.load_state_dict(source_sampler.state_dict())
    return store, sampler


def _assert_restored_elite_index(
    store: InMemoryReplayStore,
    sampler: PrioritizedSampler,
    priorities: np.ndarray[Any, Any],
) -> None:
    weights = priorities[:4].copy()
    weights[:2] *= 100.0
    assert [store.sampling_pace_s(index) for index in (0, 1)] == pytest.approx([36.5, 36.5])
    assert sampler._elite_active_count == 2
    assert sampler._tree is not None
    assert sampler._tree.leaves[:4].tolist() == pytest.approx(weights.tolist())
    assert sampler._priorities.tolist() == pytest.approx(priorities.tolist())


def _reject_full_rebuild(_: prioritized_index._SynchronizationRequest) -> None:
    raise AssertionError("episode relabelling must not rebuild the full PER index")


def _sample_many(
    sampler: PrioritizedSampler, store: InMemoryReplayStore, request: BatchRequest
) -> list[int]:
    return [
        transition_id
        for _ in range(250)
        for transition_id in sampler.sample(store, request).transition_ids
    ]


def test_episode_sampling_pace_labels_only_active_matching_transitions_once() -> None:
    store = InMemoryReplayStore(capacity=8)
    _append_episode(store, _EpisodeSpec("fast", 3))
    _append_episode(store, _EpisodeSpec("other", 3))
    revision_before, _ = store.changes_since(None)

    assert store.label_episode_sampling_pace("fast", 36.75) == 3

    assert [store.sampling_pace_s(index) for index in range(3)] == pytest.approx([36.75] * 3)
    assert all(np.isinf(store.sampling_pace_s(index)) for index in range(3, 6))
    revision_after, changes = store.changes_since(revision_before)
    assert revision_after == revision_before + 1
    assert changes is not None
    assert len(changes) == 1
    assert changes[0].reclassified == (0, 1, 2)

    assert store.label_episode_sampling_pace("fast", 36.75) == 0
    assert store.changes_since(revision_after) == (revision_after, [])


def test_summary_first_pace_is_applied_to_later_episode_transitions() -> None:
    store = InMemoryReplayStore(capacity=4)
    revision_before, _ = store.changes_since(None)

    assert store.label_episode_sampling_pace("late-episode", 36.25) == 0
    assert store.changes_since(revision_before) == (revision_before, [])

    _append_episode(store, _EpisodeSpec("late-episode", 2))

    assert [store.sampling_pace_s(index) for index in (0, 1)] == pytest.approx([36.25, 36.25])
    revision_after, _ = store.changes_since(None)
    assert store.label_episode_sampling_pace("late-episode", 36.25) == 0
    assert store.changes_since(revision_after) == (revision_after, [])


def test_episode_sampling_pace_survives_checkpoint_round_trip() -> None:
    source = InMemoryReplayStore(capacity=4)
    _append_episode(source, _EpisodeSpec("completed", 2))
    source.label_episode_sampling_pace("completed", 36.5)

    restored = InMemoryReplayStore(capacity=4)
    restored.load_state_dict(source.state_dict())

    assert [restored.sampling_pace_s(index) for index in (0, 1)] == pytest.approx([36.5, 36.5])
    assert restored.label_episode_sampling_pace("completed", 36.5) == 0


def test_pending_episode_sampling_pace_survives_checkpoint_before_transitions() -> None:
    source = InMemoryReplayStore(capacity=4)
    assert source.label_episode_sampling_pace("pending", 36.125) == 0

    state = source.state_dict()
    restored = InMemoryReplayStore(capacity=4)
    restored.load_state_dict(state)
    _append_episode(restored, _EpisodeSpec("pending", 2))

    assert state["format"] == "columnar-v2"
    assert state["episode_sampling_paces"] == {"pending": pytest.approx(36.125)}
    assert [restored.sampling_pace_s(index) for index in (0, 1)] == pytest.approx([36.125, 36.125])


def test_store_and_prioritized_sampler_checkpoint_rebuilds_elite_index() -> None:
    source_store, source_sampler, request = _checkpoint_source()
    expected_priorities = source_sampler._priorities.copy()
    restored_store, restored_sampler = _restore_checkpoint(source_store, source_sampler)
    assert restored_sampler._tree is None

    batch = restored_sampler.sample(restored_store, request)

    _assert_restored_elite_index(restored_store, restored_sampler, expected_priorities)
    assert batch.metadata["replay/elite_active_fraction"] == 0.5
    assert batch.metadata["replay/elite_sample_fraction"] > 0.0


def test_pending_episode_sampling_pace_handles_out_of_order_interleaving() -> None:
    store = InMemoryReplayStore(capacity=8)
    store.label_episode_sampling_pace("late", 36.75)

    store.append(_transition(_EpisodeSpec("other", 2), 0))
    store.append(_transition(_EpisodeSpec("late", 3), 2))
    store.append(_transition(_EpisodeSpec("other", 2), 1))
    store.append(_transition(_EpisodeSpec("late", 3), 0))
    store.append(_transition(_EpisodeSpec("late", 3), 1))

    assert all(np.isinf(store.sampling_pace_s(index)) for index in (0, 2))
    assert [store.sampling_pace_s(index) for index in (1, 3, 4)] == pytest.approx([36.75] * 3)


def test_episode_sampling_pace_is_cleared_after_full_eviction_and_id_reuse() -> None:
    store = InMemoryReplayStore(capacity=2)
    store.label_episode_sampling_pace("old", 36.0)
    _append_episode(store, _EpisodeSpec("old", 2))
    assert [store.sampling_pace_s(index) for index in (0, 1)] == pytest.approx([36.0, 36.0])

    _append_episode(store, _EpisodeSpec("replacement", 2))

    assert all(np.isinf(store.sampling_pace_s(index)) for index in (2, 3))
    assert "old" not in store.state_dict()["episode_sampling_paces"]

    _append_episode(store, _EpisodeSpec("old", 1))

    assert np.isinf(store.sampling_pace_s(4))


def test_replay_checkpoint_requires_valid_episode_sampling_paces() -> None:
    state = InMemoryReplayStore().state_dict()
    state.pop("episode_sampling_paces")
    with pytest.raises(ValueError, match="missing required fields"):
        InMemoryReplayStore().load_state_dict(state)

    state = InMemoryReplayStore().state_dict()
    state["episode_sampling_paces"] = {"episode": "36.0"}
    with pytest.raises(ValueError, match="values must be floats"):
        InMemoryReplayStore().load_state_dict(state)


def test_replay_checkpoint_restores_v1_without_pending_episode_pace() -> None:
    source = InMemoryReplayStore(capacity=2)
    source.label_episode_sampling_pace("future", 36.0)
    state = source.state_dict()
    state["format"] = "columnar-v1"
    state.pop("episode_sampling_paces")

    restored = InMemoryReplayStore(capacity=2)
    restored.load_state_dict(state)
    _append_episode(restored, _EpisodeSpec("future", 1))

    assert np.isinf(restored.sampling_pace_s(0))
    assert restored.state_dict()["episode_sampling_paces"] == {}


@pytest.mark.parametrize("finish_time_s", [0.0, -1.0, float("nan"), float("inf")])
def test_episode_sampling_pace_rejects_invalid_finish_time(finish_time_s: float) -> None:
    store = InMemoryReplayStore()

    with pytest.raises(ValueError, match="positive finite float32"):
        store.label_episode_sampling_pace("episode", finish_time_s)


def test_prioritized_sampler_incrementally_reclassifies_relabelled_online_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _two_episode_store()
    sampler, request = _initialized_sampler(store, 3)
    priorities_before = sampler._priorities.copy()

    monkeypatch.setattr(prioritized_index, "_rebuild_index", _reject_full_rebuild)
    assert store.label_episode_sampling_pace("episode-0", 36.5) == 2
    batch = sampler.sample(store, request)

    assert batch.metadata["replay/elite_active_fraction"] == 0.5
    assert sampler._elite_active_count == 2
    assert sampler._elite_slots.tolist()[:4] == [True, True, False, False]
    assert sampler._priorities.tolist()[:4] == pytest.approx(priorities_before.tolist()[:4])
    sampled = _sample_many(sampler, store, request)
    assert sum(transition_id < 2 for transition_id in sampled) / len(sampled) > 0.95


def test_prioritized_sampler_clears_elite_classification_after_ring_eviction() -> None:
    store = InMemoryReplayStore(capacity=4)
    _append_episode(store, _EpisodeSpec("elite", 2, 50.0))
    _append_episode(store, _EpisodeSpec("regular", 2, 50.0))
    sampler = _elite_sampler(1, 4.0)
    request = BatchRequest(batch_size=4)
    assert store.label_episode_sampling_pace("elite", 36.0) == 2
    sampler.sample(store, request)
    assert sampler._elite_active_count == 2

    _append_episode(store, _EpisodeSpec("replacement", 2, 50.0))
    batch = sampler.sample(store, request)

    assert sampler._elite_active_count == 0
    assert batch.metadata["replay/elite_active_fraction"] == 0.0
    assert store.label_episode_sampling_pace("elite", 35.0) == 0
