"""Contract tests for all-step recurrent sequence training and demo protection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest
import torch

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, PrioritizedSampler


class _EpisodeKind(StrEnum):
    ONLINE = "online"
    DEMONSTRATION = "demonstration"


@dataclass(frozen=True)
class _EpisodeSpec:
    episode_id: str
    steps: int
    kind: _EpisodeKind = _EpisodeKind.ONLINE


def _fill_episode(store: InMemoryReplayStore, spec: _EpisodeSpec) -> None:
    for step in range(spec.steps):
        store.append(
            Transition(
                observation=float(step),
                action=0.0,
                reward=1.0,
                next_observation=float(step + 1),
                terminated=step == spec.steps - 1,
                truncated=False,
                episode_id=spec.episode_id,
                step=step,
                info={"is_demo": spec.kind is _EpisodeKind.DEMONSTRATION},
            )
        )


def _assert_demo_transitions_survive_fifo_eviction() -> None:
    store = InMemoryReplayStore(capacity=16)
    _fill_episode(store, _EpisodeSpec("demo-lap", 4, _EpisodeKind.DEMONSTRATION))
    for episode in range(10):
        _fill_episode(store, _EpisodeSpec(f"online-{episode}", 4))

    flags = store.demo_flags(store.available_ids())

    assert sum(flags) == 4
    demo_ids = [
        transition_id
        for transition_id, flag in zip(store.available_ids(), flags, strict=True)
        if flag
    ]
    resurrected = store.get(demo_ids)
    assert sorted(item.step for item in resurrected) == [0, 1, 2, 3]
    assert all(item.episode_id == "demo-lap" for item in resurrected)


def _assert_prioritized_sequence_masks_cover_complete_histories() -> None:
    store = InMemoryReplayStore()
    _fill_episode(store, _EpisodeSpec("episode-0", 6))
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=3)

    batch = sampler.sample(store, BatchRequest(batch_size=3, sequence_length=4, n_step=1))

    assert isinstance(batch.masks, torch.Tensor)
    assert batch.masks.shape == (3, 4)
    assert bool(batch.masks.all())
    assert batch.metadata["gamma"] == pytest.approx(0.99)
    assert batch.metadata["n_step"] == 1
    assert len(batch.metadata["demo_flags"]) == 3


def test_replay_preserves_demonstrations_and_sequence_histories() -> None:
    _assert_demo_transitions_survive_fifo_eviction()
    _assert_prioritized_sequence_masks_cover_complete_histories()
