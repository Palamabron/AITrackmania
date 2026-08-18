"""The TrackMania collector must work against a minimal Gymnasium-like adapter."""

from __future__ import annotations

from typing import Any

from trackmaniarl.core.builtins import IdentityFeaturePipeline, ZeroPolicy
from trackmaniarl.core.replay import InMemoryReplayStore
from trackmaniarl.trackmania.collector import TrackmaniaCollector


class FakeTrackmania:
    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
        del seed
        return {"speed": 0.0}, {}

    def step(self, action: Any) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
        del action
        self.step_index += 1
        return (
            {"speed": float(self.step_index)},
            1.0,
            self.step_index == 2,
            False,
            {"observation_ref": f"frame-{self.step_index}"},
        )


class ResettableZeroPolicy(ZeroPolicy):
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_episode(self) -> None:
        self.reset_count += 1


def test_collector_stores_transitions_and_keeps_only_observation_refs() -> None:
    store = InMemoryReplayStore()
    collector = TrackmaniaCollector(store, IdentityFeaturePipeline(), ZeroPolicy())
    result = collector.collect(FakeTrackmania(), "episode", max_steps=10)
    assert result.transitions == 2
    assert len(store) == 2
    assert result.artifact.observation_refs == ["frame-1", "frame-2"]


def test_collector_resets_policy_and_truncates_at_the_step_limit() -> None:
    store = InMemoryReplayStore()
    policy = ResettableZeroPolicy()
    collector = TrackmaniaCollector(store, IdentityFeaturePipeline(), policy)

    result = collector.collect(FakeTrackmania(), "episode", max_steps=1)
    transition = store.get(store.available_ids())[0]

    assert result.transitions == 1
    assert policy.reset_count == 1
    assert transition.truncated
    assert transition.info["termination_reason"] == "max_steps"
    assert result.artifact.metadata["termination"] == "max_steps"
