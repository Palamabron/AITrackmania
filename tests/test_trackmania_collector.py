"""The TrackMania collector must work against a minimal Gymnasium-like adapter."""

from __future__ import annotations

from typing import Any

from tmrl.core.builtins import IdentityFeaturePipeline, ZeroPolicy
from tmrl.core.replay import InMemoryReplayStore
from tmrl.trackmania.collector import TrackmaniaCollector


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


def test_collector_stores_transitions_and_keeps_only_observation_refs() -> None:
    store = InMemoryReplayStore()
    collector = TrackmaniaCollector(store, IdentityFeaturePipeline(), ZeroPolicy())
    result = collector.collect(FakeTrackmania(), "episode", max_steps=10)
    assert result.transitions == 2
    assert len(store) == 2
    assert result.artifact.observation_refs == ["frame-1", "frame-2"]
