"""TrackMania-specific collection adapter built on top of the neutral runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from tmrl.core.contracts import FeaturePipeline, Policy, ReplayStore
from tmrl.core.data import EpisodeArtifact, Transition


class TrackmaniaEnvironment(Protocol):
    """Gymnasium-compatible surface required by the TrackMania collector."""

    def reset(self, *, seed: int | None = None) -> tuple[Any, Mapping[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """Outcome of one collected episode and its lightweight artifact."""

    transitions: int
    total_reward: float
    artifact: EpisodeArtifact


class TrackmaniaCollector:
    """Turns a TrackMania episode into transitions and an episode artifact."""

    def __init__(self, store: ReplayStore, pipeline: FeaturePipeline, policy: Policy) -> None:
        self.store = store
        self.pipeline = pipeline
        self.policy = policy

    def collect(
        self, environment: TrackmaniaEnvironment, episode_id: str, max_steps: int
    ) -> CollectionResult:
        observation, reset_info = environment.reset()
        reset_pipeline = getattr(self.pipeline, "reset_episode", None)
        if callable(reset_pipeline):
            reset_pipeline()
        prepared = self.pipeline.transform_observation(observation)
        telemetry: list[Mapping[str, Any]] = []
        actions: list[Any] = []
        rewards: list[float] = []
        observation_refs: list[str] = []
        for step in range(max_steps):
            started = perf_counter()
            action = self.policy.act(prepared)
            latency_ms = (perf_counter() - started) * 1000.0
            next_observation, reward, terminated, truncated, info = environment.step(action)
            next_prepared = self.pipeline.transform_observation(next_observation)
            self.store.append(
                Transition(
                    observation=prepared,
                    action=action,
                    reward=float(reward),
                    next_observation=next_prepared,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    episode_id=episode_id,
                    step=step,
                )
            )
            telemetry.append({"step": step, "action_latency_ms": latency_ms, **dict(info)})
            actions.append(action)
            rewards.append(float(reward))
            observation_refs.append(str(info.get("observation_ref", "")))
            prepared = next_prepared
            if terminated or truncated:
                break
        artifact = EpisodeArtifact(
            episode_id=episode_id,
            telemetry=telemetry,
            actions=actions,
            rewards=rewards,
            observation_refs=observation_refs,
            metadata={
                "termination": str(info.get("termination_reason", "max_steps"))
                if rewards
                else "empty",
                "telemetry_health": str(
                    info.get("telemetry_health", reset_info.get("telemetry_health", "unknown"))
                )
                if rewards
                else str(reset_info.get("telemetry_health", "unknown")),
            },
        )
        return CollectionResult(len(rewards), sum(rewards), artifact)
