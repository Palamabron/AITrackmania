"""Environment-neutral episode collection over core runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from trackmaniarl.core.contracts import FeaturePipeline, Policy, ReplayStore
from trackmaniarl.core.data import EpisodeArtifact, Transition


class Environment(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[Any, Mapping[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class CollectionResult:
    transitions: int
    total_reward: float
    artifact: EpisodeArtifact


class EpisodeCollector:
    """Turn one environment episode into replay transitions and an artifact."""

    def __init__(self, store: ReplayStore, pipeline: FeaturePipeline, policy: Policy) -> None:
        self.store = store
        self.pipeline = pipeline
        self.policy = policy

    def collect(
        self, environment: Environment, episode_id: str, max_steps: int
    ) -> CollectionResult:
        observation, reset_info = environment.reset()
        self._reset_episode_state()
        prepared = self.pipeline.transform_observation(observation)
        telemetry: list[Mapping[str, Any]] = []
        actions: list[Any] = []
        rewards: list[float] = []
        observation_refs: list[str] = []
        info: Mapping[str, Any] = {}
        for step in range(max_steps):
            started = perf_counter()
            action, policy_info = self._act(prepared)
            latency_ms = (perf_counter() - started) * 1000.0
            next_observation, reward, terminated, truncated, info = environment.step(action)
            if step + 1 == max_steps and not terminated and not truncated:
                truncated = True
                info = {**info, "termination_reason": "max_steps"}
            next_prepared = self.pipeline.transform_observation(next_observation)
            self.store.append(
                Transition(
                    observation=prepared,
                    action=action,
                    reward=float(reward),
                    next_observation=next_prepared,
                    terminated=terminated,
                    truncated=truncated,
                    info={**info, **policy_info},
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
            metadata=self._artifact_metadata(info, reset_info, bool(rewards)),
        )
        return CollectionResult(len(rewards), sum(rewards), artifact)

    def _act(self, observation: Any) -> tuple[Any, Mapping[str, Any]]:
        sample = getattr(self.policy, "act_with_info", None)
        if not callable(sample):
            return self.policy.act(observation), {}
        action, info = sample(observation)
        if not isinstance(info, Mapping):
            raise TypeError("Policy act_with_info() must return action and a mapping")
        return action, info

    def _reset_episode_state(self) -> None:
        for component in (self.pipeline, self.policy):
            reset = getattr(component, "reset_episode", None)
            if callable(reset):
                reset()

    @staticmethod
    def _artifact_metadata(
        info: Mapping[str, Any], reset_info: Mapping[str, Any], has_rewards: bool
    ) -> dict[str, str]:
        return {
            "termination": str(info.get("termination_reason", "max_steps"))
            if has_rewards
            else "empty",
            "telemetry_health": str(
                info.get("telemetry_health", reset_info.get("telemetry_health", "unknown"))
            )
            if has_rewards
            else str(reset_info.get("telemetry_health", "unknown")),
        }
