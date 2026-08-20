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
    completed_episodes: int = 1


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


class FixedStepRolloutCollector:
    """Collect fixed-size on-policy segments without resetting at segment boundaries."""

    def __init__(
        self,
        store: ReplayStore,
        pipeline: FeaturePipeline,
        policy: Policy,
        environment: Environment,
        *,
        max_episode_steps: int,
        seed: int = 0,
        start_episode_index: int = 0,
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.policy = policy
        self.environment = environment
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.episode_index = start_episode_index
        self.episode_step = 0
        self.prepared: Any = None
        self.reset_info: Mapping[str, Any] = {}

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy

    def collect(self, transition_count: int, rollout_id: str) -> CollectionResult:
        telemetry: list[Mapping[str, Any]] = []
        actions: list[Any] = []
        rewards: list[float] = []
        observation_refs: list[str] = []
        completed_episodes = 0
        for rollout_step in range(transition_count):
            self._ensure_episode()
            action, policy_info, latency_ms = self._act()
            info, reward, episode_ended = self._step(action, policy_info)
            completed_episodes += int(episode_ended)
            telemetry.append(
                {
                    "step": rollout_step,
                    "episode_step": self.episode_step - 1,
                    "action_latency_ms": latency_ms,
                    **dict(info),
                }
            )
            actions.append(action)
            rewards.append(reward)
            observation_refs.append(str(info.get("observation_ref", "")))
        artifact = EpisodeArtifact(
            episode_id=rollout_id,
            telemetry=telemetry,
            actions=actions,
            rewards=rewards,
            observation_refs=observation_refs,
            metadata={"termination": "fixed_rollout", "telemetry_health": "ok"},
        )
        return CollectionResult(transition_count, sum(rewards), artifact, completed_episodes)

    def _ensure_episode(self) -> None:
        if self.prepared is not None:
            return
        observation, self.reset_info = self.environment.reset(seed=self.seed + self.episode_index)
        self._reset_episode_state()
        self.prepared = self.pipeline.transform_observation(observation)
        self.episode_step = 0

    def _act(self) -> tuple[Any, Mapping[str, Any], float]:
        started = perf_counter()
        sample = getattr(self.policy, "act_with_info", None)
        if callable(sample):
            action, info = sample(self.prepared)
        else:
            action, info = self.policy.act(self.prepared), {}
        if not isinstance(info, Mapping):
            raise TypeError("Policy act_with_info() must return action and a mapping")
        return action, info, (perf_counter() - started) * 1000.0

    def _step(
        self, action: Any, policy_info: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float, bool]:
        next_observation, reward, terminated, truncated, info = self.environment.step(action)
        self.episode_step += 1
        if self.episode_step == self.max_episode_steps and not terminated and not truncated:
            truncated = True
            info = {**info, "termination_reason": "max_steps"}
        next_prepared = self.pipeline.transform_observation(next_observation)
        self.store.append(
            Transition(
                observation=self.prepared,
                action=action,
                reward=float(reward),
                next_observation=next_prepared,
                terminated=terminated,
                truncated=truncated,
                info={**info, **policy_info},
                episode_id=f"episode-{self.episode_index:08d}",
                step=self.episode_step - 1,
            )
        )
        self.prepared = next_prepared
        if terminated or truncated:
            self.prepared = None
            self.episode_index += 1
        return info, float(reward), terminated or truncated

    def _reset_episode_state(self) -> None:
        for component in (self.pipeline, self.policy):
            reset = getattr(component, "reset_episode", None)
            if callable(reset):
                reset()
