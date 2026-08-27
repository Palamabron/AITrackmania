"""Environment-neutral episode collection over core runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class _EpisodeTrace:
    telemetry: list[Mapping[str, Any]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    observation_refs: list[str] = field(default_factory=list)
    info: Mapping[str, Any] = field(default_factory=dict)

    def record(self, record: _EpisodeRecord) -> None:
        self.telemetry.append(
            {"step": record.step, "action_latency_ms": record.latency_ms, **dict(self.info)}
        )
        self.actions.append(record.action)
        self.rewards.append(record.reward)
        self.observation_refs.append(str(self.info.get("observation_ref", "")))


@dataclass(frozen=True, slots=True)
class _EpisodeRecord:
    step: int
    action: Any
    reward: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class _EpisodeStep:
    environment: Environment
    episode_id: str
    step: int
    max_steps: int


@dataclass(frozen=True, slots=True)
class _EpisodeTransition:
    context: _EpisodeStep
    observations: tuple[Any, Any]
    outcome: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _EpisodeEndState:
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RolloutCollectionConfig:
    policy: Policy
    environment: Environment
    max_episode_steps: int
    seed: int = 0
    start_episode_index: int = 0


@dataclass(slots=True)
class _RolloutTrace:
    telemetry: list[Mapping[str, Any]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    observation_refs: list[str] = field(default_factory=list)
    completed_episodes: int = 0

    def record(self, step: int, result: tuple[Mapping[str, Any], float, bool, Any]) -> None:
        info, reward, ended, action = result
        self.completed_episodes += int(ended)
        self.telemetry.append({"step": step, **dict(info)})
        self.actions.append(action)
        self.rewards.append(reward)
        self.observation_refs.append(str(info.get("observation_ref", "")))


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
        trace = _EpisodeTrace()
        for step in range(max_steps):
            context = _EpisodeStep(environment, episode_id, step, max_steps)
            prepared, ended = self._collect_step(context, prepared, trace)
            if ended:
                break
        return self._episode_result(episode_id, trace, reset_info)

    def _collect_step(
        self, context: _EpisodeStep, prepared: Any, trace: _EpisodeTrace
    ) -> tuple[Any, bool]:
        action, policy_info, latency_ms = self._timed_act(prepared)
        outcome = self._episode_outcome(context, action)
        next_observation, reward, terminated, truncated, info = outcome
        next_prepared = self.pipeline.transform_observation(next_observation)
        transition = self._transition(
            _EpisodeTransition(
                context,
                (prepared, next_prepared),
                (action, reward, terminated, truncated, info, policy_info),
            )
        )
        self.store.append(transition)
        trace.info = info
        trace.record(_EpisodeRecord(context.step, action, float(reward), latency_ms))
        return next_prepared, terminated or truncated

    def _timed_act(self, observation: Any) -> tuple[Any, Mapping[str, Any], float]:
        started = perf_counter()
        action, policy_info = self._act(observation)
        return action, policy_info, (perf_counter() - started) * 1000.0

    @staticmethod
    def _episode_outcome(context: _EpisodeStep, action: Any) -> tuple[Any, ...]:
        next_observation, reward, terminated, truncated, info = context.environment.step(action)
        reached_limit = context.step + 1 == context.max_steps
        if reached_limit and not terminated and not truncated:
            return (
                next_observation,
                reward,
                terminated,
                True,
                {
                    **info,
                    "termination_reason": "max_steps",
                },
            )
        return next_observation, reward, terminated, truncated, info

    @staticmethod
    def _transition(build: _EpisodeTransition) -> Transition:
        action, reward, terminated, truncated, info, policy_info = build.outcome
        prepared, next_prepared = build.observations
        return Transition(
            observation=prepared,
            action=action,
            reward=float(reward),
            next_observation=next_prepared,
            terminated=terminated,
            truncated=truncated,
            info={**info, **policy_info},
            episode_id=build.context.episode_id,
            step=build.context.step,
        )

    def _episode_result(
        self, episode_id: str, trace: _EpisodeTrace, reset_info: Mapping[str, Any]
    ) -> CollectionResult:
        artifact = EpisodeArtifact(
            episode_id=episode_id,
            telemetry=trace.telemetry,
            actions=trace.actions,
            rewards=trace.rewards,
            observation_refs=trace.observation_refs,
            metadata=self._artifact_metadata(trace, reset_info),
        )
        return CollectionResult(len(trace.rewards), sum(trace.rewards), artifact)

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
    def _artifact_metadata(trace: _EpisodeTrace, reset_info: Mapping[str, Any]) -> dict[str, str]:
        if not trace.rewards:
            return {
                "termination": "empty",
                "telemetry_health": str(reset_info.get("telemetry_health", "unknown")),
            }
        return {
            "termination": str(trace.info.get("termination_reason", "max_steps")),
            "telemetry_health": str(
                trace.info.get("telemetry_health", reset_info.get("telemetry_health", "unknown"))
            ),
        }


class FixedStepRolloutCollector:
    """Collect fixed-size on-policy segments without resetting at segment boundaries."""

    def __init__(
        self, store: ReplayStore, pipeline: FeaturePipeline, config: RolloutCollectionConfig
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.policy = config.policy
        self.environment = config.environment
        self.max_episode_steps = config.max_episode_steps
        self.seed = config.seed
        self.episode_index = config.start_episode_index
        self.episode_step = 0
        self.prepared: Any = None
        self.reset_info: Mapping[str, Any] = {}

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy

    def collect(self, transition_count: int, rollout_id: str) -> CollectionResult:
        trace = _RolloutTrace()
        for rollout_step in range(transition_count):
            self._ensure_episode()
            trace.record(rollout_step, self._collect_rollout_step())
        return self._rollout_result(transition_count, rollout_id, trace)

    def _collect_rollout_step(self) -> tuple[Mapping[str, Any], float, bool, Any]:
        action, policy_info, latency_ms = self._act()
        info, reward, episode_ended = self._step(action, policy_info)
        details = {
            "episode_step": self.episode_step - 1,
            "action_latency_ms": latency_ms,
            **dict(info),
        }
        return details, reward, episode_ended, action

    @staticmethod
    def _rollout_result(
        transition_count: int, rollout_id: str, trace: _RolloutTrace
    ) -> CollectionResult:
        artifact = EpisodeArtifact(
            episode_id=rollout_id,
            telemetry=trace.telemetry,
            actions=trace.actions,
            rewards=trace.rewards,
            observation_refs=trace.observation_refs,
            metadata={"termination": "fixed_rollout", "telemetry_health": "ok"},
        )
        return CollectionResult(
            transition_count, sum(trace.rewards), artifact, trace.completed_episodes
        )

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
        end_state = self._apply_episode_limit(_EpisodeEndState(terminated, truncated, info))
        truncated, info = end_state.truncated, end_state.info
        next_prepared = self.pipeline.transform_observation(next_observation)
        transition = self._rollout_transition(
            action, next_prepared, (reward, terminated, truncated, info, policy_info)
        )
        self.store.append(transition)
        self.prepared = next_prepared
        if terminated or truncated:
            self._finish_episode()
        return info, float(reward), terminated or truncated

    def _apply_episode_limit(self, state: _EpisodeEndState) -> _EpisodeEndState:
        if self.episode_step != self.max_episode_steps or state.terminated or state.truncated:
            return state
        return _EpisodeEndState(
            state.terminated, True, {**state.info, "termination_reason": "max_steps"}
        )

    def _rollout_transition(
        self, action: Any, next_prepared: Any, outcome: tuple[Any, ...]
    ) -> Transition:
        reward, terminated, truncated, info, policy_info = outcome
        return Transition(
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

    def _finish_episode(self) -> None:
        self.prepared = None
        self.episode_index += 1

    def _reset_episode_state(self) -> None:
        for component in (self.pipeline, self.policy):
            reset = getattr(component, "reset_episode", None)
            if callable(reset):
                reset()
