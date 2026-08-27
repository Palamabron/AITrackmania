"""First-party evaluation of a configured TrackMania environment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

from trackmaniarl.core.contracts import (
    EnvironmentFactory,
    EvaluatorRuntimeRequest,
    Policy,
    PolicyMode,
)
from trackmaniarl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec
from trackmaniarl.experiments.evaluation import EvaluationResult, aggregate_results
from trackmaniarl.trackmania.diagnostics import (
    ProgressBinDiagnostics,
    ProgressDiagnosticRecord,
    aggregate_progress_bins,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.session import PLUGIN_PROTOCOL_VERSION

_TRIAL_FIELDS = (
    "map_id",
    "map_uid",
    "trial_index",
    "steps",
    "finished",
    "finish_time_s",
    "crashed",
    "reward",
    "action_latency_ms",
    "controller_apply_ms",
    "telemetry_wait_ms",
    "telemetry_skipped_frames_total",
    "telemetry_skipped_frames_mean",
    "telemetry_skipped_frames_max",
    "telemetry_steps_with_skipped_frames_fraction",
    "throughput_fps",
    "progress_pct",
    "telemetry_error",
    "controller_error",
    "progress_bins",
)


@dataclass(frozen=True, slots=True)
class _EpisodeRequest:
    policy: Policy
    map_spec: EvaluationMapSpec
    trial_index: int
    seed: int
    environment: Any


@dataclass(slots=True)
class _EpisodeState:
    action_latency_ms: float = 0.0
    controller_apply_ms: float = 0.0
    telemetry_wait_ms: float = 0.0
    telemetry_skipped_frames_total: int = 0
    telemetry_skipped_frames_max: int = 0
    telemetry_steps_with_skipped_frames: int = 0
    reward_sum: float = 0.0
    steps: int = 0
    finished: bool = False
    crashed: bool = False
    finish_time_s: float | None = None
    progress_pct: float = 0.0
    telemetry_error: str | None = None


@dataclass(frozen=True, slots=True)
class _EpisodeContext:
    environment: Any
    diagnostics: ProgressBinDiagnostics
    started: float


@dataclass(frozen=True, slots=True)
class _EpisodeOutcome:
    request: _EpisodeRequest
    context: _EpisodeContext
    state: _EpisodeState


@dataclass(slots=True)
class _EpisodeLoop:
    policy: Policy
    environment: Any
    diagnostics: ProgressBinDiagnostics
    observation: Any
    prepared: Any


@dataclass(frozen=True, slots=True)
class _EvaluatedStep:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    action: Any
    action_duration_ms: float


def _validate_evaluator_request(request: EvaluatorRuntimeRequest) -> None:
    if not isinstance(request.suite, EvaluationSuiteSpec):
        raise TypeError("TrackmaniaEvaluator requires an EvaluationSuiteSpec")
    if request.environment_factory is None:
        raise ValueError("TrackmaniaEvaluator requires components.environment")
    if request.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")


class TrackmaniaEvaluator:
    """Evaluate a policy over the declared seeds and episode budget."""

    def __init__(self, request: EvaluatorRuntimeRequest) -> None:
        _validate_evaluator_request(request)
        self.suite = cast(EvaluationSuiteSpec, request.suite)
        self.environment_factory = cast(EnvironmentFactory, request.environment_factory)
        self.feature_pipeline = request.feature_pipeline
        self.max_episode_steps = request.max_episode_steps
        self.run_dir = Path(request.run_dir) if request.run_dir is not None else None
        self.checkpoint: str | None = None

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        """Attach the exact policy checkpoint to the next versioned evaluation artifact."""

        self.checkpoint = str(checkpoint)

    def evaluate(self, policy: Policy) -> dict[str, float]:
        """Run the fixed suite and return the standard comparable metric set."""

        results = self._evaluation_results(policy)
        metrics = self._evaluation_metrics(results)
        if self.run_dir is not None:
            self._write_artifact(results, metrics)
        return metrics

    def _evaluation_results(self, policy: Policy) -> list[EvaluationResult]:
        return [
            result for map_spec in self.suite.maps for result in self._map_results(policy, map_spec)
        ]

    def _map_results(self, policy: Policy, map_spec: EvaluationMapSpec) -> list[EvaluationResult]:
        geometry = BoundaryGeometry(
            map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
        )
        geometry.validate_map(map_spec.map_path)
        self._set_evaluation_map(map_spec)
        environment = self._create_environment(map_spec, seed=0)
        try:
            return [
                self._evaluate_episode(_EpisodeRequest(policy, map_spec, index, 0, environment))
                for index in range(self.suite.trials_per_map)
            ]
        finally:
            self._close_environment(environment)

    def _set_evaluation_map(self, map_spec: EvaluationMapSpec) -> None:
        set_evaluation_map = getattr(self.feature_pipeline, "set_evaluation_map", None)
        if callable(set_evaluation_map):
            set_evaluation_map(map_spec)

    def _evaluation_metrics(self, results: list[EvaluationResult]) -> dict[str, float]:
        metrics = dict(aggregate_results(results))
        metrics["eval/median_finish_time_s"] = self._median_finished(results)
        metrics.update(
            aggregate_progress_bins(
                result.progress_bins for result in results if result.progress_bins is not None
            )
        )
        return metrics

    @staticmethod
    def _median_finished(results: list[EvaluationResult]) -> float:
        times = [
            result.finish_time_s for result in results if result.finished and result.finish_time_s
        ]
        return float(median(times)) if times else 0.0

    def _evaluate_episode(self, request: _EpisodeRequest) -> EvaluationResult:
        context = self._episode_context(request)
        state = _EpisodeState()
        try:
            self._run_episode(request, context, state)
        except (TimeoutError, ConnectionError) as exc:
            state.telemetry_error = f"{type(exc).__name__}: {exc}"
        return self._episode_result(_EpisodeOutcome(request, context, state))

    def _episode_context(self, request: _EpisodeRequest) -> _EpisodeContext:
        diagnostics = ProgressBinDiagnostics(_policy_action_count(request.policy), bin_count=20)
        return _EpisodeContext(request.environment, diagnostics, perf_counter())

    def _run_episode(
        self, request: _EpisodeRequest, context: _EpisodeContext, state: _EpisodeState
    ) -> None:
        loop = self._start_episode(request, context)
        for _ in range(self.max_episode_steps):
            step = self._take_step(loop)
            if self._record_step(loop, state, step):
                break

    def _start_episode(self, request: _EpisodeRequest, context: _EpisodeContext) -> _EpisodeLoop:
        observation, _ = context.environment.reset(seed=request.seed)
        self._reset_component(self.feature_pipeline)
        self._reset_component(request.policy)
        prepared = self.feature_pipeline.transform_observation(observation)
        return _EpisodeLoop(
            request.policy, context.environment, context.diagnostics, observation, prepared
        )

    @staticmethod
    def _reset_component(component: Any) -> None:
        reset_episode = getattr(component, "reset_episode", None)
        if callable(reset_episode):
            reset_episode()

    def _take_step(self, loop: _EpisodeLoop) -> _EvaluatedStep:
        observation = (
            loop.observation
            if getattr(loop.policy, "requires_raw_observation", False)
            else loop.prepared
        )
        started = perf_counter()
        action = loop.policy.act(observation, PolicyMode.EVALUATION)
        action_duration_ms = (perf_counter() - started) * 1_000.0
        next_observation, reward, terminated, truncated, info = loop.environment.step(action)
        return _EvaluatedStep(
            next_observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
            action,
            action_duration_ms,
        )

    def _record_step(self, loop: _EpisodeLoop, state: _EpisodeState, step: _EvaluatedStep) -> bool:
        self._record_step_timings(state, step)
        record = ProgressDiagnosticRecord(
            float(step.info.get("progress_pct", state.progress_pct)),
            step.action,
            loop.policy,
            step.info,
        )
        loop.diagnostics.record(record)
        loop.observation = step.observation
        loop.prepared = self.feature_pipeline.transform_observation(step.observation)
        state.reward_sum += step.reward
        state.steps += 1
        self._record_outcome(state, step.info)
        return step.terminated or step.truncated

    @staticmethod
    def _record_step_timings(state: _EpisodeState, step: _EvaluatedStep) -> None:
        state.action_latency_ms += step.action_duration_ms
        state.controller_apply_ms += float(step.info.get("controller_apply_ms", 0.0))
        state.telemetry_wait_ms += float(step.info.get("telemetry_wait_ms", 0.0))
        skipped_frames = int(step.info.get("telemetry_skipped_frames", 0))
        state.telemetry_skipped_frames_total += skipped_frames
        state.telemetry_skipped_frames_max = max(state.telemetry_skipped_frames_max, skipped_frames)
        state.telemetry_steps_with_skipped_frames += int(skipped_frames > 0)

    @staticmethod
    def _record_outcome(state: _EpisodeState, info: dict[str, Any]) -> None:
        termination_reason = str(info.get("termination_reason", ""))
        state.progress_pct = float(info.get("progress_pct", state.progress_pct))
        state.finished = termination_reason == "finished"
        state.crashed = termination_reason in {"crashed", "off_track"}
        race_time_ms = info.get("race_time_ms")
        if state.finished and isinstance(race_time_ms, (float, int)) and race_time_ms > 0.0:
            state.finish_time_s = float(race_time_ms) / 1_000.0

    def _episode_result(self, outcome: _EpisodeOutcome) -> EvaluationResult:
        context, state = outcome.context, outcome.state
        elapsed_s = perf_counter() - context.started
        result = EvaluationResult(
            finished=state.finished,
            finish_time_s=(state.finish_time_s or elapsed_s) if state.finished else None,
            crashed=state.crashed,
            reward=state.reward_sum,
            action_latency_ms=state.action_latency_ms / max(state.steps, 1),
            throughput_fps=state.steps / elapsed_s if elapsed_s > 0.0 else 0.0,
            progress_pct=state.progress_pct,
        )
        return self._annotated_result(result, outcome)

    def _annotated_result(
        self, result: EvaluationResult, outcome: _EpisodeOutcome
    ) -> EvaluationResult:
        request, context, state = outcome.request, outcome.context, outcome.state
        identified = replace(
            result,
            map_id=request.map_spec.id,
            map_uid=request.map_spec.expected_map_uid,
            trial_index=request.trial_index,
            telemetry_error=state.telemetry_error,
            progress_bins=context.diagnostics.summary(),
            steps=state.steps,
        )
        return self._telemetry_result(identified, state)

    @staticmethod
    def _telemetry_result(result: EvaluationResult, state: _EpisodeState) -> EvaluationResult:
        steps = max(state.steps, 1)
        return replace(
            result,
            controller_apply_ms=state.controller_apply_ms / steps,
            telemetry_wait_ms=state.telemetry_wait_ms / steps,
            telemetry_skipped_frames_total=state.telemetry_skipped_frames_total,
            telemetry_skipped_frames_mean=state.telemetry_skipped_frames_total / steps,
            telemetry_skipped_frames_max=state.telemetry_skipped_frames_max,
            telemetry_steps_with_skipped_frames_fraction=(
                state.telemetry_steps_with_skipped_frames / steps
            ),
        )

    @staticmethod
    def _close_environment(environment: Any) -> None:
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    def _create_environment(self, map_spec: EvaluationMapSpec, *, seed: int) -> Any:
        factory = cast(Any, self.environment_factory)
        return factory.create(seed=seed, evaluation_map=map_spec)

    def _write_artifact(self, results: list[EvaluationResult], metrics: dict[str, float]) -> None:
        assert self.run_dir is not None
        target = self.run_dir / "evaluation.json"
        payload = self._artifact_payload(results, metrics)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)

    def _artifact_payload(
        self, results: list[EvaluationResult], metrics: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "plugin_protocol_version": PLUGIN_PROTOCOL_VERSION,
            "suite": {"name": self.suite.name, "version": self.suite.version},
            "checkpoint": self.checkpoint,
            "metrics": metrics,
            "trials": [self._trial_payload(result) for result in results],
        }

    @staticmethod
    def _trial_payload(result: EvaluationResult) -> dict[str, Any]:
        return {name: getattr(result, name) for name in _TRIAL_FIELDS}


def _policy_action_count(policy: Policy) -> int:
    model = getattr(policy, "model", None)
    count = getattr(policy, "action_count", getattr(model, "action_count", 78))
    if not isinstance(count, int) or count < 2:
        raise ValueError("TrackMania policy must expose at least two actions")
    return count
