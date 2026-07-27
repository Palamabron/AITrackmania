"""First-party evaluation of a configured TrackMania environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

from tmrl.core.contracts import EnvironmentFactory, FeaturePipeline, Policy
from tmrl.core.spec import EvaluationSuiteSpec
from tmrl.experiments.evaluation import EvaluationResult, aggregate_results
from tmrl.trackmania.geometry import BoundaryGeometry
from tmrl.trackmania.session import PLUGIN_PROTOCOL_VERSION


class TrackmaniaEvaluator:
    """Evaluate a policy over the declared seeds and episode budget.

    Each evaluation episode requests one explicit local map and session-ready
    UID confirmation from the environment factory.
    """

    def __init__(
        self,
        suite: EvaluationSuiteSpec | None,
        environment_factory: EnvironmentFactory | None,
        feature_pipeline: FeaturePipeline,
        max_episode_steps: int = 2_000,
        run_dir: str | Path | None = None,
    ) -> None:
        if suite is None:
            raise ValueError("TrackmaniaEvaluator requires a configured evaluation suite")
        if environment_factory is None:
            raise ValueError("TrackmaniaEvaluator requires components.environment")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.suite = suite
        self.environment_factory = environment_factory
        self.feature_pipeline = feature_pipeline
        self.max_episode_steps = max_episode_steps
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.checkpoint: str | None = None

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        """Attach the exact policy checkpoint to the next versioned evaluation artifact."""

        self.checkpoint = str(checkpoint)

    def evaluate(self, policy: Policy) -> dict[str, float]:
        """Run the fixed suite and return the standard comparable metric set."""

        results: list[EvaluationResult] = []
        maps = getattr(self.suite, "maps", ())
        if maps:
            for map_spec in maps:
                geometry = BoundaryGeometry(
                    map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
                )
                geometry.validate_map(map_spec.map_path)
                set_evaluation_map = getattr(self.feature_pipeline, "set_evaluation_map", None)
                if callable(set_evaluation_map):
                    set_evaluation_map(map_spec)
                for trial_index in range(self.suite.trials_per_map):
                    results.append(self._evaluate_episode(policy, map_spec, trial_index))
        else:  # Compatibility for programmatic pre-1.1 test doubles only.
            for seed in getattr(self.suite, "seeds", (0,)):
                for trial_index in range(getattr(self.suite, "episodes_per_seed", 1)):
                    results.append(self._evaluate_episode(policy, None, trial_index, seed=seed))
        metrics = dict(aggregate_results(results))
        metrics["eval/median_finish_time_s"] = self._median_finished(results)
        if self.run_dir is not None:
            self._write_artifact(results, metrics)
        return metrics

    @staticmethod
    def _median_finished(results: list[EvaluationResult]) -> float:
        times = [
            result.finish_time_s for result in results if result.finished and result.finish_time_s
        ]
        return float(median(times)) if times else 0.0

    def _evaluate_episode(
        self, policy: Policy, map_spec: Any | None, trial_index: int, *, seed: int = 0
    ) -> EvaluationResult:
        if map_spec is None:
            environment = self.environment_factory.create(seed=seed)
        else:
            try:
                factory = cast(Any, self.environment_factory)
                environment = factory.create(seed=0, evaluation_map=map_spec)
            except TypeError as exc:
                raise RuntimeError(
                    "evaluation requires an environment factory accepting evaluation_map"
                ) from exc
        action_latency_ms = 0.0
        reward_sum = 0.0
        steps = 0
        finished = False
        crashed = False
        termination_reason: str | None = None
        finish_time_s: float | None = None
        progress_pct = 0.0
        telemetry_error: str | None = None
        started = perf_counter()
        try:
            observation, _ = environment.reset(seed=seed)
            reset_pipeline = getattr(self.feature_pipeline, "reset_episode", None)
            if callable(reset_pipeline):
                reset_pipeline()
            reset_policy = getattr(policy, "reset_episode", None)
            if callable(reset_policy):
                reset_policy()
            prepared = self.feature_pipeline.transform_observation(observation)
            for _ in range(self.max_episode_steps):
                action_started = perf_counter()
                action = policy.act(prepared, deterministic=True)
                action_latency_ms += (perf_counter() - action_started) * 1_000.0
                observation, reward, terminated, truncated, info = environment.step(action)
                prepared = self.feature_pipeline.transform_observation(observation)
                reward_sum += float(reward)
                steps += 1
                termination_reason = str(info.get("termination_reason", ""))
                progress_pct = float(info.get("progress_pct", progress_pct))
                finished = termination_reason == "finished"
                crashed = termination_reason in {"crashed", "off_track"}
                if finished:
                    race_time_ms = info.get("race_time_ms")
                    if isinstance(race_time_ms, (float, int)) and race_time_ms > 0.0:
                        finish_time_s = float(race_time_ms) / 1_000.0
                if terminated or truncated:
                    break
        except (TimeoutError, ConnectionError) as exc:
            telemetry_error = f"{type(exc).__name__}: {exc}"
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        elapsed_s = perf_counter() - started
        return EvaluationResult(
            finished=finished,
            finish_time_s=(finish_time_s if finish_time_s is not None else elapsed_s)
            if finished
            else None,
            crashed=crashed,
            reward=reward_sum,
            action_latency_ms=action_latency_ms / max(steps, 1),
            throughput_fps=steps / elapsed_s if elapsed_s > 0.0 else 0.0,
            progress_pct=progress_pct,
            map_id="" if map_spec is None else map_spec.id,
            map_uid="" if map_spec is None else map_spec.expected_map_uid,
            trial_index=trial_index,
            telemetry_error=telemetry_error,
        )

    def _write_artifact(self, results: list[EvaluationResult], metrics: dict[str, float]) -> None:
        assert self.run_dir is not None
        target = self.run_dir / "evaluation.json"
        payload = {
            "schema_version": "1",
            "plugin_protocol_version": PLUGIN_PROTOCOL_VERSION,
            "suite": {"name": self.suite.name, "version": self.suite.version},
            "checkpoint": self.checkpoint,
            "metrics": metrics,
            "trials": [
                {
                    "map_id": result.map_id,
                    "map_uid": result.map_uid,
                    "trial_index": result.trial_index,
                    "finished": result.finished,
                    "finish_time_s": result.finish_time_s,
                    "crashed": result.crashed,
                    "reward": result.reward,
                    "action_latency_ms": result.action_latency_ms,
                    "throughput_fps": result.throughput_fps,
                    "progress_pct": result.progress_pct,
                    "telemetry_error": result.telemetry_error,
                    "controller_error": result.controller_error,
                }
                for result in results
            ],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
