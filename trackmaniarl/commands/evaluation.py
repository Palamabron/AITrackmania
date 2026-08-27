"""Checkpoint benchmark command and shared benchmark reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from trackmaniarl.commands.helpers import _training_learner_state
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import EvaluationSuiteSpec, RunSpec


@dataclass(frozen=True, slots=True)
class _BenchmarkArtifact:
    trials: list[dict[str, Any]]
    checkpoint: str | None
    evaluation: EvaluationSuiteSpec
    checkpoint_path: Path


def _benchmark(args: argparse.Namespace) -> None:
    spec, evaluation = _benchmark_spec(args)
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    trials, metrics, checkpoint = _evaluate_checkpoint(run, args.checkpoint)
    artifact = _BenchmarkArtifact(trials, checkpoint, evaluation, args.checkpoint)
    _validate_benchmark_artifact(artifact)
    _print_benchmark_report(trials, metrics)
    _apply_benchmark_gate(trials, metrics, evaluation)


def _benchmark_spec(args: argparse.Namespace) -> tuple[RunSpec, EvaluationSuiteSpec]:
    spec = RunSpec.from_yaml(args.config)
    evaluation = _require_evaluation(spec, "benchmark")
    updates = _evaluation_overrides(args)
    if updates:
        evaluation = evaluation.model_copy(update=updates)
        spec = spec.model_copy(update={"evaluation": evaluation})
    if evaluation.target_median_s is None:
        raise ValueError(
            "benchmark requires evaluation.target_median_s "
            "(for example 37.0 for a sub-37s release gate)"
        )
    return spec, evaluation


def _evaluation_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    candidates = (
        ("trials_per_map", getattr(args, "trials", None)),
        ("target_median_s", getattr(args, "target_median", None)),
        ("min_finish_rate", getattr(args, "min_finish_rate", None)),
    )
    return {key: value for key, value in candidates if value is not None}


def _require_evaluation(spec: RunSpec, command: str) -> EvaluationSuiteSpec:
    evaluation = spec.evaluation
    if evaluation is None or not evaluation.maps:
        raise ValueError(f"{command} requires an evaluation suite with at least one map")
    return evaluation


def _evaluate_checkpoint(
    run: ResolvedRun, checkpoint_path: Path
) -> tuple[list[dict[str, Any]], dict[str, float], str | None]:
    if run.evaluator is None:
        raise ValueError("benchmark requires components.evaluator")
    try:
        _load_checkpoint(run, checkpoint_path)
        metrics = dict(run.evaluator.evaluate(run.learner.policy()))
        artifact = _load_evaluation_artifact(run.run_dir)
    finally:
        run.logger.close()
    return _artifact_trials(artifact), metrics, _artifact_checkpoint(artifact)


def _load_checkpoint(run: ResolvedRun, checkpoint_path: Path) -> None:
    run.learner.setup(
        {"seed": run.spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}
    )
    checkpoint = run.checkpoint_codec.load(checkpoint_path)
    learner_state = _training_learner_state(checkpoint)
    load_policy_state = getattr(run.learner, "load_policy_state_dict", None)
    if callable(load_policy_state):
        load_policy_state(learner_state)
    else:
        run.learner.load_state_dict(learner_state)
    set_checkpoint = getattr(run.evaluator, "set_checkpoint", None)
    if callable(set_checkpoint):
        set_checkpoint(checkpoint_path)


def _load_evaluation_artifact(run_dir: Path) -> dict[str, Any]:
    raw = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("evaluation.json must contain an object")
    return cast(dict[str, Any], raw)


def _artifact_trials(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    trials = artifact["trials"]
    if not isinstance(trials, list) or not all(isinstance(trial, dict) for trial in trials):
        raise TypeError("evaluation.json trials must be a list of objects")
    return cast(list[dict[str, Any]], trials)


def _artifact_checkpoint(artifact: dict[str, Any]) -> str | None:
    checkpoint = artifact["checkpoint"]
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise TypeError("evaluation.json checkpoint must be a string or null")
    return checkpoint


def _validate_benchmark_artifact(artifact: _BenchmarkArtifact) -> None:
    if artifact.checkpoint != str(artifact.checkpoint_path):
        raise RuntimeError("benchmark artifact checkpoint does not match the evaluated checkpoint")
    trials = artifact.trials
    evaluation = artifact.evaluation
    expected_trials = evaluation.trials_per_map * len(evaluation.maps)
    expected_maps = {item.id for item in evaluation.maps}
    observed_maps = {str(trial["map_id"]) for trial in trials}
    if len(trials) != expected_trials or observed_maps != expected_maps:
        raise RuntimeError(
            f"benchmark artifact must contain exactly {expected_trials} trials covering "
            f"{sorted(expected_maps)}"
        )


def _apply_benchmark_gate(
    trials: list[dict[str, Any]],
    metrics: dict[str, float],
    evaluation: EvaluationSuiteSpec,
) -> None:
    completed = _completed_trials(trials)
    required = ceil(evaluation.min_finish_rate * len(trials))
    median = float(metrics["eval/median_finish_time_s"])
    passed = len(completed) >= required and median < _target_median(evaluation)
    if not passed or _has_runtime_errors(trials):
        raise RuntimeError(
            "benchmark failed: require "
            f">={required}/{len(trials)} finishes, "
            f"median completed time <{_target_median(evaluation)}s, "
            "and no telemetry/controller errors"
        )
    print(f"Benchmark passed: {len(completed)}/{len(trials)} finishes, median {median:.3f}s")


def _target_median(evaluation: EvaluationSuiteSpec) -> float:
    target = evaluation.target_median_s
    if target is None:
        raise ValueError("evaluation.target_median_s is required")
    return target


def _completed_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial["finished"]]


def _has_runtime_errors(trials: list[dict[str, Any]]) -> bool:
    return any(
        trial["telemetry_error"] is not None or trial["controller_error"] is not None
        for trial in trials
    )


def _print_benchmark_report(trials: list[dict[str, Any]], metrics: dict[str, float]) -> None:
    completed = _completed_trials(trials)
    print("Benchmark trials:")
    for trial in trials:
        _print_benchmark_trial(trial)
    print(
        f"Benchmark summary: finishes={len(completed)}/{len(trials)}, "
        f"mean_completed={float(metrics['eval/finish_time_s']):.3f}s, "
        f"median_completed={float(metrics['eval/median_finish_time_s']):.3f}s"
    )
    _print_benchmark_confidence(completed, len(trials))


def _print_benchmark_trial(trial: dict[str, Any]) -> None:
    finish_time = trial["finish_time_s"]
    time_text = "-" if finish_time is None else f"{float(finish_time):.3f}s"
    print(
        f"  trial={trial['trial_index']} map={trial['map_id']} "
        f"finished={trial['finished']} time={time_text} "
        f"progress={float(trial['progress_pct']):.1f}% "
        f"telemetry_error={trial['telemetry_error'] or '-'} "
        f"controller_error={trial['controller_error'] or '-'}"
    )


def _print_benchmark_confidence(completed: list[dict[str, Any]], trial_count: int) -> None:
    finish_low, finish_high = _wilson_interval(len(completed), trial_count)
    finish_times = [float(trial["finish_time_s"]) for trial in completed]
    interval = _bootstrap_median_interval(finish_times)
    median_text = "n/a" if interval is None else f"[{interval[0]:.3f}s, {interval[1]:.3f}s]"
    print(
        f"Benchmark 95% CI: finish_rate Wilson=[{finish_low:.4f}, {finish_high:.4f}], "
        f"median_completed bootstrap={median_text}"
    )


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials or not 0.0 < confidence < 1.0:
        raise ValueError("Wilson interval requires valid successes, trials, and confidence")
    probability = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * sqrt(probability * (1.0 - probability) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _bootstrap_median_interval(
    values: list[float], confidence: float = 0.95, samples: int = 10_000
) -> tuple[float, float] | None:
    if not values:
        return None
    if samples < 1 or not 0.0 < confidence < 1.0 or not np.isfinite(values).all():
        raise ValueError("bootstrap interval requires finite values, samples, and confidence")
    observed = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(0)
    indices = generator.integers(0, len(observed), size=(samples, len(observed)))
    medians = np.median(observed[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, (tail, 1.0 - tail))
    return float(low), float(high)
