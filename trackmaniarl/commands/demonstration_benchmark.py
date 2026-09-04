"""Human-demonstration replay benchmark command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from trackmaniarl.commands.evaluation import (
    _artifact_trials,
    _has_runtime_errors,
    _load_evaluation_artifact,
    _print_benchmark_report,
    _require_evaluation,
    _target_median,
)
from trackmaniarl.commands.trajectory import _with_environment_decision_interval
from trackmaniarl.core.contracts import Policy
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import EvaluationSuiteSpec, RunSpec
from trackmaniarl.trackmania.demonstrations import (
    demonstration_timing_summary,
    load_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.guidance import (
    DemonstrationReplayPolicy,
    PhaseLockedDemonstrationPolicy,
    TrajectoryTrackingDemonstrationPolicy,
)
from trackmaniarl.trackmania.guidance_phase import PhaseLockedPathRequest
from trackmaniarl.trackmania.guidance_replay import ReplayPathRequest, ReplaySamplingConfig
from trackmaniarl.trackmania.guidance_tracking import TrajectoryTrackingPathRequest
from trackmaniarl.trackmania.trajectory_optimization import (
    TrajectorySchedule,
    TrajectoryTrackerConfig,
    build_scheduled_policy,
)


@dataclass(frozen=True, slots=True)
class _ReplaySettings:
    open_loop: bool
    interval_ms: float | None
    action_lead_ms: float
    aggregate_controls: bool


@dataclass(frozen=True, slots=True)
class _PreparedReplay:
    spec: RunSpec
    evaluation: EvaluationSuiteSpec
    settings: _ReplaySettings
    demonstration: Any | None


@dataclass(frozen=True, slots=True)
class _ReplayInterval:
    settings: _ReplaySettings
    trajectory_tracking: bool
    interval_ms: float | None


@dataclass(frozen=True, slots=True)
class _ReplayPolicyContext:
    run: ResolvedRun
    prepared: _PreparedReplay
    args: argparse.Namespace
    config: TrackmaniaEnvironmentConfig


@dataclass(frozen=True, slots=True)
class _ReplayGate:
    trials: list[dict[str, Any]]
    metrics: dict[str, float]
    evaluation: EvaluationSuiteSpec
    report_only: bool


def _demo_benchmark(args: argparse.Namespace) -> None:
    _validate_replay_arguments(args)
    prepared = _prepare_replay(args)
    run = resolve_run(prepared.spec, base_dir=Path(args.config).parent)
    if not isinstance(run.evaluator, TrackmaniaEvaluator):
        raise ValueError("demo-benchmark requires components.evaluator")
    policy, trials, metrics = _evaluate_replay(run, prepared, args)
    _print_benchmark_report(trials, metrics)
    if isinstance(policy, TrajectoryTrackingDemonstrationPolicy):
        _print_trajectory_tracker_diagnostics(policy)
    _apply_replay_gate(_ReplayGate(trials, metrics, prepared.evaluation, args.report_only))


def _validate_replay_arguments(args: argparse.Namespace) -> None:
    if args.trajectory_schedule is not None and not args.trajectory_tracking:
        raise ValueError("--trajectory-schedule requires --trajectory-tracking")
    if args.action_offset_ms and (args.trajectory_tracking or args.phase_locked):
        raise ValueError("--action-offset-ms is only valid for open-loop replay")


def _prepare_replay(args: argparse.Namespace) -> _PreparedReplay:
    spec = RunSpec.from_yaml(args.config)
    evaluation = _require_evaluation(spec, "demo-benchmark")
    updates = _replay_evaluation_overrides(args)
    if updates:
        evaluation = evaluation.model_copy(update=updates)
        spec = spec.model_copy(update={"evaluation": evaluation})
    if evaluation.target_median_s is None:
        raise ValueError("demo-benchmark requires evaluation.target_median_s")
    settings = _replay_settings(spec, args)
    spec, demonstration = _align_replay_interval(spec, settings, args)
    return _PreparedReplay(spec, evaluation, settings, demonstration)


def _replay_evaluation_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    candidates = (
        ("trials_per_map", args.trials),
        ("target_median_s", args.target_median),
        ("min_finish_rate", args.min_finish_rate),
    )
    return {key: value for key, value in candidates if value is not None}


def _replay_settings(spec: RunSpec, args: argparse.Namespace) -> _ReplaySettings:
    component = spec.components.environment
    kwargs = {} if component is None else component.kwargs
    config = kwargs.get("config", {})
    open_loop = not args.trajectory_tracking and not args.phase_locked
    configured_interval = config.get("decision_interval_ms")
    interval_ms = (
        float(configured_interval) if open_loop and configured_interval is not None else None
    )
    configured_lead = float(config.get("demonstration_action_lead_ms", 0.0))
    action_lead_ms = configured_lead if args.action_lead_ms is None else float(args.action_lead_ms)
    aggregate = bool(open_loop and config.get("demonstration_control_aggregation", False))
    return _ReplaySettings(open_loop, interval_ms, action_lead_ms, aggregate)


def _align_replay_interval(
    spec: RunSpec, settings: _ReplaySettings, args: argparse.Namespace
) -> tuple[RunSpec, Any | None]:
    if args.phase_locked:
        return spec, None
    demonstration = load_demonstration(args.demo)
    interval_ms = settings.interval_ms or demonstration.decision_interval_ms
    if settings.aggregate_controls and interval_ms is None:
        raise ValueError("aggregated demo replay requires environment decision_interval_ms")
    requires_validation = settings.aggregate_controls or args.trajectory_tracking
    requires_validation |= interval_ms is None and demonstration.action_repeat_frames == 1
    if requires_validation:
        validate_recording_quality(demonstration)
        if not settings.aggregate_controls:
            interval_ms = demonstration_timing_summary(demonstration)["interval_median_ms"]
    alignment = _ReplayInterval(settings, args.trajectory_tracking, interval_ms)
    return _apply_replay_interval(spec, alignment), demonstration


def _apply_replay_interval(spec: RunSpec, alignment: _ReplayInterval) -> RunSpec:
    settings = alignment.settings
    interval_ms = alignment.interval_ms
    if interval_ms is None:
        return spec
    if not settings.open_loop or settings.interval_ms is None:
        spec = _with_environment_decision_interval(spec, interval_ms)
    mode = "Trajectory tracking" if alignment.trajectory_tracking else "Open-loop replay"
    print(f"{mode} interval: {interval_ms:.3f} ms")
    return spec


def _evaluate_replay(
    run: ResolvedRun, prepared: _PreparedReplay, args: argparse.Namespace
) -> tuple[Policy, list[dict[str, Any]], dict[str, float]]:
    evaluator = run.evaluator
    if not isinstance(evaluator, TrackmaniaEvaluator):
        raise ValueError("demo-benchmark requires components.evaluator")
    try:
        policy = _build_replay_policy(run, prepared, args)
        evaluator.set_checkpoint(args.demo)
        metrics = dict(evaluator.evaluate(policy))
        run.logger.log("eval/summary", metrics, step=0)
        artifact = _load_evaluation_artifact(run.run_dir)
    finally:
        run.logger.close()
    return policy, _artifact_trials(artifact), metrics


def _build_replay_policy(
    run: ResolvedRun, prepared: _PreparedReplay, args: argparse.Namespace
) -> Policy:
    config = getattr(run.environment_factory, "config", None)
    if not isinstance(config, TrackmaniaEnvironmentConfig):
        raise ValueError("demo-benchmark requires OpenPlanetEnvironmentFactory")
    context = _ReplayPolicyContext(run, prepared, args, config)
    if args.trajectory_tracking:
        return _trajectory_policy(args)
    if args.phase_locked:
        return _phase_locked_policy(context)
    return _open_loop_policy(context)


def _trajectory_policy(args: argparse.Namespace) -> Policy:
    tracker = _trajectory_tracker(args)
    if args.trajectory_schedule is not None:
        return build_scheduled_policy(
            args.demo, TrajectorySchedule.load(args.trajectory_schedule), tracker
        )
    return TrajectoryTrackingDemonstrationPolicy.from_path(
        TrajectoryTrackingPathRequest(args.demo, tracker.tracking_config())
    )


def _trajectory_tracker(args: argparse.Namespace) -> TrajectoryTrackerConfig:
    return TrajectoryTrackerConfig(
        action_lead_steps=args.action_lead_steps,
        action_lead_ms=args.action_lead_ms,
        lateral_gain=args.tracker_lateral_gain,
        heading_gain=args.tracker_heading_gain,
        lateral_velocity_gain=args.tracker_lateral_velocity_gain,
        steering_threshold=args.tracker_engage_threshold,
        steering_release_threshold=args.tracker_release_threshold,
        preview_ms=args.tracker_preview_ms,
        minimum_correction_steps=args.tracker_minimum_hold_steps,
        reversal_neutral_steps=args.tracker_reversal_neutral_steps,
    )


def _phase_locked_policy(context: _ReplayPolicyContext) -> Policy:
    run = context.run
    if not isinstance(run.feature_pipeline, LidarFeaturePipeline):
        raise ValueError("phase-locked demo-benchmark requires LidarFeaturePipeline")
    action_ids = context.config.compact_action_ids or tuple(range(78))
    request = PhaseLockedPathRequest(
        context.args.demo,
        run.feature_pipeline,
        tuple(action_ids),
        context.config.decision_interval_ms,
        context.prepared.settings.action_lead_ms,
    )
    return PhaseLockedDemonstrationPolicy.from_path(request)


def _open_loop_policy(context: _ReplayPolicyContext) -> Policy:
    prepared = context.prepared
    settings = prepared.settings
    policy = DemonstrationReplayPolicy.from_path(_open_loop_request(context))
    demonstration = prepared.demonstration
    if demonstration is None:
        raise RuntimeError("open-loop replay requires a loaded demonstration")
    _print_open_loop_timing(demonstration, settings, context.args.action_offset_ms)
    return policy


def _open_loop_request(context: _ReplayPolicyContext) -> ReplayPathRequest:
    settings = context.prepared.settings
    sampling = ReplaySamplingConfig(
        settings.interval_ms,
        settings.action_lead_ms,
        settings.aggregate_controls,
    )
    return ReplayPathRequest(
        context.args.demo,
        context.config.compact_action_ids,
        context.args.action_offset_ms,
        sampling,
    )


def _print_open_loop_timing(
    demonstration: Any, settings: _ReplaySettings, action_offset_ms: float
) -> None:
    print(
        "Open-loop action timing: "
        f"alignment={demonstration.control_alignment}, "
        f"offset={action_offset_ms:+.1f} ms, "
        f"lead={settings.action_lead_ms:+.1f} ms, "
        f"aggregate_controls={settings.aggregate_controls}"
    )


def _apply_replay_gate(gate: _ReplayGate) -> None:
    trials = gate.trials
    evaluation = gate.evaluation
    completed = [trial for trial in trials if trial["finished"]]
    required = ceil(evaluation.min_finish_rate * len(trials))
    median = float(gate.metrics["eval/median_finish_time_s"])
    failed = len(completed) < required or median >= _target_median(evaluation)
    failed |= _has_runtime_errors(trials)
    if failed and gate.report_only:
        print("Demonstration replay gate failed; --report-only keeps the diagnostic run successful")
        return
    if failed:
        raise RuntimeError(_replay_failure(required, len(trials), evaluation))
    print(
        f"Demonstration replay passed: {len(completed)}/{len(trials)} finishes, "
        f"median {median:.3f}s"
    )


def _replay_failure(required: int, trials: int, evaluation: EvaluationSuiteSpec) -> str:
    return (
        "demonstration replay failed: require "
        f">={required}/{trials} finishes, "
        f"median completed time <{_target_median(evaluation)}s, "
        "and no telemetry/controller errors"
    )


def _print_trajectory_tracker_diagnostics(
    policy: TrajectoryTrackingDemonstrationPolicy,
) -> None:
    print(
        "Trajectory tracker diagnostics: "
        f"reference_index={policy.reference_index}, "
        f"correction_events={policy.correction_count}, "
        f"correction_steps={policy.correction_step_count}, "
        f"neutralized_expert_steps={policy.neutralized_expert_step_count}, "
        f"output_steering_switches={policy.output_switch_count}, "
        f"expert_steering_switches={policy.expert_steering_switch_count}, "
        f"opposing_reversals={policy.opposing_switch_count}, "
        f"max_error(position={policy.max_position_error_m:.3f}m, "
        f"lateral={policy.max_abs_lateral_error_m:.3f}m, "
        f"heading={policy.max_abs_heading_error:.4f}, "
        f"lateral_velocity={policy.max_abs_lateral_velocity_error_mps:.3f}m/s)"
    )
