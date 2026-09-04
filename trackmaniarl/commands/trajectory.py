"""Command-line entrypoint for the current TrackmaniaRL project workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from trackmaniarl.commands.assets import _trackmania_factory
from trackmaniarl.commands.helpers import _compact_action_ids, _recovery_contract
from trackmaniarl.core.spec import EvaluationMapSpec, RunSpec
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    demonstration_timing_summary,
    load_demonstration,
    save_demonstration,
    validate_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.synthetic_recovery import (
    SyntheticRecoveryPathRequest,
    generate_synthetic_recovery_from_path,
)
from trackmaniarl.trackmania.synthetic_recovery_types import (
    SyntheticRecoveryConfig,
    SyntheticRecoveryDataset,
)
from trackmaniarl.trackmania.trajectory_optimization import (
    SafeTrajectoryOptimizer,
    TrajectorySchedule,
    TrajectorySearchConfig,
    TrajectorySearchOutcome,
    TrajectorySearchResult,
    TrajectoryTrackerConfig,
    build_scheduled_policy,
    run_trajectory_trial,
)
from trackmaniarl.trackmania.trajectory_stitching import (
    TrajectoryJoin,
    build_fastest_compatible_trajectory,
)


def _trajectory_stitch(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    factory = _trackmania_factory(config_path)
    geometry = _trajectory_geometry(factory, "trajectory-stitch")
    result = build_fastest_compatible_trajectory(args.demo, geometry)
    output = save_demonstration(args.output.resolve(), result.demonstration)
    print(
        f"Trajectory stitch complete: estimated={result.demonstration.finish_time_s:.3f}s, "
        f"gain={result.estimated_gain_s:.3f}s, joins={len(result.joins)}, output={output}"
    )
    _print_trajectory_joins(result.joins)


def _print_trajectory_joins(joins: tuple[TrajectoryJoin, ...]) -> None:
    for join in joins:
        print(
            f"  join={join.progress_fraction * 100.0:.3f}% "
            f"position_gap={join.position_gap_m:.3f}m "
            f"velocity_gap={join.velocity_gap_mps:.3f}m/s "
            f"heading_gap={join.heading_gap_degrees:.3f}deg"
        )


def _trajectory_geometry(factory: OpenPlanetEnvironmentFactory, command: str) -> BoundaryGeometry:
    path = factory.config.geometry_path
    if path is None:
        raise ValueError(f"{command} requires environment.geometry_path")
    return BoundaryGeometry(path, expected_map_uid=factory.config.expected_map_uid)


def _trajectory_synthetic_recovery(args: argparse.Namespace) -> None:
    context = _synthetic_recovery_context(args)
    dataset = _generate_synthetic_recovery(args, context)
    output = dataset.save(args.output.resolve())
    print(
        f"Synthetic trajectory recovery: samples={len(dataset.frames)}, "
        f"interventions={int(dataset.interventions.sum())}, output={output}"
    )


@dataclass(frozen=True, slots=True)
class _SyntheticRecoveryContext:
    spec: RunSpec
    environment: Any
    geometry: BoundaryGeometry


def _synthetic_recovery_context(args: argparse.Namespace) -> _SyntheticRecoveryContext:
    config_path = args.config.resolve()
    spec = RunSpec.from_yaml(config_path)
    factory = _trackmania_factory(config_path)
    geometry = _trajectory_geometry(factory, "trajectory-synthetic-recovery")
    demonstration = load_demonstration(args.demo)
    validate_recording_quality(demonstration)
    validate_demonstration(demonstration, factory.config, geometry)
    return _SyntheticRecoveryContext(spec, factory.config, geometry)


def _generate_synthetic_recovery(
    args: argparse.Namespace, context: _SyntheticRecoveryContext
) -> SyntheticRecoveryDataset:
    request = SyntheticRecoveryPathRequest(
        args.demo,
        _compact_action_ids(context.spec),
        SyntheticRecoveryConfig(
            sample_stride=args.sample_stride,
            action_lead_ms=args.action_lead_ms,
        ),
        _recovery_contract(context.environment, context.geometry),
        context.environment.demonstration_control_aggregation,
    )
    return generate_synthetic_recovery_from_path(request)


@dataclass(frozen=True, slots=True)
class _TrajectoryEnvironmentRequest:
    spec: RunSpec
    config_path: Path
    demonstration: Demonstration
    seed: int | None


@dataclass(frozen=True, slots=True)
class _TrajectoryOptimization:
    spec: RunSpec
    demonstration_path: Path
    output: Path
    schedule: TrajectorySchedule
    tracker: TrajectoryTrackerConfig
    search: TrajectorySearchConfig
    environment: Any


@dataclass(slots=True)
class _TrajectoryEvaluator:
    optimization: _TrajectoryOptimization
    trial: int = 0

    def __call__(self, candidate: TrajectorySchedule) -> TrajectorySearchOutcome:
        self.trial += 1
        policy = build_scheduled_policy(
            self.optimization.demonstration_path, candidate, self.optimization.tracker
        )
        outcome = run_trajectory_trial(
            self.optimization.environment,
            policy,
            self.optimization.spec.training.max_episode_steps,
        )
        _print_trajectory_trial(self.trial, self.optimization.search.max_trials, outcome)
        return outcome


def _trajectory_optimize(args: argparse.Namespace) -> None:
    optimization = _trajectory_optimization(args)
    evaluator = _TrajectoryEvaluator(optimization)
    try:
        result = SafeTrajectoryOptimizer(optimization.search).optimize(
            optimization.schedule, evaluator
        )
    finally:
        optimization.environment.close()
    result.schedule.save(optimization.output)
    _print_trajectory_result(result, evaluator.trial, optimization)


def _trajectory_optimization(args: argparse.Namespace) -> _TrajectoryOptimization:
    config_path = args.config.resolve()
    demonstration = load_demonstration(args.demo)
    validate_recording_quality(demonstration)
    interval_ms = _demonstration_interval_ms(demonstration)
    spec = _with_environment_decision_interval(RunSpec.from_yaml(config_path), interval_ms)
    output = _npz_path(args.output.resolve())
    schedule = (
        TrajectorySchedule.load(output)
        if output.exists()
        else TrajectorySchedule.from_controls(demonstration.controls)
    )
    tracker = TrajectoryTrackerConfig(action_lead_ms=args.action_lead_ms)
    search = _trajectory_search_config(args, output, interval_ms)
    build_scheduled_policy(args.demo, schedule, tracker)
    request = _TrajectoryEnvironmentRequest(spec, config_path, demonstration, args.seed)
    environment = _trajectory_search_environment(request)
    return _TrajectoryOptimization(spec, args.demo, output, schedule, tracker, search, environment)


def _print_trajectory_trial(trial: int, max_trials: int, outcome: TrajectorySearchOutcome) -> None:
    time_text = f"{outcome.finish_time_s:.3f}s" if outcome.finish_time_s else "-"
    print(
        f"Trajectory search trial={trial}/{max_trials} "
        f"finished={outcome.finished} time={time_text} "
        f"progress={outcome.progress_pct:.1f}% error={outcome.error or '-'}"
    )


def _print_trajectory_result(
    result: TrajectorySearchResult, trials: int, context: _TrajectoryOptimization
) -> None:
    print(
        f"Trajectory optimization complete: median={result.median_finish_time_s:.3f}s, "
        f"live_trials={trials}, schedule={context.output}"
    )
    if result.median_finish_time_s > context.search.target_time_s:
        raise RuntimeError(
            f"trajectory target not reached: best confirmed median "
            f"{result.median_finish_time_s:.3f}s > {context.search.target_time_s:.3f}s"
        )


def _demonstration_interval_ms(demonstration: Demonstration) -> float:
    interval_ms = demonstration.decision_interval_ms
    if interval_ms is None:
        interval_ms = demonstration_timing_summary(demonstration)["interval_median_ms"]
    if interval_ms <= 0.0:
        raise ValueError("trajectory optimization requires a positive demonstration cadence")
    return float(interval_ms)


def _trajectory_search_environment(request: _TrajectoryEnvironmentRequest) -> Any:
    spec = request.spec
    evaluation = spec.evaluation
    if evaluation is None or len(evaluation.maps) != 1:
        raise ValueError("trajectory-optimize requires exactly one configured evaluation map")
    component = spec.components.environment
    if component is None or component.class_path != (
        "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
    ):
        raise ValueError("trajectory-optimize requires OpenPlanetEnvironmentFactory")
    factory = OpenPlanetEnvironmentFactory(base_dir=request.config_path.parent, **component.kwargs)
    resolved_map = _trajectory_evaluation_map(request)
    geometry = BoundaryGeometry(
        resolved_map.geometry_path, expected_map_uid=resolved_map.expected_map_uid
    )
    validate_demonstration(request.demonstration, factory.config, geometry)
    seed = spec.seed if request.seed is None else request.seed
    return factory.create(seed=seed, evaluation_map=resolved_map)


def _trajectory_evaluation_map(request: _TrajectoryEnvironmentRequest) -> EvaluationMapSpec:
    evaluation = request.spec.evaluation
    if evaluation is None:
        raise ValueError("trajectory-optimize requires an evaluation suite")
    map_spec = evaluation.maps[0]
    return map_spec.model_copy(
        update={
            "map_path": _relative_to(request.config_path.parent, map_spec.map_path),
            "geometry_path": _relative_to(request.config_path.parent, map_spec.geometry_path),
        }
    )


def _trajectory_search_config(
    args: argparse.Namespace,
    output: Path,
    interval_ms: float,
) -> TrajectorySearchConfig:
    _validate_trajectory_search_arguments(args)
    shortening_ticks, minimum_window_ticks = _trajectory_search_ticks(args, interval_ms)
    return TrajectorySearchConfig(
        shortening_ticks=shortening_ticks,
        minimum_window_ticks=minimum_window_ticks,
        baseline_trials=args.baseline_trials,
        confirmation_trials=args.confirmation_trials,
        minimum_improvement_s=args.minimum_improvement_ms / 1_000.0,
        target_time_s=args.target_time,
        max_trials=args.max_trials,
        checkpoint_path=output,
        journal_path=output.with_suffix(".jsonl"),
    )


def _trajectory_search_ticks(
    args: argparse.Namespace, interval_ms: float
) -> tuple[tuple[int, ...], int]:
    shortening = tuple(
        dict.fromkeys(max(1, round(value / interval_ms)) for value in args.shortening_ms)
    )
    minimum_window = max(1, ceil(args.minimum_window_ms / interval_ms))
    return shortening, minimum_window


def _validate_trajectory_search_arguments(args: argparse.Namespace) -> None:
    if any(value <= 0.0 for value in args.shortening_ms):
        raise ValueError("--shortening-ms values must be positive")
    if args.minimum_window_ms <= 0.0:
        raise ValueError("--minimum-window-ms must be positive")
    if args.action_lead_ms < 0.0:
        raise ValueError("--action-lead-ms must be non-negative")


def _relative_to(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _npz_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".npz" else path.with_suffix(".npz")


def _with_environment_decision_interval(spec: RunSpec, interval_ms: float) -> RunSpec:
    environment = spec.components.environment
    if environment is None:
        raise ValueError("demo-benchmark requires components.environment")
    kwargs = dict(environment.kwargs)
    config = dict(kwargs.get("config", {}))
    config.update({"action_repeat_frames": 1, "decision_interval_ms": interval_ms})
    kwargs["config"] = config
    environment = environment.model_copy(update={"kwargs": kwargs})
    components = spec.components.model_copy(update={"environment": environment})
    return spec.model_copy(update={"components": components})
