"""DAgger recovery-data collection command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Any

import numpy as np

from trackmaniarl.commands.helpers import (
    _behavior_policy_state,
    _file_sha256,
    _learner_context,
    _recovery_contract,
)
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import EvaluationMapSpec, RunSpec
from trackmaniarl.trackmania.actions import (
    continuous_control_to_discrete_index,
    select_brake_tap_actions,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    load_demonstration,
    validate_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.guidance import TrajectoryTrackingDemonstrationPolicy
from trackmaniarl.trackmania.guidance_tracking import (
    TrajectoryTrackingConfig,
    TrajectoryTrackingPathRequest,
)
from trackmaniarl.trackmania.imitation_learning import (
    BehaviorCloningPolicy,
    RecoveryArrays,
    RecoveryMetadata,
    RecoveryProvenance,
    RecoverySaveRequest,
    save_behavior_cloning_recovery,
)


@dataclass(frozen=True, slots=True)
class _DaggerContext:
    run: ResolvedRun
    spec: RunSpec
    factory: OpenPlanetEnvironmentFactory
    pipeline: LidarFeaturePipeline
    environment: TrackmaniaEnvironmentConfig
    action_ids: tuple[int, ...]
    demonstration: Demonstration
    evaluation_map: EvaluationMapSpec


@dataclass(frozen=True, slots=True)
class _DaggerActors:
    student: BehaviorCloningPolicy
    teacher: TrajectoryTrackingDemonstrationPolicy
    action_table: list[np.ndarray]
    generator: np.random.Generator
    environment: Any


@dataclass(frozen=True, slots=True)
class _DaggerDecision:
    teacher_control: np.ndarray
    teacher_action: int
    student_action: int
    state_error: float
    intervention: bool
    disagreement: bool


@dataclass(frozen=True, slots=True)
class _DaggerRecord:
    frame: np.ndarray
    label: int
    episode_start: bool
    sample_weight: float
    student_action: int
    intervention: bool
    state_error: float


@dataclass(slots=True)
class _DaggerSamples:
    frames: list[np.ndarray] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    episode_starts: list[bool] = field(default_factory=list)
    sample_weights: list[float] = field(default_factory=list)
    student_actions: list[int] = field(default_factory=list)
    interventions: list[bool] = field(default_factory=list)
    state_errors: list[float] = field(default_factory=list)
    finished: int = 0

    def add(self, record: _DaggerRecord) -> None:
        self.frames.append(record.frame)
        self.labels.append(record.label)
        self.episode_starts.append(record.episode_start)
        self.sample_weights.append(record.sample_weight)
        self.student_actions.append(record.student_action)
        self.interventions.append(record.intervention)
        self.state_errors.append(record.state_error)


@dataclass(slots=True)
class _DaggerRun:
    context: _DaggerContext
    actors: _DaggerActors
    args: argparse.Namespace
    samples: _DaggerSamples = field(default_factory=_DaggerSamples)


@dataclass(frozen=True, slots=True)
class _DaggerStepState:
    raw: np.ndarray
    prepared: Any
    step: int


@dataclass(frozen=True, slots=True)
class _DaggerRecordInput:
    raw: np.ndarray
    step: int
    decision: _DaggerDecision
    intervention_error: float


def _dagger_collect(args: argparse.Namespace) -> None:
    _validate_dagger_arguments(args)
    config = args.config.resolve()
    source = RunSpec.from_yaml(config)
    spec = source.model_copy(update={"run_id": f"{source.run_id}-dagger-{time_ns()}"})
    run = resolve_run(spec, base_dir=config.parent)
    actors: _DaggerActors | None = None
    try:
        context = _dagger_context(run, spec, args)
        actors = _dagger_actors(context, args)
        dagger_run = _DaggerRun(context, actors, args)
        _collect_dagger_samples(dagger_run)
        output = _save_dagger_samples(dagger_run)
    finally:
        if actors is not None:
            actors.environment.close()
        run.logger.close()
    _print_dagger_summary(dagger_run, output)


def _print_dagger_summary(dagger_run: _DaggerRun, output: Path) -> None:
    print(
        f"DAgger recovery data: {output} ({len(dagger_run.samples.frames)} samples, "
        f"{dagger_run.samples.finished}/{dagger_run.args.episodes} finishes)"
    )


def _validate_dagger_arguments(args: argparse.Namespace) -> None:
    if args.episodes < 1 or not 0.0 <= args.teacher_probability <= 1.0:
        raise ValueError("DAgger episodes and teacher probability are invalid")
    if args.intervention_error <= 0.0:
        raise ValueError("DAgger intervention error must be positive")
    if args.action_lead_ms < 0.0:
        raise ValueError("DAgger action lead must be non-negative")


def _dagger_context(run: ResolvedRun, spec: RunSpec, args: argparse.Namespace) -> _DaggerContext:
    evaluation_map = _dagger_evaluation_map(run, spec)
    factory, pipeline = _dagger_components(run)
    action_ids = _dagger_action_ids(factory)
    demonstration = load_demonstration(args.demo)
    validate_recording_quality(demonstration)
    validate_demonstration(demonstration, factory.config, pipeline.geometry)
    return _DaggerContext(
        run,
        spec,
        factory,
        pipeline,
        factory.config,
        action_ids,
        demonstration,
        evaluation_map,
    )


def _dagger_evaluation_map(run: ResolvedRun, spec: RunSpec) -> EvaluationMapSpec:
    evaluation = spec.evaluation
    if run.evaluator is None or evaluation is None or not evaluation.maps:
        raise ValueError("dagger-collect requires a configured evaluation map")
    return evaluation.maps[0]


def _dagger_components(
    run: ResolvedRun,
) -> tuple[OpenPlanetEnvironmentFactory, LidarFeaturePipeline]:
    factory = run.environment_factory
    pipeline = run.feature_pipeline
    if not isinstance(factory, OpenPlanetEnvironmentFactory):
        raise ValueError("dagger-collect requires OpenPlanetEnvironmentFactory")
    if not isinstance(pipeline, LidarFeaturePipeline):
        raise ValueError("dagger-collect requires LidarFeaturePipeline")
    return factory, pipeline


def _dagger_action_ids(factory: OpenPlanetEnvironmentFactory) -> tuple[int, ...]:
    action_ids = factory.config.compact_action_ids
    if action_ids is None:
        raise ValueError("dagger-collect requires compact_action_ids")
    return action_ids


def _dagger_actors(context: _DaggerContext, args: argparse.Namespace) -> _DaggerActors:
    run = context.run
    run.learner.setup(_learner_context(run))
    student = _load_dagger_student(context, args.checkpoint)
    teacher = TrajectoryTrackingDemonstrationPolicy.from_path(
        TrajectoryTrackingPathRequest(
            args.demo,
            TrajectoryTrackingConfig(action_lead_steps=0, action_lead_ms=args.action_lead_ms),
        )
    )
    _, action_table = select_brake_tap_actions(context.action_ids)
    environment = context.factory.create(
        seed=context.spec.seed, evaluation_map=context.evaluation_map
    )
    generator = np.random.default_rng(context.spec.seed)
    return _DaggerActors(student, teacher, action_table, generator, environment)


def _load_dagger_student(context: _DaggerContext, checkpoint_path: Path) -> BehaviorCloningPolicy:
    run = context.run
    model = getattr(run.learner, "model", None)
    if model is None or bool(model.previous_action_conditioning):
        raise ValueError("dagger-collect requires BC without previous-action conditioning")
    checkpoint = run.checkpoint_codec.load(checkpoint_path)
    run.learner.load_state_dict(_behavior_policy_state(checkpoint))
    student = run.learner.policy()
    if not isinstance(student, BehaviorCloningPolicy):
        raise ValueError("dagger-collect requires BehaviorCloningPolicy")
    return student


def _collect_dagger_samples(dagger_run: _DaggerRun) -> None:
    for episode in range(dagger_run.args.episodes):
        finished = _collect_dagger_episode(dagger_run, episode)
        dagger_run.samples.finished += int(finished)
        print(
            f"DAgger episode {episode + 1}/{dagger_run.args.episodes}: "
            f"finished={finished}, samples={len(dagger_run.samples.frames)}"
        )


def _collect_dagger_episode(dagger_run: _DaggerRun, episode: int) -> bool:
    context = dagger_run.context
    actors = dagger_run.actors
    raw, _ = actors.environment.reset(seed=context.spec.seed + episode)
    context.pipeline.reset_episode()
    actors.student.reset_episode()
    actors.teacher.reset_episode()
    prepared = context.pipeline.transform_observation(raw)
    for step in range(context.spec.training.max_episode_steps):
        state = _DaggerStepState(raw, prepared, step)
        raw, prepared, finished = _collect_dagger_step(dagger_run, state)
        if finished is not None:
            return finished
    return False


def _collect_dagger_step(
    dagger_run: _DaggerRun, state: _DaggerStepState
) -> tuple[np.ndarray, Any, bool | None]:
    decision = _dagger_decision(dagger_run, state.raw, state.prepared)
    record_input = _DaggerRecordInput(
        state.raw, state.step, decision, dagger_run.args.intervention_error
    )
    record = _dagger_record(record_input)
    dagger_run.samples.add(record)
    action = decision.teacher_control if decision.intervention else decision.student_action
    raw, _, terminated, truncated, info = dagger_run.actors.environment.step(action)
    prepared = dagger_run.context.pipeline.transform_observation(raw)
    finished = bool(info["termination_reason"] == "finished") if terminated or truncated else None
    return raw, prepared, finished


def _dagger_decision(dagger_run: _DaggerRun, raw: np.ndarray, prepared: Any) -> _DaggerDecision:
    actors = dagger_run.actors
    teacher_control = actors.teacher.act(raw, mode=PolicyMode.EVALUATION)
    teacher_action = continuous_control_to_discrete_index(teacher_control, actors.action_table)
    student_action = actors.student.act(prepared, mode=PolicyMode.EVALUATION)
    state_error = _trajectory_teacher_state_error(actors.teacher)
    intervene = state_error >= dagger_run.args.intervention_error
    intervene |= actors.generator.random() < dagger_run.args.teacher_probability
    return _DaggerDecision(
        teacher_control,
        teacher_action,
        student_action,
        state_error,
        intervene,
        student_action != teacher_action,
    )


def _dagger_record(record: _DaggerRecordInput) -> _DaggerRecord:
    decision = record.decision
    return _DaggerRecord(
        np.asarray(record.raw, dtype=np.float32).copy(),
        decision.teacher_action,
        record.step == 0,
        _dagger_sample_weight(decision, record.intervention_error),
        decision.student_action,
        decision.intervention,
        decision.state_error,
    )


def _save_dagger_samples(dagger_run: _DaggerRun) -> Path:
    return save_behavior_cloning_recovery(_dagger_recovery_request(dagger_run))


def _dagger_recovery_request(dagger_run: _DaggerRun) -> RecoverySaveRequest:
    samples = dagger_run.samples
    context = dagger_run.context
    arrays = _dagger_recovery_arrays(samples, context.action_ids)
    metadata = _dagger_recovery_metadata(samples)
    return RecoverySaveRequest(
        dagger_run.args.output,
        arrays,
        _dagger_provenance(dagger_run),
        metadata,
    )


def _dagger_recovery_arrays(samples: _DaggerSamples, action_ids: tuple[int, ...]) -> RecoveryArrays:
    return RecoveryArrays(
        np.asarray(samples.frames, dtype=np.float32),
        np.asarray(samples.labels, dtype=np.int64),
        np.asarray(samples.episode_starts, dtype=np.bool_),
        action_ids,
    )


def _dagger_recovery_metadata(samples: _DaggerSamples) -> RecoveryMetadata:
    return RecoveryMetadata(
        np.asarray(samples.sample_weights, dtype=np.float32),
        np.asarray(samples.student_actions, dtype=np.int64),
        np.asarray(samples.interventions, dtype=np.bool_),
        np.asarray(samples.state_errors, dtype=np.float32),
    )


def _dagger_provenance(dagger_run: _DaggerRun) -> RecoveryProvenance:
    context = dagger_run.context
    return RecoveryProvenance.from_demonstration(
        context.demonstration,
        contract=_recovery_contract(context.environment, context.pipeline.geometry),
        source_checkpoint_sha256=_file_sha256(Path(dagger_run.args.checkpoint)),
    )


def _trajectory_teacher_state_error(
    teacher: TrajectoryTrackingDemonstrationPolicy,
) -> float:
    return max(
        teacher.last_position_error_m,
        8.0 * abs(teacher.last_heading_error),
        0.5 * abs(teacher.last_lateral_velocity_error_mps),
    )


def _dagger_sample_weight(decision: _DaggerDecision, intervention_error: float) -> float:
    relative_error = np.clip(decision.state_error / intervention_error, 0.0, 2.0)
    return float(
        np.clip(
            0.25 + 1.75 * decision.disagreement + 2.0 * decision.intervention + relative_error,
            0.25,
            6.0,
        )
    )
