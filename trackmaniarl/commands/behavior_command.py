"""Behavior-cloning training and rollout commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import time_ns
from typing import Any

from trackmaniarl.commands.behavior_training import _train_behavior_cloning
from trackmaniarl.commands.behavior_types import _BehaviorTrainingRequest
from trackmaniarl.commands.common import _new_attempt_spec, _resumed_attempt_spec
from trackmaniarl.commands.evaluation import (
    _artifact_trials,
    _load_evaluation_artifact,
    _print_benchmark_report,
)
from trackmaniarl.commands.helpers import (
    _behavior_policy_state,
    _compact_action_ids,
    _file_sha256,
    _learner_context,
    _recovery_contract,
)
from trackmaniarl.core.runtime import ResolvedRun, prepare_run, resolve_run
from trackmaniarl.core.spec import EvaluationSuiteSpec, RunSpec
from trackmaniarl.trackmania.demonstrations import load_demonstration, resolve_demonstration_paths
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.imitation_learning import (
    BehaviorCloningLap,
    LapLoadRequest,
    RecoveryLoadRequest,
    RecoveryProvenance,
    augment_behavior_cloning_laps,
    load_behavior_cloning_laps,
    load_behavior_cloning_recovery,
    split_behavior_cloning_laps,
)


@dataclass(frozen=True, slots=True)
class _BehaviorContext:
    run: ResolvedRun
    spec: RunSpec
    paths: tuple[Path, ...]
    action_ids: tuple[int, ...]
    model: Any
    environment: TrackmaniaEnvironmentConfig


@dataclass(frozen=True, slots=True)
class _BehaviorSplit:
    training: list[BehaviorCloningLap]
    validation: list[BehaviorCloningLap]
    recovery_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _DatasetManifest:
    run: ResolvedRun
    paths: tuple[Path, ...]
    training: list[BehaviorCloningLap]
    validation: list[BehaviorCloningLap]
    action_ids: tuple[int, ...]


def _bc_train(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    spec = RunSpec.from_yaml(config)
    spec = _new_attempt_spec(config, _resumed_attempt_spec(config, spec, args), args)
    paths = tuple(resolve_demonstration_paths(args.demo))
    run = resolve_run(spec, base_dir=config.parent)
    try:
        context = _behavior_context(run, spec, paths)
        _run_bc_training(context, args)
    finally:
        run.logger.close()


def _run_bc_training(context: _BehaviorContext, args: argparse.Namespace) -> None:
    split = _behavior_split(context, args)
    manifest = _DatasetManifest(
        context.run,
        (*context.paths, *split.recovery_paths),
        split.training,
        split.validation,
        context.action_ids,
    )
    fingerprint = _write_behavior_cloning_dataset_manifest(manifest)
    _bind_behavior_dataset(context, fingerprint)
    resume = getattr(args, "resume", None)
    state = None if resume is None else context.run.checkpoint_codec.load(resume)
    request = _BehaviorTrainingRequest(context.run, split.training, split.validation, state)
    _train_behavior_cloning(request)


def _behavior_context(run: ResolvedRun, spec: RunSpec, paths: tuple[Path, ...]) -> _BehaviorContext:
    prepare_run(run)
    run.learner.setup(_learner_context(run))
    model = getattr(run.learner, "model", None)
    action_ids = _compact_action_ids(spec)
    if model is None or tuple(model.action_ids) != action_ids:
        raise ValueError("model action_ids must exactly match environment compact_action_ids")
    _validate_behavior_features(run)
    factory = run.environment_factory
    if not isinstance(factory, OpenPlanetEnvironmentFactory):
        raise ValueError("behavior cloning requires OpenPlanetEnvironmentFactory")
    return _BehaviorContext(run, spec, paths, action_ids, model, factory.config)


def _validate_behavior_features(run: ResolvedRun) -> None:
    pipeline = run.feature_pipeline
    include_controls = bool(getattr(pipeline, "include_control_inputs", True))
    mask_controls = bool(getattr(pipeline, "mask_current_control_inputs", False))
    if include_controls and not mask_controls:
        raise ValueError(
            "behavior cloning control inputs require mask_current_control_inputs=true "
            "to prevent target leakage"
        )


def _behavior_split(context: _BehaviorContext, args: argparse.Namespace) -> _BehaviorSplit:
    laps = _load_behavior_laps(context)
    training, validation = split_behavior_cloning_laps(laps, context.spec.seed)
    split = _with_recovery_laps(context, args, _BehaviorSplit(training, validation))
    use_flip = bool(
        getattr(args, "horizontal_flip_augmentation", False)
        or getattr(context.run.learner, "horizontal_flip_augmentation", False)
    )
    if not use_flip:
        return split
    if not getattr(context.run.feature_pipeline, "local_velocity_features", False):
        raise ValueError("horizontal flip augmentation requires local_velocity_features")
    augmented = augment_behavior_cloning_laps(split.training, context.action_ids)
    return _BehaviorSplit(augmented, split.validation, split.recovery_paths)


def _load_behavior_laps(context: _BehaviorContext) -> list[BehaviorCloningLap]:
    environment = context.environment
    interval = environment.decision_interval_ms
    request = LapLoadRequest(
        paths=context.paths,
        pipeline=context.run.feature_pipeline,
        action_ids=context.action_ids,
        expected_action_repeat_frames=environment.action_repeat_frames,
        expected_decision_interval_ms=None if interval is None else float(interval),
        action_lead_ms=environment.demonstration_action_lead_ms,
        aggregate_controls=environment.demonstration_control_aggregation,
        previous_action_conditioning=bool(context.model.previous_action_conditioning),
    )
    return load_behavior_cloning_laps(request)


def _with_recovery_laps(
    context: _BehaviorContext, args: argparse.Namespace, split: _BehaviorSplit
) -> _BehaviorSplit:
    paths = tuple(Path(path).resolve() for path in getattr(args, "recovery", ()))
    if not paths:
        return split
    geometry = getattr(context.run.feature_pipeline, "geometry", None)
    if not isinstance(geometry, BoundaryGeometry):
        raise ValueError("behavior-cloning recovery requires a feature geometry")
    recovery = _load_recovery_laps(context, paths, geometry)
    recovery_training, recovery_validation = _split_recovery(recovery, context.spec.seed)
    split.training.extend(recovery_training)
    split.validation.extend(recovery_validation)
    return _BehaviorSplit(split.training, split.validation, paths)


def _load_recovery_laps(
    context: _BehaviorContext, paths: tuple[Path, ...], geometry: BoundaryGeometry
) -> list[BehaviorCloningLap]:
    request = RecoveryLoadRequest(
        paths=paths,
        pipeline=context.run.feature_pipeline,
        action_ids=context.action_ids,
        expected_contract=_recovery_contract(context.environment, geometry),
        expected_source_demonstration_sha256=_source_demonstration_hashes(context.paths),
        previous_action_conditioning=bool(context.model.previous_action_conditioning),
    )
    return load_behavior_cloning_recovery(request)


def _source_demonstration_hashes(paths: tuple[Path, ...]) -> frozenset[str]:
    return frozenset(
        RecoveryProvenance.from_demonstration(load_demonstration(path)).source_demonstration_sha256
        for path in paths
    )


def _split_recovery(
    laps: list[BehaviorCloningLap], seed: int
) -> tuple[list[BehaviorCloningLap], list[BehaviorCloningLap]]:
    if len(laps) < 3:
        return laps, []
    return split_behavior_cloning_laps(laps, seed + 1)


def _bind_behavior_dataset(context: _BehaviorContext, fingerprint: str) -> None:
    bind_dataset = getattr(context.run.learner, "bind_dataset", None)
    if not callable(bind_dataset):
        raise TypeError("bc-train learner must expose bind_dataset()")
    bind_dataset(fingerprint)


def _write_behavior_cloning_dataset_manifest(manifest: _DatasetManifest) -> str:
    contract = _behavior_dataset_contract(manifest)
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    candidate = json.dumps({**contract, "fingerprint": fingerprint}, indent=2, sort_keys=True)
    target = manifest.run.run_dir / "bc-dataset-manifest.json"
    if target.exists() and target.read_text(encoding="utf-8") != candidate:
        raise ValueError("BC dataset or split differs from the immutable run manifest")
    target.write_text(candidate, encoding="utf-8")
    return fingerprint


def _behavior_dataset_contract(manifest: _DatasetManifest) -> dict[str, Any]:
    model_factory = manifest.run.spec.components.model_factory
    if model_factory is None:
        raise ValueError("behavior cloning requires components.model_factory")
    return {
        "schema_version": "trackmaniarl-bc-dataset-v2",
        "files": _behavior_manifest_files(manifest.paths),
        "training_sources": [lap.source_id for lap in manifest.training],
        "validation_sources": [lap.source_id for lap in manifest.validation],
        "action_ids": manifest.action_ids,
        "feature_pipeline": manifest.run.spec.components.feature_pipeline.model_dump(mode="json"),
        "model_factory": model_factory.model_dump(mode="json"),
    }


def _behavior_manifest_files(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _bc_benchmark(args: argparse.Namespace) -> None:
    spec, suite = _bc_benchmark_spec(args)
    run = resolve_run(spec, base_dir=args.config.parent)
    trials, metrics = _evaluate_bc_checkpoint(run, args.checkpoint)
    _print_benchmark_report(trials, metrics)
    passed = _print_bc_rollout_gate(trials, metrics, suite)
    if not passed and getattr(args, "report_only", False):
        print("BC rollout gate failed; --report-only keeps the diagnostic run successful")
        return
    if not passed:
        raise RuntimeError("behavior-cloning rollout failed the configured release gate")


def _bc_benchmark_spec(args: argparse.Namespace) -> tuple[RunSpec, EvaluationSuiteSpec]:
    if args.trials < 1:
        raise ValueError("bc-benchmark --trials must be positive")
    hold_steps = getattr(args, "minimum_action_hold_steps", None)
    if hold_steps is not None and hold_steps < 1:
        raise ValueError("bc-benchmark --minimum-action-hold-steps must be positive")
    spec = RunSpec.from_yaml(args.config)
    if hold_steps is not None:
        spec = _with_action_hold(spec, hold_steps)
    suite = spec.evaluation
    if suite is None or not suite.maps:
        raise ValueError("bc-benchmark requires an evaluation suite with at least one map")
    suite = suite.model_copy(update={"trials_per_map": args.trials})
    run_id = f"{spec.run_id}-bc-eval-{time_ns()}"
    return spec.model_copy(update={"run_id": run_id, "evaluation": suite}), suite


def _with_action_hold(spec: RunSpec, minimum_steps: int) -> RunSpec:
    factory = spec.components.model_factory
    if factory is None:
        raise ValueError("bc-benchmark action hold override requires components.model_factory")
    kwargs = {**factory.kwargs, "minimum_action_hold_steps": minimum_steps}
    factory = factory.model_copy(update={"kwargs": kwargs})
    components = spec.components.model_copy(update={"model_factory": factory})
    return spec.model_copy(update={"components": components})


def _evaluate_bc_checkpoint(
    run: ResolvedRun, checkpoint_path: Path
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if run.evaluator is None:
        raise ValueError("bc-benchmark requires components.evaluator")
    try:
        run.learner.setup(_learner_context(run))
        checkpoint = run.checkpoint_codec.load(checkpoint_path)
        run.learner.load_state_dict(_behavior_policy_state(checkpoint))
        set_checkpoint = getattr(run.evaluator, "set_checkpoint", None)
        if callable(set_checkpoint):
            set_checkpoint(checkpoint_path)
        metrics = dict(run.evaluator.evaluate(run.learner.policy()))
        run.logger.log("eval/summary", metrics, step=0)
        artifact = _load_evaluation_artifact(run.run_dir)
    finally:
        run.logger.close()
    return _artifact_trials(artifact), metrics


def _print_bc_rollout_gate(
    trials: list[dict[str, Any]], metrics: dict[str, float], suite: EvaluationSuiteSpec
) -> bool:
    completed = [trial for trial in trials if trial["finished"]]
    target = suite.target_median_s
    if target is None:
        print("BC rollout gate: no target_median_s configured")
        return False
    required = ceil(suite.min_finish_rate * len(trials))
    faster = [trial for trial in completed if float(trial["finish_time_s"]) < target]
    median = float(metrics["eval/median_finish_time_s"])
    go = len(completed) >= required and bool(faster)
    full_success = go and median < target
    print(
        f"BC rollout gate: go={go}, full_success={full_success}, "
        f"finishes={len(completed)}/{len(trials)}, under_target={len(faster)}, "
        f"target={target:.3f}s, median={median:.3f}s"
    )
    return full_success
