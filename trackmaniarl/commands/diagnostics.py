"""Command-line entrypoint for the current TrackmaniaRL project workflow."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any, cast

import torch

from trackmaniarl.commands.helpers import _learner_context, _training_learner_state
from trackmaniarl.core.pytree import sanitize_finite, tree_map, tree_to_device
from trackmaniarl.core.runtime import prepare_run, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    load_demonstration,
    resolve_demonstration_paths,
    validate_demonstration,
)
from trackmaniarl.trackmania.diagnostics import (
    ExpertActionDiagnostics,
    ExpertDiagnosticRecord,
    aggregate_expert_bins,
)
from trackmaniarl.trackmania.environment import (
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.pace import PaceDemonstrationRequest, ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_types import TransitionInput


@dataclass(frozen=True, slots=True)
class _ExpertContext:
    learner: Any
    pipeline: Any
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry


@dataclass(frozen=True, slots=True)
class _ExpertStepContext:
    expert: _ExpertContext
    model: Any
    reward: TrajectoryReward
    diagnostics: ExpertActionDiagnostics


def _diagnose_expert(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    source_spec = RunSpec.from_yaml(config)
    spec = source_spec.model_copy(update={"run_id": f"{source_spec.run_id}-expert-{time_ns()}"})
    run = resolve_run(spec, base_dir=config.parent)
    try:
        context = _prepare_expert_context(run, args.checkpoint)
        paths = resolve_demonstration_paths(args.demo)
        reports = [_expert_demonstration_report(path, context) for path in paths]
        target = _write_expert_reports(run.run_dir, args.checkpoint, reports)
    finally:
        run.logger.close()
    print(f"Expert diagnostics: {target}")


def _prepare_expert_context(run: Any, checkpoint_path: Path) -> _ExpertContext:
    run.learner.setup(_learner_context(run))
    checkpoint = run.checkpoint_codec.load(checkpoint_path)
    run.learner.load_state_dict(_training_learner_state(checkpoint))
    config = _expert_environment_config(run)
    if config.geometry_path is None:
        raise ValueError("expert diagnostics require geometry_path")
    geometry = BoundaryGeometry(
        config.geometry_path,
        expected_map_uid=config.expected_map_uid,
    )
    prepare_run(run)
    return _ExpertContext(run.learner, run.feature_pipeline, config, geometry)


def _write_expert_reports(run_dir: Path, checkpoint: Path, reports: list[dict[str, Any]]) -> Path:
    payload = {
        "schema_version": "1",
        "checkpoint": str(checkpoint),
        "demos": reports,
        "summary": {
            "demonstrations": len(reports),
            "progress_bins": aggregate_expert_bins(report["progress_bins"] for report in reports),
        },
    }
    target = run_dir / "expert-diagnostics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target


def _expert_environment_config(run: Any) -> TrackmaniaEnvironmentConfig:
    environment_config = getattr(run.environment_factory, "config", None)
    if not isinstance(environment_config, TrackmaniaEnvironmentConfig):
        raise ValueError("expert diagnostics require OpenPlanetEnvironmentFactory")
    if environment_config.compact_action_ids is not None:
        raise ValueError("expert diagnostics require the canonical 78-action IQN head")
    if getattr(run.learner.model, "action_count", None) != 78:
        raise ValueError("expert diagnostics require the canonical 78-action IQN head")
    return environment_config


def _expert_demonstration_report(
    path: Path,
    context: _ExpertContext,
) -> dict[str, Any]:
    demonstration, diagnostics = _expert_diagnostics(path, context)
    return {
        "path": str(path),
        "finish_time_s": demonstration.finish_time_s,
        "progress_bins": diagnostics.summary(),
    }


def _expert_diagnostics(
    path: Path, context: _ExpertContext
) -> tuple[Demonstration, ExpertActionDiagnostics]:
    demonstration = load_demonstration(path)
    validate_demonstration(demonstration, context.config, context.geometry)
    model = context.learner.model
    if model is None:
        raise RuntimeError("expert diagnostics require an initialized IQN learner model")
    model.eval()
    _reset_expert_state(context, model)
    reward = _expert_reward(context, demonstration)
    diagnostics = ExpertActionDiagnostics()
    step_context = _ExpertStepContext(context, model, reward, diagnostics)
    _record_expert_steps(step_context, demonstration)
    return demonstration, diagnostics


def _reset_expert_state(context: _ExpertContext, model: Any) -> None:
    reset = getattr(model, "reset_policy_state", None)
    if callable(reset):
        reset()
    pipeline_reset = getattr(context.pipeline, "reset_episode", None)
    if callable(pipeline_reset):
        pipeline_reset()


def _expert_reward(context: _ExpertContext, demonstration: Any) -> TrajectoryReward:
    config = context.config
    geometry = context.geometry
    reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
    pace_profile = (
        ReferencePaceProfile.from_demonstration(
            PaceDemonstrationRequest(config.pace_reference_path, geometry, reference)
        )
        if config.pace_reference_path is not None
        else None
    )
    reward = TrajectoryReward(reference, config.reward_config(pace_profile))
    reward.reset(
        demonstration.frames[0, list(config.position_indices)],
        velocity=demonstration.frames[0, list(config.velocity_indices)],
        race_time_ms=float(demonstration.frames[0, 3]),
    )
    return reward


def _record_expert_steps(context: _ExpertStepContext, demonstration: Demonstration) -> None:
    transitions = zip(
        demonstration.actions, demonstration.frames[:-1], demonstration.frames[1:], strict=True
    )
    for transition in transitions:
        if _record_expert_step(context, transition):
            break


def _record_expert_step(context: _ExpertStepContext, transition: tuple[Any, Any, Any]) -> bool:
    action, frame, next_frame = transition
    observation = context.expert.pipeline.transform_observation(frame)
    q_values = _raw_q_values(context.expert, context.model, observation)
    source_action = int(action)
    if not 0 <= source_action < q_values.shape[-1]:
        raise ValueError("demonstration action is outside the raw IQN action head")
    expert_q = float(q_values[source_action])
    rank = int((q_values > expert_q).sum()) + 1
    result = _advance_expert_reward(context.expert.config, context.reward, next_frame)
    record = ExpertDiagnosticRecord(
        context.reward.progress_pct, expert_q, float(q_values.max()), rank
    )
    context.diagnostics.record(record)
    return bool(result.terminated)


def _advance_expert_reward(
    config: TrackmaniaEnvironmentConfig, reward: TrajectoryReward, frame: Any
) -> Any:
    transition = TransitionInput(
        frame[list(config.position_indices)],
        bool(frame[2]),
        frame[list(config.velocity_indices)],
        float(frame[3]),
        False,
        0.0,
    )
    return reward.step(transition)


def _raw_q_values(context: _ExpertContext, model: Any, observation: Any) -> torch.Tensor:
    prepare = getattr(model, "prepare_policy_observation", None)
    if callable(prepare):
        observation = prepare(observation)
    observation = tree_to_device(sanitize_finite(observation), context.learner.device)
    detector = getattr(model, "observation_is_single", None)
    single = bool(detector(observation)) if callable(detector) else observation.ndim == 1
    if single:
        observation = tree_map(
            lambda value: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value,
            observation,
        )
    with torch.inference_mode():
        values = model.q_values(observation, context.learner.evaluation_quantile_count)
    return cast(torch.Tensor, values).squeeze(0).float().cpu()
