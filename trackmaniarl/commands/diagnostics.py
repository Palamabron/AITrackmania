"""Command-line entrypoint for the current TrackmaniaRL project workflow."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import time_ns
from typing import Any

import torch

from trackmaniarl.commands.expert_progress import ExpertProgressReporter
from trackmaniarl.commands.helpers import _learner_context, _training_learner_state
from trackmaniarl.core.data import Transition
from trackmaniarl.core.pytree import sanitize_finite, tree_collate, tree_to_device
from trackmaniarl.core.runtime import prepare_run, record_run_attempt, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import ValuePhase
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationTransitionContext,
    demonstration_transitions,
    load_demonstration,
    resample_demonstration_for_environment,
    resolve_demonstration_paths,
    validate_demonstration,
)
from trackmaniarl.trackmania.diagnostics import (
    ExpertActionDiagnostics,
    ExpertDiagnosticRecord,
    aggregate_expert_actions,
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


@dataclass(slots=True)
class _ExpertStepContext:
    expert: _ExpertContext
    model: CompositeValueModel
    reward: TrajectoryReward
    diagnostics: ExpertActionDiagnostics
    policy_state: Any


@dataclass(frozen=True, slots=True)
class _ExpertEpisode:
    demonstration: Demonstration
    transitions: list[Transition]
    next_frames: Any


@dataclass(frozen=True, slots=True)
class _ExpertReportRun:
    context: _ExpertContext
    paths: tuple[Path, ...]
    reporter: ExpertProgressReporter


def _diagnose_expert(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    source_spec = RunSpec.from_yaml(config)
    spec = source_spec.model_copy(update={"run_id": f"{source_spec.run_id}-expert-{time_ns()}"})
    run = resolve_run(spec, base_dir=config.parent)
    try:
        payload, target = _run_expert_diagnostics(run, args)
    finally:
        run.logger.close()
    print(_expert_terminal_summary(payload))
    print(f"Expert diagnostics: {target}")


def _run_expert_diagnostics(run: Any, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    paths = tuple(resolve_demonstration_paths(args.demo))
    print(
        f"Expert diagnostics: loading checkpoint for {len(paths)} demonstration(s)...", flush=True
    )
    context = _prepare_expert_context(run, args.checkpoint)
    request = _ExpertReportRun(context, paths, ExpertProgressReporter(run.logger, len(paths)))
    reports = _collect_expert_reports(request)
    payload = _expert_report_payload(args.checkpoint, reports, context)
    event = _expert_event_payload(payload)
    run.logger.log("diagnose/expert", event, step=int(event["count"]))
    return payload, _write_expert_reports(run.run_dir, payload)


def _collect_expert_reports(request: _ExpertReportRun) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in request.paths:
        reports.append(_expert_demonstration_report(path, request.context))
        request.reporter.report(reports)
    return reports


def _prepare_expert_context(run: Any, checkpoint_path: Path) -> _ExpertContext:
    prepare_run(run)
    run.learner.setup({**_learner_context(run), "restoring_checkpoint": True})
    record_run_attempt(run)
    checkpoint = run.checkpoint_codec.load(checkpoint_path)
    run.learner.load_state_dict(_training_learner_state(checkpoint))
    config = _expert_environment_config(run)
    if config.geometry_path is None:
        raise ValueError("expert diagnostics require geometry_path")
    geometry = BoundaryGeometry(
        config.geometry_path,
        expected_map_uid=config.expected_map_uid,
    )
    return _ExpertContext(run.learner, run.feature_pipeline, config, geometry)


def _expert_report_payload(
    checkpoint: Path, reports: list[dict[str, Any]], context: _ExpertContext
) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "checkpoint": str(checkpoint),
        "demonstration_contract": _demonstration_contract(context),
        "metric_definitions": _expert_metric_definitions(),
        "demos": reports,
        "summary": _expert_report_summary(reports),
    }


def _demonstration_contract(context: _ExpertContext) -> dict[str, Any]:
    config = context.config
    compact = config.compact_action_ids
    return {
        "decision_interval_ms": config.decision_interval_ms,
        "action_lead_ms": config.demonstration_action_lead_ms,
        "aggregate_controls": config.demonstration_control_aggregation,
        "compact_action_ids": list(compact) if compact is not None else None,
        "model_action_count": int(context.learner.model.action_count),
    }


def _expert_report_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "demonstrations": len(reports),
        "actions": aggregate_expert_actions(report["actions"] for report in reports),
        "progress_bins": aggregate_expert_bins(report["progress_bins"] for report in reports),
    }


def _expert_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    return {"demonstrations/count": summary["demonstrations"], **summary["actions"]}


def _expert_terminal_summary(payload: dict[str, Any]) -> str:
    actions = payload["summary"]["actions"]
    return (
        "Expert aggregate: "
        f"exact={actions['exact_action_accuracy']:.4f}, "
        f"steering={actions['steering_bin_accuracy']:.4f}, "
        f"steering-switch-recall={actions['steering_switch_recall']:.4f}, "
        f"switch-step={actions['expert_steering_switch_step_accuracy']:.4f}, "
        f"steady={actions['expert_steering_steady_step_accuracy']:.4f}"
    )


def _expert_metric_definitions() -> dict[str, str]:
    return {
        **_action_metric_definitions(),
        **_steering_metric_definitions(),
    }


def _action_metric_definitions() -> dict[str, str]:
    return {
        "exact_action_accuracy": "policy argmax equals the expert model-space action",
        "action_switch_recall": "expert action-switch steps where the policy also switches action",
        "expert_action_switch_step_exact_accuracy": (
            "exact action accuracy on expert action-switch steps"
        ),
        "expert_action_steady_step_exact_accuracy": (
            "exact action accuracy on expert action-steady steps"
        ),
    }


def _steering_metric_definitions() -> dict[str, str]:
    return {
        "steering_bin_accuracy": "policy and expert canonical actions use the same steering bin",
        "steering_switch_recall": (
            "expert steering-switch steps where the policy also switches steering bin"
        ),
        "expert_steering_switch_step_accuracy": (
            "steering-bin accuracy on expert steering-switch steps"
        ),
        "expert_steering_steady_step_accuracy": (
            "steering-bin accuracy on expert steering-steady steps"
        ),
    }


def _write_expert_reports(run_dir: Path, payload: dict[str, Any]) -> Path:
    target = run_dir / "expert-diagnostics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target


def _expert_environment_config(run: Any) -> TrackmaniaEnvironmentConfig:
    environment_config = getattr(run.environment_factory, "config", None)
    if not isinstance(environment_config, TrackmaniaEnvironmentConfig):
        raise ValueError("expert diagnostics require OpenPlanetEnvironmentFactory")
    action_count, _ = select_brake_tap_actions(environment_config.compact_action_ids)
    if getattr(run.learner.model, "action_count", None) != action_count:
        raise ValueError("expert diagnostics require matching model and environment actions")
    return environment_config


def _expert_demonstration_report(
    path: Path,
    context: _ExpertContext,
) -> dict[str, Any]:
    demonstration, diagnostics = _expert_diagnostics(path, context)
    return {
        "path": str(path),
        "finish_time_s": demonstration.finish_time_s,
        "source_transition_count": len(demonstration.actions),
        "evaluated_transition_count": int(diagnostics.action_summary()["count"]),
        "actions": diagnostics.action_summary(),
        "progress_bins": diagnostics.summary(),
    }


def _expert_diagnostics(
    path: Path, context: _ExpertContext
) -> tuple[Demonstration, ExpertActionDiagnostics]:
    episode = _load_expert_episode(path, context)
    model = _expert_model(context)
    model.eval()
    reward = _expert_reward(context, episode.demonstration, len(episode.transitions))
    diagnostics = ExpertActionDiagnostics()
    state = model.initial_policy_state(1, context.learner.device)
    step_context = _ExpertStepContext(context, model, reward, diagnostics, state)
    _record_expert_steps(step_context, episode.transitions, episode.next_frames)
    return episode.demonstration, diagnostics


def _load_expert_episode(path: Path, context: _ExpertContext) -> _ExpertEpisode:
    demonstration = load_demonstration(path)
    validate_demonstration(demonstration, context.config, context.geometry)
    frames, _ = resample_demonstration_for_environment(demonstration, context.config)
    transitions = demonstration_transitions(
        path,
        context.pipeline,
        DemonstrationTransitionContext(context.config, context.geometry),
    )
    if len(transitions) != len(frames) - 1:
        raise RuntimeError("expert diagnostics transition contract is inconsistent")
    return _ExpertEpisode(demonstration, transitions, frames[1:])


def _expert_model(context: _ExpertContext) -> CompositeValueModel:
    model = context.learner.model
    if not isinstance(model, CompositeValueModel):
        raise RuntimeError("expert diagnostics require a CompositeValueModel")
    return model


def _expert_reward(
    context: _ExpertContext, demonstration: Any, transition_count: int
) -> TrajectoryReward:
    config = context.config
    reference = _expert_reference(context)
    pace_profile = _expert_pace_profile(context, reference)
    reward_config = replace(
        config.reward_config(pace_profile),
        no_progress_steps=transition_count + 1,
        slow_progress_window_steps=transition_count + 1,
    )
    reward = TrajectoryReward(reference, reward_config)
    _reset_expert_reward(reward, demonstration, config)
    return reward


def _expert_reference(context: _ExpertContext) -> Any:
    geometry = context.geometry
    return geometry.racing_line if context.config.use_racing_line else geometry.reward_center


def _expert_pace_profile(context: _ExpertContext, reference: Any) -> ReferencePaceProfile | None:
    path = context.config.pace_reference_path
    if path is None:
        return None
    request = PaceDemonstrationRequest(path, context.geometry, reference)
    return ReferencePaceProfile.from_demonstration(request)


def _reset_expert_reward(
    reward: TrajectoryReward, demonstration: Any, config: TrackmaniaEnvironmentConfig
) -> None:
    reward.reset(
        demonstration.frames[0, list(config.position_indices)],
        velocity=demonstration.frames[0, list(config.velocity_indices)],
        race_time_ms=float(demonstration.frames[0, 3]),
    )


def _record_expert_steps(
    context: _ExpertStepContext,
    transitions: list[Transition],
    next_frames: Any,
) -> None:
    for transition in zip(transitions, next_frames, strict=True):
        if _record_expert_step(context, transition):
            break


def _record_expert_step(context: _ExpertStepContext, transition: tuple[Transition, Any]) -> bool:
    sample, next_frame = transition
    q_values = _raw_q_values(context, sample.observation)
    source_action = int(sample.action)
    if not 0 <= source_action < q_values.shape[-1]:
        raise ValueError("demonstration action is outside the configured IQN action head")
    result = _advance_expert_reward(context.expert.config, context.reward, next_frame)
    context.diagnostics.record(_expert_diagnostic_record(context, q_values, source_action))
    return bool(result.terminated)


def _expert_diagnostic_record(
    context: _ExpertStepContext, q_values: torch.Tensor, source_action: int
) -> ExpertDiagnosticRecord:
    expert_q = float(q_values[source_action])
    greedy_action = int(q_values.argmax())
    return ExpertDiagnosticRecord(
        context.reward.progress_pct,
        expert_q,
        float(q_values.max()),
        int((q_values > expert_q).sum()) + 1,
        source_action,
        greedy_action,
        _steering_bin(context.expert.config, source_action),
        _steering_bin(context.expert.config, greedy_action),
    )


def _steering_bin(config: TrackmaniaEnvironmentConfig, action: int) -> int:
    action_ids = config.compact_action_ids
    canonical_action = action if action_ids is None else action_ids[action]
    return canonical_action // 6


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


def _raw_q_values(context: _ExpertStepContext, observation: Any) -> torch.Tensor:
    batched = _prepare_expert_observation(context.expert, observation)
    with torch.inference_mode():
        features, context.policy_state = context.model.policy_step(batched, context.policy_state)
        support = context.model.support(features, ValuePhase.EVALUATE)
        values = context.model.expected_all_actions(
            features, support, context.expert.learner.evaluation_risk
        )
    return values.squeeze(0).float().cpu()


def _prepare_expert_observation(context: _ExpertContext, observation: Any) -> Any:
    sanitized = sanitize_finite(observation)
    return tree_to_device(tree_collate([sanitized]), context.learner.device)
