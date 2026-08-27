"""Metric projection for the optional W&B adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_UPDATE_METRICS = {
    "replay_size": "replay/size",
    "replay_fill_fraction": "replay/fill_fraction",
    "update_credit": "pipeline/update_credit",
    "rollout_queue_depth": "pipeline/rollout_queue_depth",
    "updates_per_s": "performance/updates_per_s",
    "transitions_per_s": "performance/transitions_per_s",
    "cumulative_transitions_per_s": "performance/cumulative_transitions_per_s",
    "target_updates_per_s": "performance/target_updates_per_s",
    "update_throughput_ratio": "performance/update_throughput_ratio",
    "update_backlog_s": "pipeline/update_backlog_s",
    "episodes": "outcome/train_episodes",
    "finish_rate": "outcome/train_finish_rate",
    "per_beta": "replay/per_beta",
    "accelerator_memory_bytes": "system/torch_cuda_memory_bytes",
}
_DEBUG_METRICS = {
    "action_batch_entropy",
    "action_batch_unique_fraction",
    "action_entropy",
    "action_unique_fraction",
    "bootstrap_discount_mean",
    "bootstrap_zero_fraction",
    "demonstration_action_accuracy",
    "gradient_clip_coefficient",
    "gradient_clipped_fraction",
    "gradient_norm",
    "gradient_norm_max",
    "importance_weight_mean",
    "importance_weight_min",
    "n_step_return_mean",
    "q_selected_abs_max",
    "q_selected_max",
    "q_selected_mean",
    "q_selected_std_mean",
    "q_target_max",
    "q_target_mean",
    "reward_abs_max",
    "reward_mean",
    "target_abs_max",
    "target_mean",
    "target_std_mean",
    "target_synced_fraction",
    "td_abs_max",
    "td_abs_mean",
}
_TIMING_METRICS = {
    "backward_s",
    "checkpoint_snapshot_s",
    "forward_s",
    "gradient_clip_s",
    "host_to_device_s",
    "learner_update_s",
    "logging_s",
    "optimizer_s",
    "policy_publish_s",
    "replay_sample_s",
    "replay_wait_s",
    "update_s",
}
_EPISODE_METRICS = {
    "collision/count",
    "collision/detected_count",
    "control/brake_fraction",
    "control/brake_tap_fraction",
    "control/gas_fraction",
    "control/steer_abs_mean",
    "duration_s",
    "exploration_epsilon",
    "finish_rate",
    "finish_time_s",
    "finished",
    "index",
    "policy_version",
    "pace/reference_time_s",
    "pace/time_debt_s",
    "potential/progress",
    "progress/accepted_delta_m",
    "progress_m",
    "progress/nearest_distance_m",
    "progress_pct",
    "progress/steps_since",
    "progress/window_m",
    "q_margin/mean",
    "q_margin/min",
    "q_margin/start_mean",
    "race_time_s",
    "return",
    "reward/collision",
    "reward/pace",
    "reward/pbrs",
    "reward/progress",
    "reward/projected_speed",
    "reward/projected_velocity",
    "reward/steering_delta",
    "reward/terminal",
    "reward/time",
    "reward/time_attack_terminal",
    "reward_per_transition",
    "steps",
    "telemetry/error",
    "termination/max_steps",
    "termination/no_progress",
    "termination/off_track",
    "termination/slow_progress",
    "termination/telemetry_error",
    "termination/time_limit",
    "timing/policy_inference_ms_max",
    "timing/policy_inference_ms_mean",
    "timing/step_race_ms_max",
    "timing/step_race_ms_mean",
    "timing/step_race_ms_p99",
    "controller_apply_ms_mean",
    "controller_apply_ms_max",
    "telemetry_wait_ms_mean",
    "telemetry_wait_ms_max",
    "telemetry_skipped_frames_total",
    "telemetry_skipped_frames_mean",
    "telemetry_skipped_frames_max",
    "telemetry_steps_with_skipped_frames_fraction",
    "velocity/projected_mps",
    "velocity/ratio",
    "velocity/ratio_mean",
    "velocity/ratio_max",
}
_EVALUATION_METRICS = {
    "action_latency_ms",
    "crash_rate",
    "collision_rate",
    "control_brake_fraction_mean",
    "control_brake_tap_fraction_mean",
    "control_gas_fraction_mean",
    "control_steer_abs_mean",
    "failure_progress_best_pct",
    "failure_progress_mean_pct",
    "failure_progress_median_pct",
    "finish_rate",
    "finish_time_best_s",
    "finish_time_mean_s",
    "finish_time_median_s",
    "finished_trials",
    "off_track_rate",
    "policy_version",
    "projected_velocity_ratio_mean",
    "q_margin_start_mean",
    "reward",
    "sub_36_rate",
    "sub_38_rate",
    "sub_40_rate",
    "telemetry_error_rate",
    "controller_apply_ms",
    "telemetry_wait_ms",
    "telemetry_skipped_frames_total",
    "telemetry_skipped_frames_mean",
    "telemetry_skipped_frames_max",
    "telemetry_steps_with_skipped_frames_fraction",
    "throughput_fps",
    "trials",
}
_IMITATION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "best",
    "control_score",
    "intervention_accuracy",
    "learning_rate",
    "loss",
    "steering_accuracy",
    "steering_transition_accuracy",
    "student_disagreement_accuracy",
    "transition_accuracy",
    "weighted_accuracy",
}
_IGNORED_REMOTE_EVENTS = {
    "actor/heartbeat",
    "eval/progress_bin",
    "train/execution",
    "train/progress_bin",
}
_EVALUATION_ALIASES = {
    "finish_time_s": "finish_time_mean_s",
    "median_finish_time_s": "finish_time_median_s",
    "best_finish_time_s": "finish_time_best_s",
    "sub36_rate": "sub_36_rate",
    "sub38_rate": "sub_38_rate",
    "sub40_rate": "sub_40_rate",
}


def _flat_name(namespace: str, key: str) -> str:
    return f"{namespace}/{key.replace('/', '_')}"


def _selected(payload: Mapping[str, Any], keys: set[str], namespace: str) -> dict[str, Any]:
    return {_flat_name(namespace, key): payload[key] for key in keys if key in payload}


def _normalized_evaluation(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.update(
        {
            key.removeprefix("eval/"): value
            for key, value in payload.items()
            if key.startswith("eval/")
        }
    )
    return normalized


def _evaluation_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_evaluation(payload)
    normalized.update(
        {
            target: normalized[source]
            for source, target in _EVALUATION_ALIASES.items()
            if source in normalized and target not in normalized
        }
    )
    return _selected(normalized, _EVALUATION_METRICS, "evaluation")


def _update_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = {name: payload[key] for key, name in _UPDATE_METRICS.items() if key in payload}
    for key, value in payload.items():
        prefix, separator, suffix = key.partition("/")
        if prefix in {"loss", "gradients"} and separator:
            values[_flat_name("learner", key)] = value
        elif prefix == "debug" and suffix in _DEBUG_METRICS:
            values[_flat_name("learner", suffix)] = value
        elif prefix == "timing" and suffix in _TIMING_METRICS:
            values[_flat_name("performance", suffix)] = value
    return values


def _event_metrics(event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if event == "train/update":
        return _update_metrics(payload)
    if event == "train/episode":
        return _selected(payload, _EPISODE_METRICS, "episode")
    if event in {"eval/summary", "eval/suite"}:
        return _evaluation_metrics(payload)
    if event in {"bc/train", "bc/validation"}:
        phase = event.removeprefix("bc/")
        return _selected(payload, _IMITATION_METRICS, f"imitation_{phase}")
    if event == "distributed/policy_published":
        return _policy_publish_metrics(payload)
    if event == "train/checkpoint":
        return _checkpoint_metrics(payload)
    return {}


def _policy_publish_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pipeline/policy_version": payload.get("policy_version", 0),
        "performance/policy_publish_s": payload.get("timing/policy_publish_s", 0.0),
    }


def _checkpoint_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "performance/checkpoint_snapshot_s": payload.get("timing/checkpoint_snapshot_s", 0.0),
    }


def _load_wandb_key_from_dotenv(run_dir: str | None) -> Path | None:
    if os.environ.get("WANDB_API_KEY"):
        return None
    starts = [Path.cwd()]
    if run_dir:
        starts.append(Path(run_dir))
    checked: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            candidate = directory / ".env"
            if candidate in checked:
                continue
            checked.add(candidate)
            value = _dotenv_value(candidate, "WANDB_API_KEY")
            if value:
                os.environ["WANDB_API_KEY"] = value
                return candidate
    return None


def _dotenv_value(path: Path, requested_key: str) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
        if separator and key.strip() == requested_key:
            return value.strip().strip('"').strip("'") or None
    return None
