"""Episode-level metrics collected by distributed actors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Self

from trackmaniarl.trackmania.diagnostics import ProgressBinDiagnostics, ProgressDiagnosticRecord

_EPISODE_START_MARGIN_STEPS = 50


class InferenceTimingTracker:
    def __init__(self) -> None:
        self.total_s = 0.0
        self.maximum_s = 0.0
        self.samples = 0

    def record(self, duration_s: float) -> None:
        self.total_s += duration_s
        self.maximum_s = max(self.maximum_s, duration_s)
        self.samples += 1

    def summary(self) -> dict[str, float]:
        mean_s = self.total_s / self.samples if self.samples else 0.0
        return {
            "policy_inference_ms_mean": mean_s * 1_000.0,
            "policy_inference_ms_max": self.maximum_s * 1_000.0,
        }


class MarginTracker:
    def __init__(self) -> None:
        self.total = 0.0
        self.minimum = float("inf")
        self.samples = 0
        self.start_total = 0.0
        self.start_samples = 0

    def record(self, policy: Any, step: int) -> None:
        margin = getattr(policy, "last_q_margin", None)
        if margin is None:
            return
        value = float(margin)
        self.total += value
        self.minimum = min(self.minimum, value)
        self.samples += 1
        if step < _EPISODE_START_MARGIN_STEPS:
            self.start_total += value
            self.start_samples += 1

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"q_margin_mean": 0.0, "q_margin_min": 0.0, "q_margin_start_mean": 0.0}
        return {
            "q_margin_mean": self.total / self.samples,
            "q_margin_min": self.minimum,
            "q_margin_start_mean": (
                self.start_total / self.start_samples if self.start_samples else 0.0
            ),
        }


class ControlUsageTracker:
    def __init__(self) -> None:
        self.gas_total = 0.0
        self.brake_total = 0.0
        self.brake_taps = 0
        self.steer_abs_total = 0.0
        self.race_ms_total = 0.0
        self.race_ms_values: list[float] = []
        self.controller_ms_values: list[float] = []
        self.telemetry_wait_ms_values: list[float] = []
        self.telemetry_skipped_frames_total = 0
        self.telemetry_skipped_frames_max = 0
        self.telemetry_steps_with_skipped_frames = 0
        self.samples = 0

    def record(self, info: Mapping[str, Any]) -> None:
        if "control_gas" not in info:
            return
        self.gas_total += float(info["control_gas"])
        self.brake_total += float(info["control_brake"])
        self.brake_taps += int(bool(info.get("control_brake_tap", False)))
        self.steer_abs_total += abs(float(info["control_steer"]))
        race_ms = float(info.get("step_race_time_ms", 0.0))
        self.race_ms_total += race_ms
        self.race_ms_values.append(race_ms)
        self.controller_ms_values.append(float(info.get("controller_apply_ms", 0.0)))
        self.telemetry_wait_ms_values.append(float(info.get("telemetry_wait_ms", 0.0)))
        skipped_frames = int(info.get("telemetry_skipped_frames", 0))
        self.telemetry_skipped_frames_total += skipped_frames
        self.telemetry_skipped_frames_max = max(self.telemetry_skipped_frames_max, skipped_frames)
        self.telemetry_steps_with_skipped_frames += int(skipped_frames > 0)
        self.samples += 1

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return self._empty_summary()
        ordered = sorted(self.race_ms_values)
        p99_index = ceil(0.99 * len(ordered)) - 1
        summary = {
            "control_gas_fraction": self.gas_total / self.samples,
            "control_brake_fraction": self.brake_total / self.samples,
            "control_brake_tap_fraction": self.brake_taps / self.samples,
            "control_steer_abs_mean": self.steer_abs_total / self.samples,
            "step_race_time_ms_mean": self.race_ms_total / self.samples,
            "step_race_time_ms_p99": ordered[p99_index],
            "step_race_time_ms_max": ordered[-1],
        }
        summary.update(self._timing_summary())
        return summary

    def _timing_summary(self) -> dict[str, float]:
        return {
            "controller_apply_ms_mean": sum(self.controller_ms_values) / self.samples,
            "controller_apply_ms_max": max(self.controller_ms_values),
            "telemetry_wait_ms_mean": sum(self.telemetry_wait_ms_values) / self.samples,
            "telemetry_wait_ms_max": max(self.telemetry_wait_ms_values),
            "telemetry_skipped_frames_total": float(self.telemetry_skipped_frames_total),
            "telemetry_skipped_frames_mean": self.telemetry_skipped_frames_total / self.samples,
            "telemetry_skipped_frames_max": float(self.telemetry_skipped_frames_max),
            "telemetry_steps_with_skipped_frames_fraction": (
                self.telemetry_steps_with_skipped_frames / self.samples
            ),
        }

    @staticmethod
    def _empty_summary() -> dict[str, float]:
        return {
            "control_gas_fraction": 0.0,
            "control_brake_fraction": 0.0,
            "control_brake_tap_fraction": 0.0,
            "control_steer_abs_mean": 0.0,
            "step_race_time_ms_mean": 0.0,
            "step_race_time_ms_p99": 0.0,
            "step_race_time_ms_max": 0.0,
            "controller_apply_ms_mean": 0.0,
            "controller_apply_ms_max": 0.0,
            "telemetry_wait_ms_mean": 0.0,
            "telemetry_wait_ms_max": 0.0,
            "telemetry_skipped_frames_total": 0.0,
            "telemetry_skipped_frames_mean": 0.0,
            "telemetry_skipped_frames_max": 0.0,
            "telemetry_steps_with_skipped_frames_fraction": 0.0,
        }


@dataclass(slots=True)
class EpisodeMetrics:
    diagnostics: ProgressBinDiagnostics
    total_reward: float = 0.0
    time_reward: float = 0.0
    pbrs_reward: float = 0.0
    progress_reward: float = 0.0
    projected_velocity_reward: float = 0.0
    projected_speed_reward: float = 0.0
    steering_delta_reward: float = 0.0
    collision_reward: float = 0.0
    collision_count: int = 0
    collision_detected_count: int = 0
    terminal_reward: float = 0.0
    time_attack_terminal_reward: float = 0.0
    pace_reward: float = 0.0
    velocity_ratio_sum: float = 0.0
    velocity_ratio_max: float = 0.0
    final_info: Mapping[str, Any] = field(default_factory=dict)
    margins: MarginTracker = field(default_factory=MarginTracker)
    controls: ControlUsageTracker = field(default_factory=ControlUsageTracker)
    inference_timing: InferenceTimingTracker = field(default_factory=InferenceTimingTracker)

    @classmethod
    def from_policy(cls, policy: Any) -> Self:
        return cls(ProgressBinDiagnostics(policy_action_count(policy), bin_count=20))

    def record_policy(self, policy: Any, step: int) -> None:
        self.margins.record(policy, step)

    def record_inference(self, duration_s: float) -> None:
        self.inference_timing.record(duration_s)

    def record_diagnostics(self, action: Any, policy: Any, info: Mapping[str, Any]) -> None:
        self.controls.record(info)
        record = ProgressDiagnosticRecord(
            float(info.get("progress_pct", 0.0)), action, policy, info
        )
        self.diagnostics.record(record)

    def record_reward(self, reward: float, info: Mapping[str, Any]) -> None:
        self.total_reward += float(reward)
        self.time_reward += float(info.get("reward_time", 0.0))
        self.pbrs_reward += float(info.get("reward_pbrs", 0.0))
        self.progress_reward += float(info.get("reward_progress", 0.0))
        self.projected_velocity_reward += float(info.get("reward_projected_velocity", 0.0))
        self.projected_speed_reward += float(info.get("reward_projected_speed", 0.0))
        self.steering_delta_reward += float(info.get("reward_steering_delta", 0.0))
        self.collision_reward += float(info.get("reward_collision", 0.0))
        self.collision_count += int(bool(info.get("collision", False)))
        self.collision_detected_count += int(bool(info.get("collision_detected", False)))
        self.terminal_reward += float(info.get("reward_terminal", 0.0))
        self.time_attack_terminal_reward += float(info.get("reward_time_attack_terminal", 0.0))
        self.pace_reward += float(info.get("reward_pace", 0.0))
        velocity_ratio = float(info.get("projected_velocity_ratio", 0.0))
        self.velocity_ratio_sum += velocity_ratio
        self.velocity_ratio_max = max(self.velocity_ratio_max, velocity_ratio)
        self.final_info = info

    def summary_info(self, epsilon: float, version: int, transitions: int) -> dict[str, Any]:
        return {
            **dict(self.final_info),
            **self._reward_totals(),
            **self._progress_totals(transitions),
            "actor_epsilon": epsilon,
            "policy_version": version,
            **self.margins.summary(),
            **self.controls.summary(),
            **self.inference_timing.summary(),
            **self.diagnostics.flat_summary(),
        }

    def _reward_totals(self) -> dict[str, float | int]:
        return {
            "reward_time": self.time_reward,
            "reward_pbrs": self.pbrs_reward,
            "reward_progress": self.progress_reward,
            "reward_projected_velocity": self.projected_velocity_reward,
            "reward_projected_speed": self.projected_speed_reward,
            "reward_steering_delta": self.steering_delta_reward,
            "reward_collision": self.collision_reward,
            "collision_count": self.collision_count,
            "collision_detected_count": self.collision_detected_count,
            "reward_terminal": self.terminal_reward,
            "reward_time_attack_terminal": self.time_attack_terminal_reward,
            "reward_pace": self.pace_reward,
        }

    def _progress_totals(self, transitions: int) -> dict[str, float]:
        return {
            "projected_velocity_ratio_mean": self.velocity_ratio_sum / max(transitions, 1),
            "projected_velocity_ratio_max": self.velocity_ratio_max,
        }


def policy_action_count(policy: Any) -> int:
    model = getattr(policy, "model", None)
    count = getattr(model, "action_count", 78)
    if not isinstance(count, int) or count < 2:
        raise ValueError("TrackMania policy must expose at least two actions")
    return count


def summarize_episode(reward: float, info: Mapping[str, Any], transitions: int) -> dict[str, Any]:
    termination = str(info.get("termination_reason") or "max_steps")
    summary: dict[str, Any] = {
        "return": reward,
        "reward_per_transition": reward / max(transitions, 1),
        "steps": transitions,
        "exploration_epsilon": float(info.get("actor_epsilon", 0.0)),
        "policy_version": info["policy_version"],
    }
    summary.update(_reward_summary(info))
    summary.update(_progress_summary(info))
    summary.update(_control_summary(info))
    summary.update(_telemetry_summary(info))
    summary.update(_outcome_summary(info, termination))
    summary.update(
        {key: float(value) for key, value in info.items() if key.startswith("progress_bin/")}
    )
    return summary


def _reward_summary(info: Mapping[str, Any]) -> dict[str, float]:
    return {
        "reward/time": float(info.get("reward_time", 0.0)),
        "reward/pbrs": float(info.get("reward_pbrs", 0.0)),
        "reward/progress": float(info.get("reward_progress", 0.0)),
        "reward/projected_velocity": float(info.get("reward_projected_velocity", 0.0)),
        "reward/projected_speed": float(info.get("reward_projected_speed", 0.0)),
        "reward/steering_delta": float(info.get("reward_steering_delta", 0.0)),
        "reward/collision": float(info.get("reward_collision", 0.0)),
        "reward/terminal": float(info.get("reward_terminal", 0.0)),
        "reward/time_attack_terminal": float(info.get("reward_time_attack_terminal", 0.0)),
        "reward/pace": float(info.get("reward_pace", 0.0)),
    }


def _progress_summary(info: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "pace/reference_time_s": float(info.get("reference_time_s", 0.0)),
        "pace/time_debt_s": float(info.get("time_debt_s", 0.0)),
        "progress/nearest_distance_m": float(info.get("nearest_distance_m", 0.0)),
        "progress/accepted_delta_m": float(info.get("accepted_progress_delta_m", 0.0)),
        "progress/window_m": float(info.get("window_progress_m", 0.0)),
        "progress/steps_since": float(info.get("steps_since_progress", 0.0)),
        "potential/progress": float(info.get("potential_progress", 0.0)),
        "velocity/projected_mps": float(info.get("projected_velocity_mps", 0.0)),
        "velocity/ratio": float(info.get("projected_velocity_ratio", 0.0)),
        "velocity/ratio_mean": float(info.get("projected_velocity_ratio_mean", 0.0)),
        "velocity/ratio_max": float(info.get("projected_velocity_ratio_max", 0.0)),
        "collision/count": int(info.get("collision_count", 0)),
        "collision/detected_count": int(info.get("collision_detected_count", 0)),
        "q_margin/mean": float(info.get("q_margin_mean", 0.0)),
        "q_margin/min": float(info.get("q_margin_min", 0.0)),
        "q_margin/start_mean": float(info.get("q_margin_start_mean", 0.0)),
    }


def _control_summary(info: Mapping[str, Any]) -> dict[str, float]:
    return {
        "control/gas_fraction": float(info.get("control_gas_fraction", 0.0)),
        "control/brake_fraction": float(info.get("control_brake_fraction", 0.0)),
        "control/brake_tap_fraction": float(info.get("control_brake_tap_fraction", 0.0)),
        "control/steer_abs_mean": float(info.get("control_steer_abs_mean", 0.0)),
        "timing/step_race_ms_mean": float(info.get("step_race_time_ms_mean", 0.0)),
        "timing/step_race_ms_p99": float(info.get("step_race_time_ms_p99", 0.0)),
        "timing/step_race_ms_max": float(info.get("step_race_time_ms_max", 0.0)),
        "timing/policy_inference_ms_mean": float(info.get("policy_inference_ms_mean", 0.0)),
        "timing/policy_inference_ms_max": float(info.get("policy_inference_ms_max", 0.0)),
        "controller_apply_ms_mean": float(info.get("controller_apply_ms_mean", 0.0)),
        "controller_apply_ms_max": float(info.get("controller_apply_ms_max", 0.0)),
        "telemetry_wait_ms_mean": float(info.get("telemetry_wait_ms_mean", 0.0)),
        "telemetry_wait_ms_max": float(info.get("telemetry_wait_ms_max", 0.0)),
    }


def _telemetry_summary(info: Mapping[str, Any]) -> dict[str, float]:
    return {
        "telemetry_skipped_frames_total": float(info.get("telemetry_skipped_frames_total", 0.0)),
        "telemetry_skipped_frames_mean": float(info.get("telemetry_skipped_frames_mean", 0.0)),
        "telemetry_skipped_frames_max": float(info.get("telemetry_skipped_frames_max", 0.0)),
        "telemetry_steps_with_skipped_frames_fraction": float(
            info.get("telemetry_steps_with_skipped_frames_fraction", 0.0)
        ),
        "telemetry/error": float(info.get("telemetry_error", 0.0)),
    }


def _outcome_summary(info: Mapping[str, Any], termination: str) -> dict[str, float | str]:
    finished = termination == "finished"
    race_time_s = float(info.get("race_time_ms", 0.0)) / 1_000.0
    return {
        "progress_pct": float(info.get("progress_pct", 0.0)),
        "progress_m": float(info.get("progress_m", 0.0)),
        "duration_s": float(info.get("episode_elapsed_s", 0.0)),
        "race_time_s": race_time_s,
        "finish_time_s": race_time_s if finished else 0.0,
        "finished": float(finished),
        "termination": termination,
        "termination/finished": float(finished),
        "termination/no_progress": float(termination == "no_progress"),
        "termination/slow_progress": float(termination == "slow_progress"),
        "termination/off_track": float(termination == "off_track"),
        "termination/time_limit": float(termination == "time_limit"),
        "termination/max_steps": float(termination == "max_steps"),
        "termination/telemetry_error": float(termination == "telemetry_error"),
    }
