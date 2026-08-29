from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean, median
from typing import TYPE_CHECKING, Any

from trackmaniarl.distributed.coordinator_leaders import candidate_from_stats
from trackmaniarl.trackmania.diagnostics import aggregate_progress_bins

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")


@dataclass(frozen=True, slots=True)
class _EvaluationBatch:
    summaries: list[dict[str, Any]]
    time_buckets_s: tuple[float, ...]
    trials: int
    policy_version: int
    finished_times: list[float]
    failure_progress: list[float]


@dataclass(frozen=True, slots=True)
class _TimingBatch:
    summaries: list[dict[str, Any]]
    steps: list[int]
    total_steps: int


def _bucket_key(bucket: float) -> str:
    return f"sub_{bucket:g}"


def finish_evaluation_batch(coordinator: Coordinator, summaries: list[dict[str, Any]]) -> None:
    stats = _evaluation_batch_stats(summaries, coordinator._time_buckets)
    coordinator.run.logger.log("eval/summary", stats, step=coordinator.counters.updates)
    _log_progress_bins(coordinator, stats)
    _log_evaluation_summary(coordinator, stats)
    candidate = candidate_from_stats(stats)
    coordinator._record_evaluation_leaders(candidate)
    coordinator._record_evaluation_stop(stats)


def _log_progress_bins(coordinator: Coordinator, stats: Mapping[str, float]) -> None:
    progress_bins = _progress_bin_metrics(stats)
    if not progress_bins:
        return
    coordinator.run.logger.log(
        "eval/progress_bin", progress_bins, step=coordinator.counters.updates
    )


def _log_evaluation_summary(coordinator: Coordinator, stats: Mapping[str, float]) -> None:
    logger.info(
        "Deterministic evaluation @update %d: %d/%d finished, mean=%.2fs, "
        "best=%.2fs, policy_version=%d",
        coordinator.counters.updates,
        int(stats["finished_trials"]),
        int(stats["trials"]),
        stats["finish_time_mean_s"],
        stats["finish_time_best_s"],
        int(stats["policy_version"]),
    )


def record_evaluation_stop(coordinator: Coordinator, stats: Mapping[str, float]) -> None:
    target = _evaluation_stop_target(coordinator)
    if target is None:
        return
    required_finish_rate, maximum_median_s, required_batches = target
    passed = stats["finish_rate"] >= required_finish_rate
    passed = passed and stats["finish_time_median_s"] <= maximum_median_s
    coordinator._consecutive_evaluation_passes = (
        coordinator._consecutive_evaluation_passes + 1 if passed else 0
    )
    if coordinator._consecutive_evaluation_passes < required_batches:
        return
    _record_evaluation_stop(coordinator, stats)


def _evaluation_stop_target(coordinator: Coordinator) -> tuple[float, float, int] | None:
    training = coordinator.run.spec.training
    finish_rate = getattr(training, "evaluation_stop_min_finish_rate", None)
    median_s = getattr(training, "evaluation_stop_median_s", None)
    batches = getattr(training, "evaluation_stop_consecutive_batches", None)
    if finish_rate is None or median_s is None or batches is None:
        return None
    return float(finish_rate), float(median_s), int(batches)


def _record_evaluation_stop(coordinator: Coordinator, stats: Mapping[str, float]) -> None:
    coordinator._evaluation_stop_reason = (
        "evaluation target passed "
        f"{coordinator._consecutive_evaluation_passes} consecutive times: "
        f"finish_rate={stats['finish_rate']:.3f}, "
        f"median_finish_time_s={stats['finish_time_median_s']:.3f}"
    )
    coordinator.run.logger.log(
        "train/early_stop",
        {
            "reason": coordinator._evaluation_stop_reason,
            "consecutive_passes": coordinator._consecutive_evaluation_passes,
            "finish_rate": stats["finish_rate"],
            "median_finish_time_s": stats["finish_time_median_s"],
        },
        step=coordinator.counters.updates,
    )
    logger.info("Stopping training: %s", coordinator._evaluation_stop_reason)


def _evaluation_batch_stats(
    summaries: list[dict[str, Any]], time_buckets_s: tuple[float, ...]
) -> dict[str, float]:
    batch = _evaluation_batch(summaries, time_buckets_s)
    stats = {
        **_finish_stats(batch),
        **_failure_stats(batch),
        **_control_stats(batch),
        **_termination_stats(batch),
        **_evaluation_timing_stats(summaries),
    }
    stats.update(aggregate_progress_bins(_progress_bin_summary(item) for item in summaries))
    return stats


def _evaluation_batch(
    summaries: list[dict[str, Any]], time_buckets_s: tuple[float, ...]
) -> _EvaluationBatch:
    policy_version = _evaluation_policy_version(summaries)
    finished_times = sorted(
        float(item["finish_time_s"]) for item in summaries if bool(item["finished"])
    )
    failure_progress = [
        float(item.get("progress_pct", 0.0)) for item in summaries if not bool(item["finished"])
    ]
    return _EvaluationBatch(
        summaries,
        time_buckets_s,
        len(summaries),
        policy_version,
        finished_times,
        failure_progress,
    )


def _evaluation_policy_version(summaries: list[dict[str, Any]]) -> int:
    if not summaries:
        raise ValueError("deterministic evaluation batch must not be empty")
    versions = {int(item["policy_version"]) for item in summaries}
    if len(versions) != 1:
        raise ValueError("deterministic evaluation batch mixed policy versions")
    return versions.pop()


def _finish_stats(batch: _EvaluationBatch) -> dict[str, float]:
    times = batch.finished_times
    return {
        "trials": float(batch.trials),
        "finished_trials": float(len(times)),
        "finish_rate": len(times) / batch.trials,
        "finish_time_mean_s": fmean(times) if times else 0.0,
        "finish_time_median_s": median(times) if times else 0.0,
        "finish_time_best_s": times[0] if times else 0.0,
        **{
            f"{_bucket_key(bucket)}_rate": sum(time_s < bucket for time_s in times) / batch.trials
            for bucket in batch.time_buckets_s
        },
        "policy_version": float(batch.policy_version),
    }


def _failure_stats(batch: _EvaluationBatch) -> dict[str, float]:
    progress = batch.failure_progress
    return {
        "failure_progress_mean_pct": fmean(progress) if progress else 100.0,
        "failure_progress_median_pct": median(progress) if progress else 100.0,
        "failure_progress_best_pct": max(progress) if progress else 100.0,
    }


def _control_stats(batch: _EvaluationBatch) -> dict[str, float]:
    return {**_driving_stats(batch), **_value_stats(batch)}


def _driving_stats(batch: _EvaluationBatch) -> dict[str, float]:
    summaries = batch.summaries
    return {
        "collision_rate": _collision_rate(batch),
        "control_brake_fraction_mean": _summary_mean(summaries, "control/brake_fraction"),
        "control_brake_tap_fraction_mean": _summary_mean(summaries, "control/brake_tap_fraction"),
        "control_gas_fraction_mean": _summary_mean(summaries, "control/gas_fraction"),
        "control_steer_abs_mean": _summary_mean(summaries, "control/steer_abs_mean"),
    }


def _collision_rate(batch: _EvaluationBatch) -> float:
    collisions = sum(int(float(item.get("collision/count", 0.0)) > 0.0) for item in batch.summaries)
    return collisions / batch.trials


def _summary_mean(summaries: list[dict[str, Any]], metric: str) -> float:
    return fmean(float(item.get(metric, 0.0)) for item in summaries)


def _value_stats(batch: _EvaluationBatch) -> dict[str, float]:
    return {
        "projected_velocity_ratio_mean": fmean(
            float(item.get("velocity/ratio_mean", 0.0)) for item in batch.summaries
        ),
        "q_margin_start_mean": fmean(
            float(item.get("q_margin/start_mean", 0.0)) for item in batch.summaries
        ),
    }


def _termination_stats(batch: _EvaluationBatch) -> dict[str, float]:
    summaries = batch.summaries
    return {
        "off_track_rate": sum(
            int(str(item.get("termination", "")) == "off_track") for item in summaries
        )
        / batch.trials,
        "telemetry_error_rate": sum(
            int(str(item.get("termination", "")) == "telemetry_error") for item in summaries
        )
        / batch.trials,
    }


def _evaluation_timing_stats(summaries: list[dict[str, Any]]) -> dict[str, float]:
    steps = [int(item.get("steps", 0)) for item in summaries]
    batch = _TimingBatch(summaries, steps, sum(steps))
    skipped_total = sum(
        float(item.get("telemetry_skipped_frames_total", 0.0)) for item in summaries
    )
    return {
        "action_latency_ms": _weighted_mean(batch, "timing/policy_inference_ms_mean"),
        "controller_apply_ms": _weighted_mean(batch, "controller_apply_ms_mean"),
        "telemetry_wait_ms": _weighted_mean(batch, "telemetry_wait_ms_mean"),
        "telemetry_skipped_frames_total": skipped_total,
        "telemetry_skipped_frames_mean": skipped_total / batch.total_steps
        if batch.total_steps
        else 0.0,
        "telemetry_skipped_frames_max": _maximum_metric(summaries, "telemetry_skipped_frames_max"),
        "telemetry_steps_with_skipped_frames_fraction": _weighted_mean(
            batch, "telemetry_steps_with_skipped_frames_fraction"
        ),
    }


def _maximum_metric(summaries: list[dict[str, Any]], key: str) -> float:
    return max(float(item.get(key, 0.0)) for item in summaries)


def _weighted_mean(batch: _TimingBatch, key: str) -> float:
    if batch.total_steps == 0:
        return 0.0
    weighted = sum(
        float(item.get(key, 0.0)) * count
        for item, count in zip(batch.summaries, batch.steps, strict=True)
    )
    return weighted / batch.total_steps


def _progress_bin_summary(summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    bins: dict[str, dict[str, float]] = {}
    for key, value in summary.items():
        prefix, separator, suffix = key.partition("progress_bin/")
        if prefix or not separator:
            continue
        name, metric_separator, metric = suffix.partition("/")
        if not metric_separator:
            continue
        bins.setdefault(name, {})[metric] = float(value)
    return bins


def _progress_bin_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        key.removeprefix("progress_bin/"): float(value)
        for key, value in summary.items()
        if key.startswith("progress_bin/")
    }
