from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from queue import Empty
from time import monotonic
from typing import TYPE_CHECKING, Any

from trackmaniarl.distributed import coordinator_episode_ingest
from trackmaniarl.distributed.coordinator_evaluation import (
    _bucket_key,
    _progress_bin_metrics,
)
from trackmaniarl.distributed.coordinator_support import snapshot_value
from trackmaniarl.distributed.protocol import transition_from_wire

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")


@dataclass(frozen=True, slots=True)
class _IngestBatch:
    coordinator: Coordinator
    value: Mapping[str, Any]
    row_id: int
    transitions: list[Any]
    now: float
    ingest_fps: float


@dataclass(frozen=True, slots=True)
class _EvaluationSummary:
    coordinator: Coordinator
    summary: Mapping[str, Any]
    finished: bool


def log_wal_error(coordinator: Coordinator, operation: str, exc: BaseException) -> None:
    coordinator.run.logger.log(
        "distributed/wal_error",
        {
            "operation": operation,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "journal_path": str(coordinator.journal.path),
            "journal_applied_frontier": coordinator.counters.journal_applied_frontier,
        },
        step=coordinator.counters.updates,
    )


def journal_rows(
    coordinator: Coordinator, watermark: int, operation: str
) -> Iterator[tuple[int, bytes]]:
    try:
        yield from coordinator.journal.rows_after(watermark)
    except Exception as exc:
        coordinator._log_wal_error(operation, exc)
        raise


def decode_journal_payload(
    coordinator: Coordinator, payload: bytes, operation: str
) -> Mapping[str, Any]:
    try:
        value = coordinator.codec.decode(payload)
        if not isinstance(value, Mapping):
            raise ValueError("journal chunk must decode to a mapping")
    except Exception as exc:
        coordinator._log_wal_error(operation, exc)
        raise
    return value


def ingest(coordinator: Coordinator, value: Mapping[str, Any], row_id: int) -> None:
    if row_id <= coordinator.counters.journal_applied_frontier:
        raise ValueError("journal rows must be ingested in strictly increasing order")
    batch = _ingest_batch(coordinator, value, row_id)
    before = coordinator.counters.transitions
    _append_transitions(batch)
    _credit_updates(coordinator, before)
    _record_actor_sequence(coordinator, value)
    coordinator_episode_ingest.ingest_episode_summaries(coordinator, value)
    evaluations = [dict(summary) for summary in value["evaluations"]]
    _ingest_evaluation_snapshot(coordinator, value, evaluations)
    _ingest_evaluation_summaries(coordinator, value, evaluations)
    coordinator.counters.journal_applied_frontier = row_id
    _finish_ingest(batch, evaluations)


def _ingest_batch(coordinator: Coordinator, value: Mapping[str, Any], row_id: int) -> _IngestBatch:
    transitions = [transition_from_wire(item) for item in value["transitions"]]
    now = monotonic()
    elapsed = max(now - coordinator._last_ingest_at, 1e-6)
    ingest_fps = len(transitions) / elapsed
    coordinator._last_ingest_at = now
    return _IngestBatch(coordinator, value, row_id, transitions, now, ingest_fps)


def _append_transitions(batch: _IngestBatch) -> None:
    for transition in batch.transitions:
        replay_info = replay_info_for_transition(transition.info)
        batch.coordinator.run.replay_store.append(replace(transition, info=replay_info))
    batch.coordinator.counters.transitions += len(batch.transitions)


def _credit_updates(coordinator: Coordinator, before: int) -> None:
    ready = coordinator.run.spec.training.warmup_transitions
    newly_trainable = max(0, coordinator.counters.transitions - ready) - max(0, before - ready)
    coordinator.counters.update_credit = min(
        coordinator.run.spec.distributed.max_update_credit,
        coordinator.counters.update_credit
        + newly_trainable * coordinator.run.spec.training.updates_per_transition,
    )


def _record_actor_sequence(coordinator: Coordinator, value: Mapping[str, Any]) -> None:
    session_id = str(value["session_id"])
    coordinator.counters.actor_sequences[session_id] = max(
        coordinator.counters.actor_sequences.get(session_id, -1),
        int(value["sequence"]),
    )


def _finish_ingest(batch: _IngestBatch, evaluations: list[dict[str, Any]]) -> None:
    coordinator = batch.coordinator
    if evaluations and not coordinator._recovering:
        coordinator._finish_evaluation_batch(evaluations)
    if not coordinator._recovering:
        _log_ingest(batch)


def _ingest_evaluation_snapshot(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    evaluations: list[dict[str, Any]],
) -> None:
    evaluation_snapshot = value["evaluation_snapshot"]
    if not evaluations or not evaluation_snapshot:
        return
    if not isinstance(evaluation_snapshot, bytes):
        raise ValueError("evaluation snapshot must be bytes")
    policy_state = coordinator.codec.decode(evaluation_snapshot)
    if not isinstance(policy_state, Mapping):
        raise ValueError("evaluation snapshot must decode to a mapping")
    versions = {int(summary["policy_version"]) for summary in evaluations}
    if len(versions) != 1:
        raise ValueError("evaluation snapshot cannot cover mixed policy versions")
    with coordinator._lock:
        coordinator._evaluation_policy_states[versions.pop()] = snapshot_value(policy_state)


def _ingest_evaluation_summaries(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    evaluations: list[dict[str, Any]],
) -> None:
    for summary in evaluations:
        _ingest_evaluation_summary(coordinator, value, summary)


def _ingest_evaluation_summary(
    coordinator: Coordinator, value: Mapping[str, Any], summary: dict[str, Any]
) -> None:
    coordinator.counters.evaluations += 1
    finished = bool(summary["finished"])
    coordinator.counters.evaluation_finishes += int(finished)
    bucket_metrics = _evaluation_bucket_metrics(_EvaluationSummary(coordinator, summary, finished))
    if coordinator._recovering:
        return
    payload = {
        **summary,
        "index": coordinator.counters.evaluations,
        "finish_rate": coordinator.counters.evaluation_finishes / coordinator.counters.evaluations,
        **bucket_metrics,
        "actor_id": value["actor_id"],
    }
    coordinator.run.logger.log("eval/episode", payload, step=coordinator.counters.updates)


def _evaluation_bucket_metrics(evaluation: _EvaluationSummary) -> dict[str, float]:
    coordinator = evaluation.coordinator
    metrics: dict[str, float] = {}
    finish_time_s = float(evaluation.summary["finish_time_s"])
    for bucket in coordinator._time_buckets:
        key = _bucket_key(bucket)
        hit = evaluation.finished and finish_time_s < bucket
        count = coordinator.counters.evaluation_bucket_finishes.get(key, 0) + int(hit)
        coordinator.counters.evaluation_bucket_finishes[key] = count
        metrics[key] = float(hit)
        metrics[f"{key}_rate"] = count / coordinator.counters.evaluations
    return metrics


def _log_ingest(batch: _IngestBatch) -> None:
    coordinator = batch.coordinator
    coordinator.run.logger.log(
        "distributed/ingest",
        _ingest_payload(batch),
        step=coordinator.counters.updates,
    )


def _ingest_payload(batch: _IngestBatch) -> dict[str, Any]:
    coordinator = batch.coordinator
    trainable = coordinator.counters.transitions - coordinator.run.spec.training.warmup_transitions
    return {
        "actor_id": batch.value["actor_id"],
        "chunk_transitions": len(batch.transitions),
        "transitions": coordinator.counters.transitions,
        "replay_size": len(coordinator.run.replay_store),
        "ingest_fps": batch.ingest_fps,
        "policy_lag_updates": max(
            0, coordinator.counters.updates - int(batch.value["policy_version"])
        ),
        "utd": coordinator.counters.updates / max(1, trainable),
        "queue_delay_s": max(0.0, batch.now - float(batch.value.get("_enqueued_at", batch.now))),
        "rollout_queue_depth": coordinator._rollouts.qsize(),
    }


def replay_info_for_transition(info: Mapping[str, Any]) -> dict[str, Any]:
    replay_info: dict[str, Any] = {}
    if "is_demo" in info:
        replay_info["is_demo"] = bool(info["is_demo"])
    if "sampling/projected_lap_time_s" in info:
        replay_info["sampling/projected_lap_time_s"] = float(info["sampling/projected_lap_time_s"])
    return replay_info


def log_episode(
    coordinator: Coordinator, value: Mapping[str, Any], summary: Mapping[str, Any]
) -> None:
    payload = _episode_payload(coordinator, value, summary)
    coordinator.run.logger.log("train/episode", payload, step=coordinator.counters.updates)
    progress_bins = _progress_bin_metrics(summary)
    if progress_bins:
        coordinator.run.logger.log(
            "train/progress_bin", progress_bins, step=coordinator.counters.updates
        )
    logger.info(_EPISODE_LOG_FORMAT, _episode_log_fields(coordinator, value, summary))


def _episode_payload(
    coordinator: Coordinator, value: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **summary,
        "index": coordinator.counters.episodes,
        "finish_count": coordinator.counters.finishes,
        "finish_rate": coordinator.counters.finishes / coordinator.counters.episodes,
        "best_finish_time_s": coordinator.counters.best_finish_time_s,
        "actor_id": value["actor_id"],
        "replay_size": len(coordinator.run.replay_store),
    }


def _episode_log_fields(
    coordinator: Coordinator, value: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **summary,
        "actor_id": value["actor_id"],
        "episode_index": coordinator.counters.episodes,
        "policy_version": summary["policy_version"],
        "q_margin/start_mean": summary.get("q_margin/start_mean", 0.0),
        "q_margin/min": summary.get("q_margin/min", 0.0),
    }


def drain_rollouts(coordinator: Coordinator, limit: int) -> None:
    _drain_wakeups(coordinator, limit)
    _drain_journal_rows(coordinator, limit)


def _drain_wakeups(coordinator: Coordinator, limit: int) -> None:
    for _ in range(limit):
        try:
            wake = coordinator._rollouts.get_nowait()
        except Empty:
            break
        _remember_wakeup(coordinator, wake)
        coordinator._rollouts.task_done()


def _remember_wakeup(coordinator: Coordinator, wake: tuple[int, float]) -> None:
    if wake[0] > coordinator.counters.journal_applied_frontier:
        coordinator._journal_enqueued_at[wake[0]] = float(wake[1])


def _drain_journal_rows(coordinator: Coordinator, limit: int) -> None:
    frontier = coordinator.counters.journal_applied_frontier
    rows = coordinator._journal_rows(frontier, "drain")
    for applied, (row_id, payload) in enumerate(rows, start=1):
        value = coordinator._decode_journal_payload(payload, "drain_decode")
        materialized = _materialized_ingest(coordinator, row_id, value)
        coordinator._ingest(materialized, row_id)
        if applied >= limit:
            return


def _materialized_ingest(
    coordinator: Coordinator, row_id: int, value: Mapping[str, Any]
) -> dict[str, Any]:
    materialized = dict(value)
    queued_at = coordinator._journal_enqueued_at.pop(row_id, None)
    if queued_at is not None:
        materialized["_enqueued_at"] = queued_at
    return materialized


def recover_journal(coordinator: Coordinator, watermark: int) -> None:
    coordinator._recovering = True
    try:
        recovered_rows, recovered_transitions = _recover_rows(coordinator, watermark)
    finally:
        coordinator._recovering = False
    if recovered_rows:
        _log_recovery(coordinator, watermark, (recovered_rows, recovered_transitions))


def _recover_rows(coordinator: Coordinator, watermark: int) -> tuple[int, int]:
    rows = 0
    transitions = 0
    for row_id, payload in coordinator._journal_rows(watermark, "recovery"):
        value = coordinator._decode_journal_payload(payload, "recovery_decode")
        coordinator._ingest(value, row_id)
        rows += 1
        transitions += len(value["transitions"])
    return rows, transitions


def _log_recovery(coordinator: Coordinator, watermark: int, recovered: tuple[int, int]) -> None:
    payload = {
        "rows": recovered[0],
        "transitions": recovered[1],
        "from_frontier": watermark,
        "to_frontier": coordinator.counters.journal_applied_frontier,
    }
    coordinator.run.logger.log(
        "distributed/wal_recovery", payload, step=coordinator.counters.updates
    )


_EPISODE_LOG_FORMAT = (
    "Actor %(actor_id)s episode %(episode_index)d: progress=%(progress_pct).1f%%, "
    "return=%(return).3f, reward(time=%(reward/time).3f, pace=%(reward/pace).3f, "
    "pbrs=%(reward/pbrs).3f, progress=%(reward/progress).3f, "
    "projected_velocity=%(reward/projected_velocity).3f, "
    "projected_speed=%(reward/projected_speed).3f, steering_delta=%(reward/steering_delta).3f, "
    "collision=%(reward/collision).3f (%(collision/count)d/%(collision/detected_count)d), "
    "terminal=%(reward/terminal).3f, "
    "time_attack_terminal=%(reward/time_attack_terminal).3f), "
    "velocity_ratio(mean=%(velocity/ratio_mean).3f, "
    "max=%(velocity/ratio_max).3f), steps=%(steps)d, race=%(race_time_s).2fs, "
    "epsilon=%(exploration_epsilon).3f, policy=%(policy_version)d, "
    "q_margin(start=%(q_margin/start_mean).2f, min=%(q_margin/min).2f), "
    "termination=%(termination)s"
)
