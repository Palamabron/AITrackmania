"""Coordinator throughput, health, and actor-liveness metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.journal import JournalStatistics


@dataclass(frozen=True, slots=True)
class _MetricWindow:
    now: float
    elapsed: float
    transitions_per_s: float
    updates_per_s: float
    target_updates_per_s: float
    replay_capacity: int
    journal: JournalStatistics


def emit_metrics_if_ready(coordinator: Coordinator) -> None:
    interval = coordinator.run.spec.training.metrics_interval_updates
    if coordinator.counters.updates % interval != 0:
        return
    window = _metric_window(coordinator, interval)
    payload = _metric_payload(coordinator, window)
    _add_execution_metrics(coordinator, payload)
    _add_credit_health(coordinator, payload)
    _advance_metric_window(coordinator, window)
    started = perf_counter()
    coordinator.run.logger.log("train/update", payload, step=coordinator.counters.updates)
    coordinator._last_logging_s = perf_counter() - started


def _metric_window(coordinator: Coordinator, interval: int) -> _MetricWindow:
    now = monotonic()
    elapsed = max(now - coordinator._metric_window_started, 1e-6)
    transitions = coordinator.counters.transitions - coordinator._last_metric_transitions
    transitions_per_s = transitions / elapsed
    updates_per_s = interval / elapsed
    target = transitions_per_s * coordinator.run.spec.training.updates_per_transition
    capacity = int(getattr(coordinator.run.replay_store, "capacity", 0))
    journal = coordinator.journal.statistics(coordinator.counters.journal_applied_frontier)
    return _MetricWindow(now, elapsed, transitions_per_s, updates_per_s, target, capacity, journal)


def _metric_payload(coordinator: Coordinator, window: _MetricWindow) -> dict[str, object]:
    payload = {
        **coordinator._metrics.flush(),
        **_throughput_metrics(coordinator, window),
        **_journal_metrics(window.journal),
        "episodes": coordinator.counters.episodes,
        "finish_rate": coordinator.counters.finishes / max(1, coordinator.counters.episodes),
        "per_beta": coordinator.run.spec.training.replay_beta(coordinator.counters.transitions),
        "timing/logging_s": coordinator._last_logging_s,
    }
    if window.target_updates_per_s > 0.0:
        payload["update_throughput_ratio"] = window.updates_per_s / window.target_updates_per_s
    return payload


def _throughput_metrics(coordinator: Coordinator, window: _MetricWindow) -> dict[str, object]:
    size = len(coordinator.run.replay_store)
    return {
        "replay_size": size,
        "replay_fill_fraction": size / window.replay_capacity if window.replay_capacity else 0.0,
        "update_credit": coordinator.counters.update_credit,
        "rollout_queue_depth": coordinator._rollouts.qsize(),
        "updates_per_s": window.updates_per_s,
        "transitions_per_s": window.transitions_per_s,
        "cumulative_transitions_per_s": coordinator.counters.transitions
        / max(window.now - coordinator._started_at, 1e-6),
        "target_updates_per_s": window.target_updates_per_s,
        "update_backlog_s": coordinator.counters.update_credit / max(window.updates_per_s, 1e-6),
    }


def _journal_metrics(journal: JournalStatistics) -> dict[str, int]:
    return {
        "health/wal_pending_rows": journal.pending_rows,
        "health/wal_pending_payload_bytes": journal.pending_payload_bytes,
        "health/wal_receipt_rows": journal.receipt_rows,
        "health/wal_database_bytes": journal.database_bytes,
        "health/wal_bytes": journal.wal_bytes,
    }


def _add_execution_metrics(coordinator: Coordinator, payload: dict[str, object]) -> None:
    execution = getattr(coordinator.run.learner, "execution_manifest", None)
    if callable(execution):
        payload["execution"] = dict(execution())
    if torch.cuda.is_available():
        payload["accelerator_memory_bytes"] = torch.cuda.memory_allocated()


def _add_credit_health(coordinator: Coordinator, payload: dict[str, object]) -> None:
    if coordinator.counters.update_credit > coordinator._last_metric_credit:
        coordinator._growing_credit_windows += 1
    else:
        coordinator._growing_credit_windows = 0
    if coordinator._growing_credit_windows >= 5:
        payload["warning"] = "update credit has grown for five metric windows"


def _advance_metric_window(coordinator: Coordinator, window: _MetricWindow) -> None:
    coordinator._last_metric_credit = coordinator.counters.update_credit
    coordinator._metric_window_started = window.now
    coordinator._last_metric_transitions = coordinator.counters.transitions


def check_actor_timeouts(coordinator: Coordinator) -> None:
    now = monotonic()
    timeout = coordinator.run.spec.distributed.actor_timeout_s
    with coordinator._lock:
        heartbeats = tuple(coordinator._last_heartbeats.items())
    for actor_id, heartbeat in heartbeats:
        if now - heartbeat <= timeout:
            continue
        _record_actor_timeout(coordinator, actor_id, now - heartbeat)


def _record_actor_timeout(coordinator: Coordinator, actor_id: str, silence_s: float) -> None:
    with coordinator._lock:
        coordinator._last_heartbeats.pop(actor_id, None)
        coordinator._timed_out_actors.discard(actor_id)
    coordinator.run.logger.log(
        "actor/timeout",
        {"actor_id": actor_id, "silence_s": silence_s},
        step=coordinator.counters.updates,
    )
