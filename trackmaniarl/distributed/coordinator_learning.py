"""Asynchronous coordinator learner loop."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any

from trackmaniarl.distributed.coordinator_metrics import (
    check_actor_timeouts as check_actor_timeouts,
)
from trackmaniarl.distributed.coordinator_metrics import (
    emit_metrics_if_ready as emit_metrics_if_ready,
)
from trackmaniarl.distributed.coordinator_offline import (
    import_demonstrations as import_demonstrations,
)
from trackmaniarl.distributed.coordinator_offline import offline_pretrain as offline_pretrain
from trackmaniarl.distributed.coordinator_policy import can_update as can_update
from trackmaniarl.distributed.coordinator_policy import (
    external_stop_requested as external_stop_requested,
)
from trackmaniarl.distributed.coordinator_policy import has_active_actor as has_active_actor
from trackmaniarl.distributed.coordinator_policy import log_execution as log_execution
from trackmaniarl.distributed.coordinator_policy import publish_policy as publish_policy
from trackmaniarl.distributed.coordinator_policy import should_stop as should_stop
from trackmaniarl.distributed.coordinator_support import ROLLOUT_QUEUE_MAXSIZE, _BatchPrefetcher

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")


@dataclass(frozen=True, slots=True)
class _LearningLoop:
    coordinator: Coordinator
    prefetcher: _BatchPrefetcher
    ready: int


@dataclass(frozen=True, slots=True)
class _PreparedUpdate:
    metrics: Mapping[str, float]
    priorities: Any
    preparation_s: float
    wait_s: float
    learner_s: float


def learn(coordinator: Coordinator) -> None:
    spec = coordinator.run.spec.training
    footprint = spec.batch_size * spec.sequence_length + spec.n_step - 1
    loop = _LearningLoop(
        coordinator,
        _BatchPrefetcher(coordinator.run),
        max(spec.warmup_transitions, footprint),
    )
    try:
        while _learning_needed(loop):
            if not _learning_step(loop):
                break
    finally:
        loop.prefetcher.close()


def _learning_needed(loop: _LearningLoop) -> bool:
    coordinator = loop.coordinator
    if coordinator._external_stop_requested():
        return False
    can_drain_credit = (
        coordinator._evaluation_stop_reason is None
        and len(coordinator.run.replay_store) >= loop.ready
        and coordinator.counters.update_credit >= 1.0
    )
    journal_pending = coordinator.journal.has_rows_after(
        coordinator.counters.journal_applied_frontier
    )
    return not coordinator._should_stop() or can_drain_credit or journal_pending


def _learning_step(loop: _LearningLoop) -> bool:
    coordinator = loop.coordinator
    coordinator._check_actor_timeouts()
    coordinator._drain_rollouts(ROLLOUT_QUEUE_MAXSIZE)
    if coordinator._evaluation_stop_reason is not None:
        return False
    if _ready_to_update(loop):
        _perform_update(loop)
    else:
        sleep(0.005)
    return True


def _ready_to_update(loop: _LearningLoop) -> bool:
    coordinator = loop.coordinator
    return (
        coordinator._can_update()
        and len(coordinator.run.replay_store) >= loop.ready
        and coordinator.counters.update_credit >= 1.0
    )


def _perform_update(loop: _LearningLoop) -> None:
    coordinator = loop.coordinator
    update = _prepare_update(loop)
    if update.priorities is not None:
        coordinator.run.sampler.update_priorities(update.priorities)
    coordinator.counters.updates += 1
    coordinator.counters.update_credit -= 1.0
    coordinator._metrics.add(_update_metrics(update))
    coordinator._emit_metrics_if_ready()
    _log_progress(coordinator, coordinator.run.spec.training.total_transitions)
    _checkpoint_if_due(coordinator)
    coordinator._publish_policy()


def _prepare_update(loop: _LearningLoop) -> _PreparedUpdate:
    coordinator = loop.coordinator
    spec = coordinator.run.spec.training
    request = spec.batch_request(
        beta=spec.replay_beta(coordinator.counters.transitions),
        transition_count=coordinator.counters.transitions,
    )
    batch, preparation_s, wait_s = loop.prefetcher.next(request)
    started = perf_counter()
    result = coordinator.run.learner.update(batch)
    learner_s = perf_counter() - started
    metrics, priorities = result if isinstance(result, tuple) else (result, None)
    return _PreparedUpdate(metrics, priorities, preparation_s, wait_s, learner_s)


def _update_metrics(update: _PreparedUpdate) -> dict[str, float]:
    return {
        **update.metrics,
        "timing/replay_sample_s": update.preparation_s,
        "timing/replay_wait_s": update.wait_s,
        "timing/learner_update_s": update.learner_s,
    }


def _checkpoint_if_due(coordinator: Coordinator) -> None:
    interval = coordinator.run.spec.training.checkpoint_interval_updates
    if interval is None or coordinator.counters.updates % interval:
        return
    coordinator._checkpoints.append(coordinator._checkpoint())


def _log_progress(coordinator: Coordinator, total_transitions: int) -> None:
    since_last = coordinator.counters.updates - coordinator._last_progress_print
    if coordinator.counters.updates != 1 and since_last < 100:
        return
    coordinator._last_progress_print = coordinator.counters.updates
    logger.info(
        "Async training progress: transitions=%d/%d, updates=%d, replay=%d, credit=%.1f",
        coordinator.counters.transitions,
        total_transitions,
        coordinator.counters.updates,
        len(coordinator.run.replay_store),
        coordinator.counters.update_credit,
    )
