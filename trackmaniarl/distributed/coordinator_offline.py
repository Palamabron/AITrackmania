"""Demonstration import and offline pretraining for the coordinator."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")


@dataclass(slots=True)
class _DemonstrationImport:
    transitions: int = 0
    finish_times: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _OfflineRun:
    coordinator: Coordinator
    updates: int
    started_at: float


def import_demonstrations(coordinator: Coordinator) -> None:
    if not coordinator.demo_paths:
        return
    loader = _demonstration_loader(coordinator)
    logger.info("Importing %d demonstration file(s) into replay...", len(coordinator.demo_paths))
    imported = _DemonstrationImport()
    for path in coordinator.demo_paths:
        count, finish_time = _import_demonstration(coordinator, loader, path)
        imported.transitions += count
        imported.finish_times.append(finish_time)
    _log_demonstration_import(coordinator, imported)


def _demonstration_loader(coordinator: Coordinator) -> Callable[..., Any]:
    loader = getattr(coordinator.run.environment_factory, "load_demonstration", None)
    if not callable(loader):
        raise ValueError("configured environment does not support replay demonstrations")
    return cast(Callable[..., Any], loader)


def _import_demonstration(
    coordinator: Coordinator, loader: Callable[..., Any], path: Path
) -> tuple[int, float]:
    transitions = loader(path, coordinator.run.feature_pipeline)
    for transition in transitions:
        coordinator.run.replay_store.append(transition)
    count = len(transitions)
    finish_time = float(transitions[0].info["sampling/projected_lap_time_s"])
    logger.info("Imported demonstration %s: %d transitions", path, count)
    return count, finish_time


def _log_demonstration_import(coordinator: Coordinator, imported: _DemonstrationImport) -> None:
    logger.info(
        "Demonstration import complete: %d transitions from %d file(s)",
        imported.transitions,
        len(coordinator.demo_paths),
    )
    payload = {
        "files": len(coordinator.demo_paths),
        "transitions": imported.transitions,
        "best_finish_time_s": min(imported.finish_times),
        "replay_size": len(coordinator.run.replay_store),
    }
    coordinator.run.logger.log("train/demonstrations", payload, step=coordinator.counters.updates)


def offline_pretrain(coordinator: Coordinator) -> None:
    updates = coordinator.run.spec.training.offline_pretrain_updates
    if updates == 0:
        return
    _validate_offline_run(coordinator)
    run = _OfflineRun(coordinator, updates, perf_counter())
    metrics = _run_offline_updates(run)
    _log_offline_pretraining(run, _mean_metrics(metrics))


def _validate_offline_run(coordinator: Coordinator) -> None:
    if not coordinator.demo_paths:
        raise ValueError("offline_pretrain_updates requires at least one demonstration")
    spec = coordinator.run.spec.training
    footprint = spec.batch_size * spec.sequence_length + spec.n_step - 1
    if len(coordinator.run.replay_store) < footprint:
        raise RuntimeError(
            "offline demonstration replay is too small for the configured batch footprint"
        )


def _run_offline_updates(run: _OfflineRun) -> list[Mapping[str, float]]:
    coordinator = run.coordinator
    begin = getattr(coordinator.run.learner, "begin_offline_pretraining", None)
    end = getattr(coordinator.run.learner, "end_offline_pretraining", None)
    if callable(begin):
        begin()
    try:
        return [_offline_update(run, index) for index in range(1, run.updates + 1)]
    finally:
        if callable(end):
            end()


def _offline_update(run: _OfflineRun, index: int) -> Mapping[str, float]:
    coordinator = run.coordinator
    spec = coordinator.run.spec.training
    request = spec.batch_request(beta=spec.replay_beta(0))
    batch = coordinator.run.sampler.sample(coordinator.run.replay_store, request)
    result = coordinator.run.learner.update(batch)
    values, priorities = result if isinstance(result, tuple) else (result, None)
    if priorities is not None:
        coordinator.run.sampler.update_priorities(priorities)
    coordinator.counters.updates += 1
    interval = min(25, run.updates)
    if index % interval == 0 or index == run.updates:
        logger.info("Offline pretraining progress: updates=%d/%d", index, run.updates)
    return values


def _mean_metrics(metrics: list[Mapping[str, float]]) -> dict[str, float]:
    keys = {key for values in metrics for key in values}
    return {key: fmean(float(values[key]) for values in metrics if key in values) for key in keys}


def _log_offline_pretraining(run: _OfflineRun, summary: Mapping[str, float]) -> None:
    coordinator = run.coordinator
    duration = perf_counter() - run.started_at
    payload = {
        **summary,
        "updates": run.updates,
        "replay_size": len(coordinator.run.replay_store),
        "duration_s": duration,
    }
    coordinator.run.logger.log("train/offline_pretrain", payload, step=coordinator.counters.updates)
    logger.info(
        "Offline pretraining complete: updates=%d, replay=%d, duration=%.1fs",
        run.updates,
        len(coordinator.run.replay_store),
        duration,
    )
