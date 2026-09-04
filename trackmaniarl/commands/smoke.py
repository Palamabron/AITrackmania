"""Bounded live smoke command and checkpoint verification."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from time import time_ns
from typing import Any

from trackmaniarl.commands.training import _train
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.spec import RunSpec, TrainingSpec


@dataclass(frozen=True, slots=True)
class _SmokeRestore:
    coordinator: Any
    run: Any
    checkpoint: Path
    spec: RunSpec


def _smoke(args: argparse.Namespace) -> None:
    source = RunSpec.from_yaml(args.config)
    spec = _smoke_spec(source, args.transitions)
    base_dir = Path(args.config).resolve().parent
    temporary = base_dir / f".trackmaniarl-{spec.run_id}.yaml"
    temporary.write_text(spec.to_yaml(), encoding="utf-8")
    try:
        _train(argparse.Namespace(config=temporary, checkpoint=None))
        _restore_smoke_checkpoint(temporary, spec)
    finally:
        temporary.unlink(missing_ok=True)
    _require_policy_refresh(base_dir, spec)
    print("Async TrackMania smoke passed with a live policy-refresh interval of 0.25s.")


def _smoke_spec(spec: RunSpec, transitions: int) -> RunSpec:
    components = spec.components.model_copy(update={"evaluator": None})
    distributed = spec.distributed.model_copy(update={"policy_refresh_s": 0.25})
    return spec.model_copy(
        update={
            "run_id": f"{spec.run_id}-smoke-{time_ns()}",
            "training": _smoke_training(spec.training, transitions),
            "components": components,
            "distributed": distributed,
            "evaluation": None,
        }
    )


def _require_policy_refresh(base_dir: Path, spec: RunSpec) -> None:
    events_path = base_dir / spec.artifacts_dir / spec.run_id / "events.jsonl"
    events = _read_events(events_path)
    refreshed = any(
        event.get("event") == "distributed/policy_published"
        and int(event.get("payload", {}).get("policy_version", 0)) > 0
        for event in events
    )
    if not refreshed:
        raise RuntimeError("async smoke completed without refreshing the actor policy")


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _smoke_training(spec: TrainingSpec, transitions: int) -> TrainingSpec:
    if transitions < 8:
        raise ValueError("smoke testing requires at least 8 transitions")
    n_step = min(spec.n_step, transitions)
    batch_size = _smoke_batch_size(spec, transitions, n_step)
    ready = batch_size * spec.sequence_length + n_step - 1
    updates = {
        "total_transitions": transitions,
        "max_episode_steps": min(spec.max_episode_steps, transitions),
        "batch_size": batch_size,
        "n_step": n_step,
        "warmup_transitions": ready,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": transitions,
    }
    return spec.model_copy(update=updates | _disabled_evaluation_schedule())


def _disabled_evaluation_schedule() -> dict[str, None]:
    return {
        "evaluate_every_episodes": None,
        "evaluation_stop_min_finish_rate": None,
        "evaluation_stop_median_s": None,
        "evaluation_stop_consecutive_batches": None,
    }


def _smoke_batch_size(spec: TrainingSpec, transitions: int, n_step: int) -> int:
    available = transitions - n_step + 1
    capacity = available // spec.sequence_length
    if capacity < 2:
        minimum = spec.sequence_length + n_step
        raise ValueError(f"transitions must be at least {minimum} for a smoke learner update")
    return min(spec.batch_size, max(1, capacity // 2))


def _restore_smoke_checkpoint(config: Path, spec: RunSpec) -> None:
    checkpoint = _latest_smoke_checkpoint(config, spec)
    restore_spec = _smoke_restore_spec(spec)
    run = resolve_run(restore_spec, base_dir=config.parent)
    coordinator = _smoke_coordinator(config, restore_spec, run)
    try:
        _restore_checkpoint(_SmokeRestore(coordinator, run, checkpoint, restore_spec))
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()
        run.logger.close()


def _smoke_coordinator(config: Path, spec: RunSpec, run: Any) -> Any:
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
    from trackmaniarl.distributed.protocol import run_fingerprint

    return Coordinator(
        run,
        CoordinatorConfig(
            bind="127.0.0.1:8787",
            token=token_urlsafe(32),
            fingerprint=run_fingerprint(spec, config.parent),
        ),
    )


def _latest_smoke_checkpoint(config: Path, spec: RunSpec) -> Path:
    directory = config.parent / spec.artifacts_dir / spec.run_id / "checkpoints"
    checkpoints = sorted(directory.glob("distributed-update-*.pt"))
    if not checkpoints:
        raise RuntimeError("async smoke did not produce a distributed checkpoint")
    return checkpoints[-1]


def _smoke_restore_spec(spec: RunSpec) -> RunSpec:
    components = spec.components.model_copy(update={"additional_loggers": ()})
    return spec.model_copy(update={"components": components})


def _restore_checkpoint(restore: _SmokeRestore) -> None:
    run = restore.run
    run.learner.setup(
        {
            "seed": restore.spec.seed,
            "run_dir": run.run_dir,
            "model_factory": run.model_factory,
            "restoring_checkpoint": True,
        }
    )
    restore.coordinator.restore_checkpoint(restore.checkpoint)
    if restore.coordinator.counters.updates < 1:
        raise RuntimeError("async smoke checkpoint contains no learner updates")
