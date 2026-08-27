"""Local and offline training commands."""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time_ns
from typing import Any

from trackmaniarl.commands.common import (
    _new_attempt_spec,
    _resumed_attempt_spec,
    _spawn_context,
    _with_model_initialization_checkpoint,
)
from trackmaniarl.commands.distributed import _actor_process, _learner_process
from trackmaniarl.core.runtime import import_symbol, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator_types import (
    CoordinatorConfig,
    LearnerProcessRequest,
    ReplayRestoreMode,
)
from trackmaniarl.trackmania.demonstrations import resolve_demonstration_paths


@dataclass(frozen=True, slots=True)
class _LocalProcesses:
    learner: Any
    actor: Any
    shutdown: Any
    endpoint: str


@dataclass(frozen=True, slots=True)
class _TrainingPlan:
    config: Path
    configured_spec: RunSpec
    spec: RunSpec


@dataclass(frozen=True, slots=True)
class _LocalRuntime:
    config: Path
    endpoint: str
    token: str
    shutdown: Any


def _train(args: argparse.Namespace) -> None:
    plan = _training_plan(args)
    learner_factory = import_symbol(plan.spec.components.learner.class_path)
    if bool(getattr(learner_factory, "on_policy", False)):
        _train_on_policy(plan, args)
        return
    _train_asynchronously(plan, args)


def _training_plan(args: argparse.Namespace) -> _TrainingPlan:
    config = args.config.resolve()
    configured_spec = RunSpec.from_yaml(config)
    initialization = getattr(args, "model_initialization_checkpoint", None)
    source_spec = configured_spec
    if initialization is not None:
        source_spec = _with_model_initialization_checkpoint(
            configured_spec, initialization.resolve()
        )
    spec = _resumed_attempt_spec(config, source_spec, args)
    return _TrainingPlan(config, configured_spec, _new_attempt_spec(config, spec, args))


def _train_on_policy(plan: _TrainingPlan, args: argparse.Namespace) -> None:
    run = resolve_run(plan.spec, base_dir=plan.config.parent)
    try:
        result = Trainer(run, resume_checkpoint=getattr(args, "checkpoint", None)).train()
    finally:
        run.logger.close()
    print(
        f"Finished local on-policy run {plan.spec.run_id}: "
        f"transitions={result.transitions}, updates={result.updates}. "
        f"Artifacts: {run.run_dir}"
    )


def _train_asynchronously(plan: _TrainingPlan, args: argparse.Namespace) -> None:
    config, temporary = _materialize_run_spec(plan)
    processes = _spawn_local_processes(plan.spec, config, args)
    stopped_by_user = False
    try:
        stopped_by_user = _supervise_local_processes(processes)
    finally:
        _stop_local_processes(processes)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if stopped_by_user:
        return
    _raise_process_failures(processes)
    print(
        f"Finished async run {plan.spec.run_id}. "
        f"Artifacts: {config.parent / plan.spec.artifacts_dir}"
    )


def _materialize_run_spec(plan: _TrainingPlan) -> tuple[Path, Path | None]:
    if plan.spec == plan.configured_spec:
        return plan.config, None
    temporary = plan.config.with_name(f".trackmaniarl-{plan.spec.run_id}-{time_ns()}.yaml")
    temporary.write_text(plan.spec.to_yaml(), encoding="utf-8")
    return temporary, temporary


def _spawn_local_processes(
    spec: RunSpec, config: Path, args: argparse.Namespace
) -> _LocalProcesses:
    context = _spawn_context()
    shutdown = context.Event()
    endpoint = f"127.0.0.1:{spec.distributed.port}"
    token = secrets.token_urlsafe(32)
    runtime = _LocalRuntime(config, endpoint, token, shutdown)
    learner_request = _local_learner_request(runtime, args)
    actor_request = ActorProcessRequest(str(config), endpoint, "local-actor", token, shutdown)
    learner = context.Process(
        target=_learner_process, args=(learner_request,), name="trackmaniarl-learner"
    )
    actor = context.Process(
        target=_actor_process, args=(actor_request,), name="trackmaniarl-local-actor"
    )
    return _LocalProcesses(learner, actor, shutdown, endpoint)


def _local_learner_request(
    runtime: _LocalRuntime, args: argparse.Namespace
) -> LearnerProcessRequest:
    checkpoint = getattr(args, "checkpoint", None)
    demos = resolve_demonstration_paths(getattr(args, "demo", ()))
    return LearnerProcessRequest(
        str(runtime.config),
        runtime.endpoint,
        runtime.token,
        str(checkpoint) if checkpoint else None,
        (
            ReplayRestoreMode.LEARNER_ONLY
            if bool(getattr(args, "reset_replay", False))
            else ReplayRestoreMode.FULL
        ),
        runtime.shutdown,
        tuple(str(path) for path in demos),
    )


def _supervise_local_processes(processes: _LocalProcesses) -> bool:
    processes.learner.start()
    processes.actor.start()
    _print_process_launch(processes)
    try:
        _wait_for_processes(processes)
    except KeyboardInterrupt:
        print("Stopping async training; saving the learner checkpoint...", flush=True)
        return True
    return False


def _print_process_launch(processes: _LocalProcesses) -> None:
    print("Local async training launched:", flush=True)
    print(
        f"  learner_pid={processes.learner.pid}  gradient updates, replay, checkpoints",
        flush=True,
    )
    print(f"  actor_pid={processes.actor.pid}      TrackMania rollouts -> learner", flush=True)
    print(f"  endpoint={processes.endpoint}  gRPC; actor connects here", flush=True)


def _wait_for_processes(processes: _LocalProcesses) -> None:
    while processes.learner.is_alive() and processes.actor.is_alive():
        sleep(0.25)
    if processes.actor.is_alive() or not processes.learner.is_alive():
        return
    if processes.actor.exitcode == 0:
        _wait_for_learner_drain(processes)
        return
    print(
        f"Actor process (pid={processes.actor.pid}) exited first with "
        f"code={processes.actor.exitcode}; stopping learner "
        f"(pid={processes.learner.pid}) gracefully...",
        flush=True,
    )


def _wait_for_learner_drain(processes: _LocalProcesses) -> None:
    print(
        f"Actor process (pid={processes.actor.pid}) completed rollout collection; "
        f"waiting for learner (pid={processes.learner.pid}) to drain update credit...",
        flush=True,
    )
    while processes.learner.is_alive():
        sleep(0.25)


def _stop_local_processes(processes: _LocalProcesses) -> None:
    _signal_shutdown(processes.shutdown, processes.learner, processes.actor)
    processes.learner.join(timeout=10)
    processes.actor.join(timeout=10)
    for process in (processes.actor, processes.learner):
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)


def _signal_shutdown(shutdown: Any, *processes: Any) -> None:
    if any(process.is_alive() for process in processes):
        shutdown.set()


def _raise_process_failures(processes: _LocalProcesses) -> None:
    failures = [
        f"{name} process exited with code {process.exitcode}"
        for name, process in (("actor", processes.actor), ("learner", processes.learner))
        if process.exitcode not in (0, None)
    ]
    if failures:
        raise RuntimeError("; ".join(failures))


def _offline_pretrain(args: argparse.Namespace) -> None:
    config, spec, demonstrations = _offline_training_spec(args)
    result = _run_offline_pretraining(config, spec, demonstrations)
    checkpoint = result.checkpoints[-1]
    print(f"Offline pretraining complete: updates={result.updates}, checkpoint={checkpoint}")


def _offline_training_spec(args: argparse.Namespace) -> tuple[Path, RunSpec, tuple[Path, ...]]:
    config = args.config.resolve()
    spec = RunSpec.from_yaml(config)
    initialization = args.model_initialization_checkpoint
    if initialization is not None:
        spec = _with_model_initialization_checkpoint(spec, initialization.resolve())
    spec = _new_attempt_spec(config, spec, args)
    if spec.training.offline_pretrain_updates == 0:
        raise ValueError("offline-pretrain requires training.offline_pretrain_updates > 0")
    return config, spec, tuple(resolve_demonstration_paths(args.demo))


def _run_offline_pretraining(config: Path, spec: RunSpec, demonstrations: tuple[Path, ...]) -> Any:
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.protocol import run_fingerprint

    run = resolve_run(spec, base_dir=config.parent)
    try:
        coordinator = Coordinator(
            run,
            CoordinatorConfig(
                bind=f"127.0.0.1:{spec.distributed.port}",
                token=secrets.token_urlsafe(32),
                fingerprint=run_fingerprint(spec, config.parent),
                demo_paths=demonstrations,
            ),
        )
        return coordinator.run_offline_pretraining()
    finally:
        run.logger.close()
