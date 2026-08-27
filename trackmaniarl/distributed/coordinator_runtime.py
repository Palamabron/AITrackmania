from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from trackmaniarl.core.runtime import prepare_run, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import TrainingResult
from trackmaniarl.distributed.coordinator_policy import PolicyPublicationMode
from trackmaniarl.distributed.coordinator_types import (
    CoordinatorConfig,
    LearnerProcessRequest,
    ReplayRestoreMode,
)
from trackmaniarl.distributed.protocol import run_fingerprint

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")


def run_forever(coordinator: Coordinator) -> TrainingResult:
    try:
        return _run_forever(coordinator)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        coordinator._log_run_failure("distributed_training", exc)
        raise
    finally:
        coordinator._close_runtime()


def _run_forever(coordinator: Coordinator) -> TrainingResult:
    coordinator._prepare_training()
    _restore_or_validate_run(coordinator)
    coordinator._import_demonstrations()
    if coordinator.resume_checkpoint is None or coordinator.demo_paths:
        coordinator._offline_pretrain()
    coordinator._publish_policy(PolicyPublicationMode.FORCED)
    coordinator._start_server()
    _log_runtime_ready(coordinator)
    try:
        return _run_learning(coordinator)
    except KeyboardInterrupt:
        _save_interrupted_checkpoint(coordinator)
        raise


def _restore_or_validate_run(coordinator: Coordinator) -> None:
    if coordinator.resume_checkpoint is not None:
        _restore_checkpoint(coordinator)
        return
    if coordinator.journal.has_history():
        raise RuntimeError(
            f"run_id {coordinator.run.spec.run_id!r} has prior rollout data in "
            f"{coordinator.journal.path}; resume with --checkpoint or choose a new run_id"
        )


def _restore_checkpoint(coordinator: Coordinator) -> None:
    path = coordinator.resume_checkpoint
    if path is None:
        raise RuntimeError("checkpoint restore requested without a checkpoint path")
    logger.info("Restoring checkpoint: %s", path)
    coordinator.restore_checkpoint(path, coordinator.restore_mode)
    restored = "learner state only; replay and runtime counters reset"
    if coordinator.restore_mode is ReplayRestoreMode.FULL:
        restored = "full state"
    logger.info(
        "Checkpoint restored (%s): transitions=%d, updates=%d",
        restored,
        coordinator.counters.transitions,
        coordinator.counters.updates,
    )


def _log_runtime_ready(coordinator: Coordinator) -> None:
    logger.info(
        "Async learner ready (pid=%d): run_id=%s, gRPC bind=%s, target_transitions=%d",
        os.getpid(),
        coordinator.run.spec.run_id,
        coordinator.bind,
        coordinator.run.spec.training.total_transitions,
    )


def _run_learning(coordinator: Coordinator) -> TrainingResult:
    coordinator._learn()
    _save_final_checkpoint(coordinator)
    coordinator._checkpoint_writer.wait()
    return _training_result(coordinator, coordinator._checkpoints)


def _save_interrupted_checkpoint(coordinator: Coordinator) -> None:
    _save_final_checkpoint(coordinator)
    coordinator._checkpoint_writer.wait()


def _save_final_checkpoint(coordinator: Coordinator) -> None:
    if coordinator.run.spec.training.save_final_checkpoint:
        coordinator._checkpoints.append(coordinator._checkpoint())


def _training_result(coordinator: Coordinator, checkpoints: list[Path]) -> TrainingResult:
    return TrainingResult(
        coordinator.counters.episodes,
        coordinator.counters.transitions,
        coordinator.counters.updates,
        tuple(checkpoints),
        None,
    )


def run_offline_pretraining(coordinator: Coordinator) -> TrainingResult:
    """Train only from configured demonstrations without opening the actor server."""

    try:
        return _run_offline_pretraining(coordinator)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        coordinator._log_run_failure("offline_pretraining", exc)
        raise
    finally:
        coordinator._close_runtime()


def _run_offline_pretraining(coordinator: Coordinator) -> TrainingResult:
    _validate_offline_pretraining(coordinator)
    checkpoints: list[Path] = []
    coordinator._prepare_training()
    _validate_empty_journal(coordinator)
    coordinator._import_demonstrations()
    coordinator._offline_pretrain()
    if coordinator.run.spec.training.save_final_checkpoint:
        checkpoints.append(coordinator._checkpoint())
    coordinator._checkpoint_writer.wait()
    return _training_result(coordinator, checkpoints)


def _validate_offline_pretraining(coordinator: Coordinator) -> None:
    if coordinator.run.spec.training.offline_pretrain_updates == 0:
        raise ValueError("offline pretraining requires offline_pretrain_updates > 0")
    if not coordinator.demo_paths:
        raise ValueError("offline pretraining requires at least one demonstration")


def _validate_empty_journal(coordinator: Coordinator) -> None:
    if coordinator.journal.has_history():
        raise RuntimeError(
            f"run_id {coordinator.run.spec.run_id!r} has prior rollout data in "
            f"{coordinator.journal.path}; choose a new run_id"
        )


def log_run_failure(coordinator: Coordinator, phase: str, exc: BaseException) -> None:
    coordinator.run.logger.log(
        "run/failure",
        {
            "phase": phase,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
        step=coordinator.counters.updates,
    )


def close_runtime(coordinator: Coordinator) -> None:
    if coordinator._server is not None:
        coordinator._server.stop(grace=2).wait(timeout=5)
    if coordinator._rpc_executor is not None:
        coordinator._rpc_executor.shutdown(wait=True, cancel_futures=True)
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def prepare_training(coordinator: Coordinator) -> None:
    coordinator.run.learner.setup(
        {
            "seed": coordinator.run.spec.seed,
            "run_dir": coordinator.run.run_dir,
            "model_factory": coordinator.run.model_factory,
        }
    )
    prepare_run(coordinator.run)
    coordinator._log_execution()


def learner_process_entry(request: LearnerProcessRequest) -> None:
    """Spawn-safe learner entrypoint used by both local and remote launchers."""

    _run_learner_process(request)


def _run_learner_process(request: LearnerProcessRequest) -> None:
    from trackmaniarl.distributed.coordinator import Coordinator

    path = Path(request.config_path).resolve()
    spec = RunSpec.from_yaml(path)
    run = resolve_run(spec, base_dir=path.parent)
    try:
        Coordinator(run, _coordinator_config(request, path, spec)).run_forever()
    finally:
        run.logger.close()


def _coordinator_config(
    request: LearnerProcessRequest, path: Path, spec: RunSpec
) -> CoordinatorConfig:
    checkpoint = Path(request.resume_checkpoint) if request.resume_checkpoint else None
    return CoordinatorConfig(
        bind=request.bind,
        token=request.token,
        fingerprint=run_fingerprint(spec, path.parent),
        resume_checkpoint=checkpoint,
        restore_mode=request.restore_mode,
        external_stop=request.external_stop,
        demo_paths=tuple(Path(item) for item in request.demo_paths),
    )
