from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from trackmaniarl.core.checkpoints import prune_checkpoint_family
from trackmaniarl.distributed.coordinator_support import (
    _CheckpointWrite,
    _Counters,
    load_state_dict,
    snapshot_value,
    state_dict,
)
from trackmaniarl.distributed.coordinator_types import ReplayRestoreMode

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator

logger = logging.getLogger("trackmaniarl.distributed.coordinator")

_CHECKPOINT_SCHEMA_VERSION = "2.0"
_JOURNAL_CONTRACT_VERSION = 2
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "journal_contract_version",
        "journal_id",
        "run_fingerprint",
        "learner",
        "replay_store",
        "sampler",
        "distributed",
        "evaluated_policy_version",
    }
)
_DISTRIBUTED_KEYS = frozenset(
    {
        "transitions",
        "episodes",
        "finishes",
        "best_finish_time_s",
        "evaluations",
        "evaluation_finishes",
        "evaluation_bucket_finishes",
        "updates",
        "update_credit",
        "journal_applied_frontier",
        "policy_version",
        "actor_sequences",
    }
)
_DISTRIBUTED_OPTIONAL_KEYS = frozenset(
    {
        "best_evaluation_rank",
        "fastest_evaluation_rank",
    }
)


class EvaluationCheckpointKind(StrEnum):
    RELIABLE = "best-eval"
    FASTEST = "fastest-eval"


@dataclass(frozen=True, slots=True)
class EvaluatedPolicyCheckpoint:
    state: Mapping[str, Any]
    version: int
    kind: EvaluationCheckpointKind
    on_saved: Callable[[Path], None] | None = None
    on_failed: Callable[[BaseException], None] | None = None


@dataclass(frozen=True, slots=True)
class _CheckpointPlan:
    coordinator: Coordinator
    evaluated_policy: EvaluatedPolicyCheckpoint | None
    path: Path
    started_at: float


@dataclass(frozen=True, slots=True)
class _CheckpointCallbacks:
    plan: _CheckpointPlan
    frontier: int
    update: int

    def saved(self) -> None:
        coordinator = self.plan.coordinator
        try:
            coordinator.journal.prune(self.frontier)
        except Exception as exc:
            coordinator._log_wal_error("prune", exc)
            raise
        _apply_checkpoint_retention(self.plan, self.update)
        _log_checkpoint_completed(self)
        evaluated = self.plan.evaluated_policy
        if evaluated is not None and evaluated.on_saved is not None:
            evaluated.on_saved(self.plan.path)

    def failed(self, exc: BaseException) -> None:
        coordinator = self.plan.coordinator
        coordinator.run.logger.log(
            "train/checkpoint_failed",
            {
                "path": str(self.plan.path),
                "journal_applied_frontier": self.frontier,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            step=self.update,
        )
        evaluated = self.plan.evaluated_policy
        if evaluated is not None and evaluated.on_failed is not None:
            evaluated.on_failed(exc)


def _log_checkpoint_completed(callbacks: _CheckpointCallbacks) -> None:
    plan = callbacks.plan
    plan.coordinator.run.logger.log(
        "train/checkpoint_completed",
        {
            "path": str(plan.path),
            "journal_applied_frontier": callbacks.frontier,
            "duration_s": perf_counter() - plan.started_at,
        },
        step=callbacks.update,
    )


def _apply_checkpoint_retention(plan: _CheckpointPlan, update: int) -> None:
    coordinator = plan.coordinator
    keep = coordinator.run.spec.training.checkpoint_keep_last
    if keep is None:
        return
    result = prune_checkpoint_family(plan.path, coordinator.run.run_dir / "checkpoints", keep)
    if not result.removed:
        return
    removed = [str(path) for path in result.removed]
    coordinator.run.logger.log(
        "train/checkpoint_retention",
        {"family": result.family, "removed_count": len(removed), "paths": removed},
        step=update,
    )
    logger.info("Checkpoint retention removed: %s", ", ".join(removed))


@dataclass(frozen=True, slots=True)
class _RestoreRequest:
    coordinator: Coordinator
    path: Path
    mode: ReplayRestoreMode


def checkpoint(
    coordinator: Coordinator,
    evaluated_policy: EvaluatedPolicyCheckpoint | None = None,
) -> Path:
    coordinator._checkpoint_writer.wait()
    plan = _checkpoint_plan(coordinator, evaluated_policy)
    state = _checkpoint_state(plan)
    callbacks = _CheckpointCallbacks(
        plan, coordinator.counters.journal_applied_frontier, coordinator.counters.updates
    )
    coordinator._checkpoint_writer.submit(
        _CheckpointWrite(state, plan.path, callbacks.saved, callbacks.failed)
    )
    _log_checkpoint_queued(plan)
    return plan.path


def _checkpoint_plan(
    coordinator: Coordinator,
    evaluated_policy: EvaluatedPolicyCheckpoint | None,
) -> _CheckpointPlan:
    if evaluated_policy is not None:
        name = (
            f"{evaluated_policy.kind.value}-policy-{evaluated_policy.version:08d}-at-update-"
            f"{coordinator.counters.updates:08d}.pt"
        )
    else:
        name = f"distributed-update-{coordinator.counters.updates:08d}.pt"
    path = coordinator.run.run_dir / "checkpoints" / name
    return _CheckpointPlan(coordinator, evaluated_policy, path, perf_counter())


def _checkpoint_state(plan: _CheckpointPlan) -> dict[str, Any]:
    coordinator = plan.coordinator
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "journal_contract_version": _JOURNAL_CONTRACT_VERSION,
        "journal_id": coordinator.journal.identity,
        "run_fingerprint": coordinator.fingerprint,
        "learner": snapshot_value(_learner_state(plan)),
        "replay_store": snapshot_value(state_dict(coordinator.run.replay_store)),
        "sampler": snapshot_value(state_dict(coordinator.run.sampler)),
        "distributed": _distributed_state(coordinator),
        "evaluated_policy_version": (
            plan.evaluated_policy.version if plan.evaluated_policy is not None else None
        ),
    }


def _learner_state(plan: _CheckpointPlan) -> Mapping[str, Any]:
    learner = plan.coordinator.run.learner
    if plan.evaluated_policy is None:
        return learner.state_dict()
    exact_state = getattr(learner, "state_dict_for_policy", None)
    if not callable(exact_state):
        raise TypeError("learner cannot build an exact evaluated-policy checkpoint")
    return cast(Mapping[str, Any], exact_state(plan.evaluated_policy.state))


def _distributed_state(coordinator: Coordinator) -> dict[str, Any]:
    counters = coordinator.counters
    return {
        "transitions": counters.transitions,
        "episodes": counters.episodes,
        "finishes": counters.finishes,
        "best_finish_time_s": counters.best_finish_time_s,
        "evaluations": counters.evaluations,
        "evaluation_finishes": counters.evaluation_finishes,
        "evaluation_bucket_finishes": dict(counters.evaluation_bucket_finishes),
        "updates": counters.updates,
        "update_credit": counters.update_credit,
        "journal_applied_frontier": counters.journal_applied_frontier,
        "policy_version": counters.policy_version,
        "actor_sequences": dict(counters.actor_sequences),
        "best_evaluation_rank": coordinator._best_evaluation,
        "fastest_evaluation_rank": coordinator._fastest_evaluation,
    }


def _log_checkpoint_queued(plan: _CheckpointPlan) -> None:
    plan.coordinator.run.logger.log(
        "train/checkpoint",
        {
            "path": str(plan.path),
            "timing/checkpoint_snapshot_s": perf_counter() - plan.started_at,
        },
        step=plan.coordinator.counters.updates,
    )
    logger.info("Checkpoint queued: %s", plan.path)


def restore_checkpoint(
    coordinator: Coordinator,
    path: Path,
    mode: ReplayRestoreMode = ReplayRestoreMode.FULL,
) -> None:
    request = _RestoreRequest(coordinator, path, mode)
    state = coordinator.run.checkpoint_codec.load(path)
    _validate_checkpoint_state(request, state)
    if request.mode is ReplayRestoreMode.LEARNER_ONLY:
        _restore_learner_only(request, state)
        return
    distributed = _validated_distributed_state(request, state)
    _restore_distributed(request, state, distributed)


def _validate_checkpoint_state(request: _RestoreRequest, state: Mapping[str, Any]) -> None:
    _validate_checkpoint_keys(state)
    if state["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("async runtime requires distributed checkpoint schema 2.0")
    if state["journal_contract_version"] != _JOURNAL_CONTRACT_VERSION:
        raise ValueError("distributed checkpoint requires journal contract version 2")
    if state["run_fingerprint"] != request.coordinator.fingerprint:
        raise ValueError("distributed checkpoint run fingerprint mismatch")
    _validate_checkpoint_components(state)


def _validate_checkpoint_keys(state: Mapping[str, Any]) -> None:
    missing = _CHECKPOINT_KEYS - state.keys()
    unexpected = state.keys() - _CHECKPOINT_KEYS
    if missing or unexpected:
        raise ValueError(
            f"distributed checkpoint keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _validate_checkpoint_components(state: Mapping[str, Any]) -> None:
    for name in ("learner", "replay_store", "sampler", "distributed"):
        if not isinstance(state[name], Mapping):
            raise TypeError(f"distributed checkpoint {name} state must be a mapping")
    if not isinstance(state["journal_id"], str) or not state["journal_id"]:
        raise TypeError("distributed checkpoint journal_id must be a non-empty string")
    version = state["evaluated_policy_version"]
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        raise TypeError("evaluated policy version must be an integer or null")


def _restore_learner_only(request: _RestoreRequest, state: Mapping[str, Any]) -> None:
    coordinator = request.coordinator
    if coordinator.journal.has_history():
        raise RuntimeError(
            f"cannot reset replay while {coordinator.journal.path} contains rollout data; "
            "choose a new run_id so stale journal rows cannot enter a later resume"
        )
    coordinator.run.learner.load_state_dict(state["learner"])
    coordinator.counters = _Counters()


def _validated_distributed_state(
    request: _RestoreRequest, state: Mapping[str, Any]
) -> dict[str, Any]:
    distributed = dict(state["distributed"])
    _validate_distributed_keys(distributed)
    _validate_journal(request.coordinator, state, distributed)
    return distributed


def _validate_distributed_keys(distributed: Mapping[str, Any]) -> None:
    missing = _DISTRIBUTED_KEYS - distributed.keys()
    unexpected = distributed.keys() - _DISTRIBUTED_KEYS - _DISTRIBUTED_OPTIONAL_KEYS
    if missing or unexpected:
        raise ValueError(
            f"distributed runtime keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _validate_journal(
    coordinator: Coordinator, state: Mapping[str, Any], distributed: Mapping[str, Any]
) -> None:
    frontier = int(distributed["journal_applied_frontier"])
    try:
        coordinator.journal.validate_checkpoint(state["journal_id"], frontier)
    except Exception as exc:
        coordinator._log_wal_error("checkpoint_validation", exc)
        raise


def _restore_distributed(
    request: _RestoreRequest, state: Mapping[str, Any], distributed: dict[str, Any]
) -> None:
    coordinator = request.coordinator
    coordinator.run.learner.load_state_dict(state["learner"])
    coordinator._best_evaluation = _evaluation_rank(distributed.pop("best_evaluation_rank", None))
    coordinator._fastest_evaluation = _evaluation_rank(
        distributed.pop("fastest_evaluation_rank", None)
    )
    coordinator.counters = _Counters(**distributed)
    coordinator.counters.update_credit = min(
        coordinator.counters.update_credit,
        float(coordinator.run.spec.distributed.max_update_credit),
    )
    load_state_dict(coordinator.run.replay_store, state["replay_store"])
    load_state_dict(coordinator.run.sampler, state["sampler"])
    coordinator._recover_journal(coordinator.counters.journal_applied_frontier)


def _evaluation_rank(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError("evaluation leader rank must contain three numbers")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise TypeError("evaluation leader rank must contain three numbers")
    rank = tuple(float(item) for item in value)
    if not all(map(math.isfinite, rank)):
        raise ValueError("evaluation leader rank must contain three finite numbers")
    return rank[0], rank[1], rank[2]
