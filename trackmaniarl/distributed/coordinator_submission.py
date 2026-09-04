"""Durable rollout submission handling for coordinator RPCs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from queue import Full
from time import monotonic
from typing import TYPE_CHECKING, Any

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.distributed.codec import WirePayloadTooLargeError
from trackmaniarl.distributed.coordinator_support import _RolloutRejection, snapshot_value
from trackmaniarl.distributed.coordinator_validation import _validate_submit_payload
from trackmaniarl.distributed.journal import JournalPayloadConflictError

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator


@dataclass(frozen=True, slots=True)
class _Submission:
    coordinator: Coordinator
    value: Mapping[str, Any]
    lag: int
    stop: bool


@dataclass(frozen=True, slots=True)
class _EvaluationAssignment:
    evaluate: bool
    policy_version: int
    snapshot: bytes


@dataclass(frozen=True, slots=True)
class _AcceptedSubmission:
    submission: _Submission
    inserted: bool
    evaluation: _EvaluationAssignment
    context: grpc.ServicerContext[Any, Any]


def submit(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> BytesValue:
    value = coordinator._request(message, context)
    _validate_submission(coordinator, value, context)
    submission = _submission(coordinator, value)
    if _hard_lagged(submission):
        return _reject_hard_lag(submission, context)
    row_id, inserted = _append_journal(submission, message, context)
    if inserted:
        _wake_learner(coordinator, row_id)
    evaluation = _assign_evaluation(coordinator, value)
    return _accepted_response(_AcceptedSubmission(submission, inserted, evaluation, context))


def _validate_submission(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    context: grpc.ServicerContext[Any, Any],
) -> None:
    try:
        _validate_submit_payload(value, coordinator.codec)
    except WirePayloadTooLargeError as exc:
        context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "distributed request exceeds the configured size limit",
        )
        raise AssertionError("gRPC abort returned") from exc
    except (KeyError, TypeError, ValueError) as exc:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid rollout payload")
        raise AssertionError("gRPC abort returned") from exc


def _submission(coordinator: Coordinator, value: Mapping[str, Any]) -> _Submission:
    policy_version = int(value["policy_version"])
    with coordinator._lock:
        lag = max(0, coordinator.counters.updates - policy_version)
        stop = coordinator._should_stop()
    return _Submission(coordinator, value, lag, stop)


def _hard_lagged(submission: _Submission) -> bool:
    maximum = submission.coordinator.run.spec.distributed.hard_policy_lag_updates
    return bool(submission.value["transitions"]) and submission.lag > maximum


def _reject_hard_lag(
    submission: _Submission, context: grpc.ServicerContext[Any, Any]
) -> BytesValue:
    coordinator = submission.coordinator
    maximum = coordinator.run.spec.distributed.hard_policy_lag_updates
    coordinator._log_rollout_rejected(
        submission.value,
        _RolloutRejection(
            "hard_policy_lag",
            {"policy_lag_updates": submission.lag, "hard_policy_lag_updates": maximum},
        ),
    )
    return coordinator._response(_hard_lag_payload(submission), context)


def _hard_lag_payload(submission: _Submission) -> dict[str, object]:
    coordinator = submission.coordinator
    return {
        "accepted": False,
        "reason": "hard_policy_lag",
        "force_refresh": True,
        "stop": submission.stop,
        "policy_lag_updates": submission.lag,
        "evaluate": False,
        "evaluation_policy_version": coordinator.counters.policy_version,
        "evaluation_snapshot": b"",
    }


def _append_journal(
    submission: _Submission,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> tuple[int, bool]:
    coordinator = submission.coordinator
    value = submission.value
    try:
        return coordinator.journal.append(value["session_id"], value["sequence"], message.value)
    except JournalPayloadConflictError as exc:
        coordinator._log_rollout_rejected(value, _RolloutRejection("payload_conflict"))
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise AssertionError("gRPC abort returned") from exc
    except Exception as exc:
        coordinator._log_wal_error("append", exc)
        raise


def _wake_learner(coordinator: Coordinator, row_id: int) -> None:
    with suppress(Full):
        coordinator._rollouts.put_nowait((row_id, monotonic()))


def _assign_evaluation(coordinator: Coordinator, value: Mapping[str, Any]) -> _EvaluationAssignment:
    with coordinator._lock:
        actor_id = str(value["actor_id"])
        evaluate = actor_id in coordinator._evaluation_due
        evaluate = evaluate and coordinator.counters.policy_version > 0
        if evaluate:
            coordinator._evaluation_due.discard(actor_id)
        version = coordinator.counters.policy_version
        snapshot = coordinator._policy_payload if evaluate else b""
        if evaluate:
            _remember_evaluation_policy(coordinator, version, snapshot)
    return _EvaluationAssignment(evaluate, version, snapshot)


def _remember_evaluation_policy(coordinator: Coordinator, version: int, snapshot: bytes) -> None:
    policy_state = coordinator.codec.decode(snapshot)
    if not isinstance(policy_state, Mapping):
        raise ValueError("published policy snapshot must decode to a mapping")
    coordinator._evaluation_policy_states[version] = snapshot_value(policy_state)
    while len(coordinator._evaluation_policy_states) > 16:
        oldest = next(iter(coordinator._evaluation_policy_states))
        coordinator._evaluation_policy_states.pop(oldest)


def _accepted_response(accepted: _AcceptedSubmission) -> BytesValue:
    submission = accepted.submission
    evaluation = accepted.evaluation
    coordinator = submission.coordinator
    payload = {
        "accepted": True,
        "duplicate": not accepted.inserted,
        "force_refresh": submission.lag > coordinator.run.spec.distributed.soft_policy_lag_updates,
        "stop": submission.stop,
        "policy_lag_updates": submission.lag,
        "evaluate": evaluation.evaluate,
        "evaluation_policy_version": evaluation.policy_version,
        "evaluation_snapshot": evaluation.snapshot,
    }
    return coordinator._response(payload, accepted.context)
