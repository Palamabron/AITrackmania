"""Required background workers for distributed actors."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from time import monotonic
from typing import Any, Protocol

import grpc

from trackmaniarl.core.contracts import ExploratoryPolicy, ReplicablePolicy
from trackmaniarl.distributed.actor_errors import ActorBackgroundError
from trackmaniarl.distributed.actor_spool import SpoolRuntime
from trackmaniarl.distributed.actor_transport import (
    Client,
    PolicyReference,
    is_retryable_rpc_error,
)

logger = logging.getLogger(__name__)


class BackgroundRuntime(SpoolRuntime, Protocol):
    target: str
    stop_reason: str
    force_refresh: threading.Event
    evaluate: threading.Event
    external_stop: Any | None
    client: Client
    _policy_ref: PolicyReference | None
    _evaluation_request: tuple[bytes, int] | None
    _evaluation_request_lock: Any
    _background_failure: ActorBackgroundError | None
    _background_failure_lock: Any

    def _policy(self) -> tuple[ReplicablePolicy, float, int]: ...

    def _new_policy(self) -> ReplicablePolicy: ...

    def _refresh_policy(self) -> None: ...

    def _sender_loop(self) -> None: ...

    def _policy_loop(self) -> None: ...

    def _heartbeat_loop(self) -> None: ...

    def _external_stop_loop(self) -> None: ...

    def _send_spooled_rollouts(self) -> None: ...

    def _discard_spooled(self, path: Path, size: int) -> None: ...

    def _stop_from_thread(self, stage: str, exc: BaseException) -> None: ...


@dataclass(frozen=True, slots=True)
class SubmittedRollout:
    path: Path
    size: int
    response: Mapping[str, Any]


def register(runtime: BackgroundRuntime) -> Mapping[str, Any]:
    while not runtime.stop.is_set():
        try:
            return runtime.client.call("Register", runtime._request_base())
        except grpc.RpcError as exc:
            if not is_retryable_rpc_error(exc):
                raise RuntimeError(f"actor registration rejected: {exc.details()}") from exc
            logger.info(
                "Actor %s (pid=%d): learner not ready at %s; retrying...",
                runtime.actor_id,
                os.getpid(),
                runtime.target,
            )
            runtime.stop.wait(1.0)
    raise RuntimeError("actor stopped before registering")


def start_background_workers(runtime: BackgroundRuntime) -> list[threading.Thread]:
    senders = _sender_workers(runtime)
    workers = [*senders, *_service_workers(runtime)]
    for worker in workers:
        worker.start()
    return senders


def _sender_workers(runtime: BackgroundRuntime) -> list[threading.Thread]:
    return [
        threading.Thread(
            target=runtime._sender_loop,
            name=f"trackmaniarl-rollout-sender-{index}",
            daemon=True,
        )
        for index in range(runtime.spec.distributed.max_inflight_chunks)
    ]


def _service_workers(runtime: BackgroundRuntime) -> list[threading.Thread]:
    return [
        threading.Thread(
            target=runtime._policy_loop,
            name="trackmaniarl-policy-refresh",
            daemon=True,
        ),
        threading.Thread(
            target=runtime._heartbeat_loop,
            name="trackmaniarl-actor-heartbeat",
            daemon=True,
        ),
        threading.Thread(
            target=runtime._external_stop_loop,
            name="trackmaniarl-actor-shutdown",
            daemon=True,
        ),
    ]


def sender_loop(runtime: BackgroundRuntime) -> None:
    try:
        runtime._send_spooled_rollouts()
    except Exception as exc:
        runtime._stop_from_thread("rollout sender", exc)


def send_spooled_rollouts(runtime: BackgroundRuntime) -> None:
    while not runtime.stop.is_set() or not runtime.queue.empty():
        try:
            path = runtime.queue.get(timeout=0.2)
        except Empty:
            continue
        _send_spooled_path(runtime, path)
        runtime.queue.task_done()


def _send_spooled_path(runtime: BackgroundRuntime, path: Path) -> None:
    while path.exists():
        try:
            payload = path.read_bytes()
            request = runtime.codec.decode(payload)
            response = runtime.client.call("Submit", request)
            submitted = SubmittedRollout(path, len(payload), response)
            if _handle_submit_response(runtime, submitted):
                break
        except grpc.RpcError as exc:
            if not is_retryable_rpc_error(exc):
                raise
            if runtime.stop.wait(1.0):
                break


def _handle_submit_response(runtime: BackgroundRuntime, submitted: SubmittedRollout) -> bool:
    response = submitted.response
    _handle_learner_signals(runtime, response)
    if response["accepted"]:
        runtime._discard_spooled(submitted.path, submitted.size)
        return False
    return _handle_rejected_rollout(runtime, submitted)


def _handle_learner_signals(runtime: BackgroundRuntime, response: Mapping[str, Any]) -> None:
    if response["force_refresh"]:
        runtime.force_refresh.set()
    if response["evaluate"]:
        _accept_evaluation_request(runtime, response)
    if response["stop"]:
        runtime.stop_reason = "learner requested stop"
        runtime.stop.set()


def _handle_rejected_rollout(runtime: BackgroundRuntime, submitted: SubmittedRollout) -> bool:
    response = submitted.response
    if response["reason"] != "hard_policy_lag":
        raise ValueError(f"unsupported rollout rejection: {response['reason']!r}")
    runtime._discard_spooled(submitted.path, submitted.size)
    runtime.force_refresh.set()
    return True


def _accept_evaluation_request(runtime: BackgroundRuntime, response: Mapping[str, Any]) -> None:
    snapshot = response["evaluation_snapshot"]
    version = response["evaluation_policy_version"]
    if not isinstance(snapshot, bytes) or not snapshot or version < 0:
        raise ValueError("evaluation request requires a policy snapshot/version")
    with runtime._evaluation_request_lock:
        runtime._evaluation_request = (snapshot, version)
    runtime.evaluate.set()


def policy_loop(runtime: BackgroundRuntime) -> None:
    refresh_at = monotonic()
    while not runtime.stop.wait(0.1):
        if not runtime.force_refresh.is_set() and monotonic() < refresh_at:
            continue
        try:
            runtime._refresh_policy()
            runtime.force_refresh.clear()
            refresh_at = monotonic() + runtime.spec.distributed.policy_refresh_s
        except grpc.RpcError as exc:
            if is_retryable_rpc_error(exc):
                continue
            runtime._stop_from_thread("policy refresh", exc)
            return
        except Exception as exc:
            runtime._stop_from_thread("policy refresh", exc)
            return


def refresh_policy(runtime: BackgroundRuntime) -> None:
    _, _, current = runtime._policy()
    response = runtime.client.call(
        "Policy", {**runtime._request_base(), "current_version": current}
    )
    if response["stop"]:
        runtime.stop_reason = "learner requested stop"
        runtime.stop.set()
    epsilon = float(response["epsilon"])
    version = int(response["policy_version"])
    policy = _refreshed_policy(runtime, response["snapshot"])
    if isinstance(policy, ExploratoryPolicy):
        policy.set_exploration_epsilon(epsilon)
    if runtime._policy_ref is None:
        raise RuntimeError("actor policy is not initialized")
    runtime._policy_ref.replace(policy, epsilon, version)


def _refreshed_policy(runtime: BackgroundRuntime, snapshot: Any) -> ReplicablePolicy:
    policy = runtime._new_policy()
    if snapshot:
        state = runtime.codec.decode(snapshot)
        if not isinstance(state, Mapping):
            raise ValueError("policy snapshot must decode to a mapping")
        policy.load_state(state)
    else:
        current_policy, _, _ = runtime._policy()
        policy.load_state(current_policy.export_state())
    return policy


def heartbeat_loop(runtime: BackgroundRuntime) -> None:
    while not runtime.stop.wait(runtime.spec.distributed.heartbeat_s):
        try:
            _send_heartbeat(runtime)
        except grpc.RpcError as exc:
            if is_retryable_rpc_error(exc):
                continue
            runtime._stop_from_thread("heartbeat", exc)
            return
        except Exception as exc:
            runtime._stop_from_thread("heartbeat", exc)
            return


def _send_heartbeat(runtime: BackgroundRuntime) -> None:
    _, _, version = runtime._policy()
    response = runtime.client.call(
        "Heartbeat",
        {
            **runtime._request_base(),
            "policy_version": version,
            "spool_bytes": runtime._current_spool_bytes(),
        },
    )
    if response["stop"]:
        runtime.stop_reason = "learner requested stop"
        runtime.stop.set()


def stop_from_thread(runtime: BackgroundRuntime, stage: str, exc: BaseException) -> None:
    logger.exception("Actor %s %s failed; stopping the actor", runtime.actor_id, stage)
    failure = ActorBackgroundError(stage, exc)
    with runtime._background_failure_lock:
        if runtime._background_failure is None:
            runtime._background_failure = failure
            runtime.stop_reason = str(failure)
    runtime.stop.set()


def raise_background_failure(runtime: BackgroundRuntime) -> None:
    with runtime._background_failure_lock:
        failure = runtime._background_failure
    if failure is not None:
        raise failure from failure.cause


def external_stop_loop(runtime: BackgroundRuntime) -> None:
    if runtime.external_stop is None:
        return
    runtime.external_stop.wait()
    runtime.stop_reason = "local launcher shutdown"
    runtime.stop.set()


def policy(runtime: BackgroundRuntime) -> tuple[ReplicablePolicy, float, int]:
    if runtime._policy_ref is None:
        raise RuntimeError("actor policy is not initialized")
    return runtime._policy_ref.get()
