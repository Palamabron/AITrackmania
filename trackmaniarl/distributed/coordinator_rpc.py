from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, NoReturn, cast

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.distributed.codec import WirePayloadFormatError, WirePayloadTooLargeError
from trackmaniarl.distributed.coordinator_submission import submit as submit
from trackmaniarl.distributed.coordinator_support import _RolloutRejection
from trackmaniarl.distributed.coordinator_validation import (
    _required_integer,
    _required_nonempty_string,
    _validate_fields,
)
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    SERVICE,
    authenticate,
    deserialize_message,
    serialize_message,
)

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator


@dataclass(frozen=True, slots=True)
class _AbortReason:
    code: grpc.StatusCode
    message: str


_OVERSIZED_REQUEST = _AbortReason(
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    "distributed request exceeds the configured size limit",
)
_MALFORMED_REQUEST = _AbortReason(
    grpc.StatusCode.INVALID_ARGUMENT,
    "distributed request payload is malformed",
)
_BASE_REQUEST_FIELDS = frozenset({"protocol_version", "fingerprint", "actor_id", "session_id"})
_REGISTER_FIELDS = _BASE_REQUEST_FIELDS
_POLICY_FIELDS = _BASE_REQUEST_FIELDS | {"current_version"}
_HEARTBEAT_FIELDS = _BASE_REQUEST_FIELDS | {"policy_version", "spool_bytes"}


def start_server(coordinator: Coordinator) -> None:
    options = _server_options(coordinator)
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="trackmaniarl-grpc")
    server = grpc.server(executor, options=options)
    handlers = _rpc_handlers(coordinator)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, handlers),))
    _bind_server(coordinator, server, executor)


def _server_options(coordinator: Coordinator) -> tuple[tuple[str, int], ...]:
    maximum = coordinator.run.spec.distributed.max_message_bytes
    return (
        ("grpc.max_receive_message_length", maximum),
        ("grpc.max_send_message_length", maximum),
    )


def _rpc_handlers(coordinator: Coordinator) -> dict[str, grpc.RpcMethodHandler]:
    callbacks = {
        "Register": coordinator._register,
        "Submit": coordinator._submit,
        "Policy": coordinator._policy,
        "Heartbeat": coordinator._heartbeat,
    }
    return {
        name: grpc.unary_unary_rpc_method_handler(
            callback,
            request_deserializer=deserialize_message,
            response_serializer=serialize_message,
        )
        for name, callback in callbacks.items()
    }


def _bind_server(
    coordinator: Coordinator, server: grpc.Server, executor: ThreadPoolExecutor
) -> None:
    bound_port = server.add_insecure_port(coordinator.bind)
    if bound_port == 0:
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f"could not bind distributed learner to {coordinator.bind}")
    server.start()
    coordinator._server = server
    coordinator._bound_port = bound_port
    coordinator._rpc_executor = executor


def request(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> Mapping[str, Any]:
    authenticate(context, coordinator.token)
    value = _decode_request(coordinator, message, context)
    _validate_request_identity(coordinator, value, context)
    return value


def _decode_request(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> Mapping[str, Any]:
    try:
        value = coordinator.codec.decode(message.value)
    except WirePayloadTooLargeError as exc:
        _abort_request(context, _OVERSIZED_REQUEST, exc)
    except WirePayloadFormatError as exc:
        _abort_request(context, _MALFORMED_REQUEST, exc)
    return _request_mapping(value, context)


def _abort_request(
    context: grpc.ServicerContext[Any, Any], reason: _AbortReason, error: ValueError
) -> NoReturn:
    context.abort(reason.code, reason.message)
    raise AssertionError("gRPC abort returned") from error


def _request_mapping(value: object, context: grpc.ServicerContext[Any, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request must be a mapping")
    return cast(Mapping[str, Any], value)


def _validate_request_identity(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    context: grpc.ServicerContext[Any, Any],
) -> None:
    if value.get("protocol_version") != PROTOCOL_VERSION:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, "protocol version mismatch")
    if value.get("fingerprint") != coordinator.fingerprint:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, "run fingerprint mismatch")


def response(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    context: grpc.ServicerContext[Any, Any],
) -> BytesValue:
    try:
        return BytesValue(value=coordinator.codec.encode(value))
    except WirePayloadTooLargeError as exc:
        context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "distributed response exceeds the configured size limit",
        )
        raise AssertionError("gRPC abort returned") from exc


def log_rollout_rejected(
    coordinator: Coordinator,
    value: Mapping[str, Any],
    rejection: _RolloutRejection,
) -> None:
    coordinator.run.logger.log(
        "distributed/rollout_rejected",
        {
            "actor_id": str(value["actor_id"]),
            "session_id": str(value["session_id"]),
            "sequence": int(value["sequence"]),
            "reason": rejection.reason,
            **rejection.details,
        },
        step=coordinator.counters.updates,
    )


def register(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> BytesValue:
    value = coordinator._request(message, context)
    actor_id = _register_actor(coordinator, value)
    profile = coordinator.journal.actor_profile(
        actor_id, len(coordinator.run.spec.distributed.epsilon_profiles)
    )
    _log_registration(coordinator, value, profile)
    with coordinator._lock:
        payload = _registration_payload(coordinator, profile)
        return coordinator._response(payload, context)


def _register_actor(coordinator: Coordinator, value: Mapping[str, Any]) -> str:
    _validate_fields(value, _REGISTER_FIELDS, "registration request")
    actor_id = _required_nonempty_string(value, "actor_id")
    _required_nonempty_string(value, "session_id")
    with coordinator._lock:
        coordinator._last_heartbeats[actor_id] = monotonic()
        coordinator._timed_out_actors.discard(actor_id)
    return actor_id


def _log_registration(coordinator: Coordinator, value: Mapping[str, Any], profile: int) -> None:
    coordinator.run.logger.log(
        "actor/registered",
        {
            "actor_id": str(value["actor_id"]),
            "session_id": value["session_id"],
            "epsilon_profile": profile,
            "run_fingerprint": coordinator.fingerprint,
        },
        step=coordinator.counters.updates,
    )


def _registration_payload(coordinator: Coordinator, profile: int) -> dict[str, Any]:
    return {
        "accepted": True,
        "policy_version": coordinator.counters.policy_version,
        "epsilon": coordinator._epsilon(profile),
        "stop": coordinator._should_stop(),
    }


def policy(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> BytesValue:
    value = coordinator._request(message, context)
    _validate_fields(value, _POLICY_FIELDS, "policy request")
    profile = coordinator.journal.actor_profile(
        _required_nonempty_string(value, "actor_id"),
        len(coordinator.run.spec.distributed.epsilon_profiles),
    )
    with coordinator._lock:
        current = _required_integer(value, "current_version", minimum=-1)
        payload = _policy_response(coordinator, profile, current)
        return coordinator._response(payload, context)


def _policy_response(coordinator: Coordinator, profile: int, current: int) -> dict[str, Any]:
    version = coordinator.counters.policy_version
    return {
        "policy_version": version,
        "snapshot": coordinator._policy_payload if current != version else b"",
        "epsilon": coordinator._epsilon(profile),
        "stop": coordinator._should_stop(),
    }


def heartbeat(
    coordinator: Coordinator,
    message: BytesValue,
    context: grpc.ServicerContext[Any, Any],
) -> BytesValue:
    value = coordinator._request(message, context)
    _validate_fields(value, _HEARTBEAT_FIELDS, "heartbeat request")
    actor_id = _required_nonempty_string(value, "actor_id")
    _required_nonempty_string(value, "session_id")
    _required_integer(value, "policy_version", minimum=-1)
    _required_integer(value, "spool_bytes", minimum=0)
    with coordinator._lock:
        coordinator._last_heartbeats[actor_id] = monotonic()
        coordinator._timed_out_actors.discard(actor_id)
    _log_heartbeat(coordinator, actor_id, value)
    return coordinator._response({"stop": coordinator._should_stop()}, context)


def _log_heartbeat(coordinator: Coordinator, actor_id: str, value: Mapping[str, Any]) -> None:
    coordinator.run.logger.log(
        "actor/heartbeat",
        {
            "actor_id": actor_id,
            "policy_version": value["policy_version"],
            "spool_bytes": value["spool_bytes"],
        },
        step=coordinator.counters.updates,
    )


def epsilon(coordinator: Coordinator, profile: int) -> float:
    spec = coordinator.run.spec.distributed
    schedule_progress = (
        coordinator.counters.transitions
        if spec.epsilon_decay_updates is None
        else coordinator.counters.updates
    )
    schedule_length = spec.epsilon_decay_updates or spec.epsilon_decay_transitions
    fraction = min(1.0, schedule_progress / schedule_length)
    scheduled = spec.epsilon_start + fraction * (spec.epsilon_final - spec.epsilon_start)
    return scheduled * spec.epsilon_profiles[profile]
