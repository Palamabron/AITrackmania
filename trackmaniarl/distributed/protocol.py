"""Versioned gRPC methods and transition wire helpers."""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Mapping
from typing import Any, cast

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.data import Transition
from trackmaniarl.core.fingerprint import run_fingerprint as run_fingerprint

PROTOCOL_VERSION = "1"
SERVICE = "trackmaniarl.Distributed"
MIN_DISTRIBUTED_TOKEN_LENGTH = 32


def grpc_method(name: str) -> str:
    return f"/{SERVICE}/{name}"


def serialize_message(message: BytesValue) -> bytes:
    return cast(bytes, message.SerializeToString())


def deserialize_message(payload: bytes) -> BytesValue:
    message = BytesValue()
    message.ParseFromString(payload)
    return message


def transition_to_wire(transition: Transition) -> dict[str, Any]:
    return {
        "observation": transition.observation,
        "action": transition.action,
        "reward": transition.reward,
        "next_observation": transition.next_observation,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "info": dict(transition.info),
        "episode_id": transition.episode_id,
        "step": transition.step,
    }


def transition_from_wire(value: Mapping[str, Any]) -> Transition:
    return Transition(
        observation=value["observation"],
        action=value["action"],
        reward=value["reward"],
        next_observation=value["next_observation"],
        terminated=value["terminated"],
        truncated=value["truncated"],
        info=value["info"],
        episode_id=value["episode_id"],
        step=value["step"],
    )


def authenticate(context: grpc.ServicerContext[Any, Any], token: str) -> None:
    metadata = dict(context.invocation_metadata())
    supplied = metadata.get("authorization", "")
    expected = f"Bearer {token}"
    if not hmac.compare_digest(supplied, expected):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid distributed token")


def auth_metadata(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


def require_distributed_token(token: str, *, name: str = "distributed token") -> str:
    if len(token) < MIN_DISTRIBUTED_TOKEN_LENGTH:
        raise ValueError(f"{name} must contain at least 32 characters")
    return token


def require_loopback_bind(bind: str) -> str:
    host, separator, port = bind.rpartition(":")
    if not separator or not port.isdecimal():
        raise ValueError("distributed bind must be a literal loopback address and port")
    try:
        address = ipaddress.ip_address(host.removeprefix("[").removesuffix("]"))
    except ValueError as exc:
        raise ValueError("distributed bind must use a literal loopback address") from exc
    if not address.is_loopback:
        raise ValueError("distributed learner only accepts loopback binds; use an encrypted tunnel")
    return bind
