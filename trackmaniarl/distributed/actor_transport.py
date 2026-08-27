"""Actor RPC client and atomic policy reference."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.contracts import ReplicablePolicy
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.protocol import (
    auth_metadata,
    deserialize_message,
    grpc_method,
    serialize_message,
)

_RETRYABLE_RPC_CODES = frozenset(
    {
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    }
)


class Client:
    def __init__(self, target: str, token: str, codec: WireCodec) -> None:
        options = (
            ("grpc.max_receive_message_length", codec.max_message_bytes),
            ("grpc.max_send_message_length", codec.max_message_bytes),
        )
        self.channel = grpc.insecure_channel(target, options=options)
        self.token = token
        self.codec = codec

    def call(
        self, method: str, value: Mapping[str, Any], *, timeout: float = 10.0
    ) -> Mapping[str, Any]:
        stub = self.channel.unary_unary(
            grpc_method(method),
            request_serializer=serialize_message,
            response_deserializer=deserialize_message,
        )
        response = stub(
            BytesValue(value=self.codec.encode(value)),
            metadata=auth_metadata(self.token),
            timeout=timeout,
        )
        decoded = self.codec.decode(response.value)
        if not isinstance(decoded, Mapping):
            raise ValueError("distributed response must be a mapping")
        return decoded

    def close(self) -> None:
        self.channel.close()


class PolicyReference:
    def __init__(self, policy: ReplicablePolicy, epsilon: float, version: int) -> None:
        self._lock = threading.Lock()
        self._policy = policy
        self._epsilon = epsilon
        self._version = version

    def get(self) -> tuple[ReplicablePolicy, float, int]:
        with self._lock:
            return self._policy, self._epsilon, self._version

    def replace(self, policy: ReplicablePolicy, epsilon: float, version: int) -> None:
        with self._lock:
            self._policy = policy
            self._epsilon = epsilon
            self._version = version


def is_retryable_rpc_error(exc: grpc.RpcError) -> bool:
    return exc.code() in _RETRYABLE_RPC_CODES
