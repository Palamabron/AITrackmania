"""Safe compressed wire encoding for PyTrees and policy tensors."""

from __future__ import annotations

import json
import struct
from base64 import b64decode, b64encode
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
import torch
import zstandard
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

_HEADER = struct.Struct(">I")
_DECODE_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    RecursionError,
    TypeError,
    UnicodeError,
    ValueError,
    SafetensorError,
)


class WirePayloadTooLargeError(ValueError):
    """The wire payload exceeds the configured transport limit."""


class WirePayloadFormatError(ValueError):
    """The wire payload cannot be decoded as the supported safe format."""


class WireCodec:
    """Encode supported PyTrees without executable pickle payloads."""

    def __init__(self, max_message_bytes: int) -> None:
        if max_message_bytes < 1:
            raise ValueError("max_message_bytes must be positive")
        self.max_message_bytes = max_message_bytes

    def encode(self, value: Any) -> bytes:
        tensors: dict[str, torch.Tensor] = {}
        manifest = self._encode_node(value, tensors)
        metadata = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
        tensor_data = save_tensors(tensors) if tensors else b""
        raw = _HEADER.pack(len(metadata)) + metadata + tensor_data
        if len(raw) > self.max_message_bytes:
            raise WirePayloadTooLargeError(
                f"encoded message is {len(raw)} bytes before compression; "
                f"limit is {self.max_message_bytes}"
            )
        compressed = zstandard.ZstdCompressor(level=3).compress(raw)
        if len(compressed) > self.max_message_bytes:
            raise WirePayloadTooLargeError(
                f"encoded message is {len(compressed)} bytes; limit is {self.max_message_bytes}"
            )
        return compressed

    def decode(self, payload: bytes) -> Any:
        if len(payload) > self.max_message_bytes:
            raise WirePayloadTooLargeError(
                f"received message is {len(payload)} bytes; limit is {self.max_message_bytes}"
            )
        raw = self._decompress(payload)
        if len(raw) > self.max_message_bytes:
            raise WirePayloadTooLargeError("wire payload exceeds the decompressed size limit")
        try:
            return self._decode_raw(raw)
        except _DECODE_ERRORS as exc:
            raise WirePayloadFormatError("wire payload is invalid") from exc

    def _decompress(self, payload: bytes) -> bytes:
        try:
            with zstandard.ZstdDecompressor().stream_reader(BytesIO(payload)) as reader:
                return reader.read(self.max_message_bytes + 1)
        except zstandard.ZstdError as exc:
            raise WirePayloadFormatError("wire payload is invalid") from exc

    def _decode_raw(self, raw: bytes) -> Any:
        if len(raw) < _HEADER.size:
            raise ValueError("wire payload is truncated")
        metadata_size = _HEADER.unpack(raw[: _HEADER.size])[0]
        metadata_end = _HEADER.size + metadata_size
        if metadata_end > len(raw):
            raise ValueError("wire metadata is truncated")
        manifest = json.loads(raw[_HEADER.size : metadata_end])
        tensors = load_tensors(raw[metadata_end:]) if metadata_end < len(raw) else {}
        return self._decode_node(manifest, tensors)

    def _encode_node(self, value: Any, tensors: dict[str, torch.Tensor]) -> Any:
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return self._encode_array(value, tensors)
        if isinstance(value, Mapping):
            return self._encode_mapping(value, tensors)
        if isinstance(value, (tuple, list)):
            return self._encode_sequence(value, tensors)
        return self._encode_scalar(value)

    @staticmethod
    def _encode_array(
        value: torch.Tensor | np.ndarray, tensors: dict[str, torch.Tensor]
    ) -> dict[str, str]:
        name = f"tensor_{len(tensors)}"
        if isinstance(value, torch.Tensor):
            tensors[name] = value.detach().cpu().contiguous()
            return {"kind": "tensor", "name": name}
        tensors[name] = torch.from_numpy(np.ascontiguousarray(value))
        return {"kind": "ndarray", "name": name}

    def _encode_mapping(
        self, value: Mapping[Any, Any], tensors: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        if any(not isinstance(key, str) for key in value):
            raise TypeError("wire mappings require string keys")
        items = {key: self._encode_node(item, tensors) for key, item in value.items()}
        return {"kind": "mapping", "items": items}

    def _encode_sequence(
        self, value: tuple[Any, ...] | list[Any], tensors: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        kind = "tuple" if isinstance(value, tuple) else "list"
        return {"kind": kind, "items": [self._encode_node(item, tensors) for item in value]}

    @staticmethod
    def _encode_scalar(value: Any) -> dict[str, Any]:
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"kind": "scalar", "value": value}
        if isinstance(value, bytes):
            return {"kind": "bytes", "value": b64encode(value).decode("ascii")}
        if isinstance(value, np.generic):
            return {"kind": "scalar", "value": value.item()}
        raise TypeError(f"unsupported wire value: {type(value).__name__}")

    def _decode_node(self, node: Any, tensors: Mapping[str, torch.Tensor]) -> Any:
        if not isinstance(node, dict) or "kind" not in node:
            raise ValueError("invalid wire node")
        kind = node["kind"]
        if kind == "tensor":
            return tensors[node["name"]]
        if kind == "ndarray":
            return tensors[node["name"]].numpy()
        if kind == "mapping":
            return {key: self._decode_node(value, tensors) for key, value in node["items"].items()}
        if kind == "tuple":
            return tuple(self._decode_node(value, tensors) for value in node["items"])
        if kind == "list":
            return [self._decode_node(value, tensors) for value in node["items"]]
        if kind == "scalar":
            return node["value"]
        if kind == "bytes":
            return b64decode(node["value"], validate=True)
        raise ValueError(f"unknown wire node kind: {kind!r}")
