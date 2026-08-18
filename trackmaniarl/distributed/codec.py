"""Safe compressed wire encoding for PyTrees and policy tensors."""

from __future__ import annotations

import json
import struct
from base64 import b64decode, b64encode
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import zstandard
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

_HEADER = struct.Struct(">I")


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
        compressed = zstandard.ZstdCompressor(level=3).compress(
            _HEADER.pack(len(metadata)) + metadata + tensor_data
        )
        if len(compressed) > self.max_message_bytes:
            raise ValueError(
                f"encoded message is {len(compressed)} bytes; limit is {self.max_message_bytes}"
            )
        return compressed

    def decode(self, payload: bytes) -> Any:
        if len(payload) > self.max_message_bytes:
            raise ValueError(
                f"received message is {len(payload)} bytes; limit is {self.max_message_bytes}"
            )
        raw = zstandard.ZstdDecompressor().decompress(
            payload, max_output_size=self.max_message_bytes * 32
        )
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
        if isinstance(value, torch.Tensor):
            name = f"tensor_{len(tensors)}"
            tensors[name] = value.detach().cpu().contiguous()
            return {"kind": "tensor", "name": name}
        if isinstance(value, np.ndarray):
            name = f"tensor_{len(tensors)}"
            tensors[name] = torch.from_numpy(np.ascontiguousarray(value))
            return {"kind": "ndarray", "name": name}
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("wire mappings require string keys")
            return {
                "kind": "mapping",
                "items": {key: self._encode_node(item, tensors) for key, item in value.items()},
            }
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self._encode_node(item, tensors) for item in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [self._encode_node(item, tensors) for item in value]}
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
