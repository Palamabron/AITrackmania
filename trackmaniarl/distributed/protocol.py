"""Versioned gRPC methods and transition wire helpers."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import inspect
import ipaddress
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, cast

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.data import Transition

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
        reward=float(value["reward"]),
        next_observation=value["next_observation"],
        terminated=bool(value["terminated"]),
        truncated=bool(value["truncated"]),
        info=value["info"],
        episode_id=value["episode_id"],
        step=int(value["step"]),
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


def run_fingerprint(spec: Any, base_dir: Path) -> str:
    config = spec.model_dump(mode="json")
    config.pop("run_id", None)
    config.pop("artifacts_dir", None)
    components = config.get("components", {})
    components.pop("logger", None)
    components.pop("additional_loggers", None)
    component_manifest: list[dict[str, Any]] = []
    config["components"] = _semantic_component_tree(components, component_manifest)
    config["component_manifest"] = sorted(
        component_manifest,
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
    )
    if any(
        str(item["resolved_symbol"]).partition(":")[0].startswith("trackmaniarl.trackmania")
        for item in component_manifest
    ):
        config["builtin_contracts"] = _trackmania_contracts()
    config = _hash_geometry_paths(config, base_dir)
    evaluation = config.get("evaluation")
    maps = evaluation.get("maps", []) if isinstance(evaluation, dict) else []
    for map_spec in maps:
        map_spec.pop("map_path", None)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _semantic_component_tree(value: Any, manifest: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        class_path = value.get("class_path")
        if isinstance(class_path, str):
            symbol = _resolved_symbol(class_path)
            provided = value.get("kwargs", {})
            if not isinstance(provided, Mapping):
                raise TypeError(f"Component {class_path!r} kwargs must be a mapping")
            parameters = _semantic_parameters(symbol, provided)
            parameters = _semantic_component_tree(parameters, manifest)
            implementation = _component_implementation(symbol)
            manifest.append(
                {
                    "class_path": class_path,
                    "resolved_symbol": implementation["resolved_symbol"],
                    "source_sha256": implementation["source_sha256"],
                    "parameters": parameters,
                }
            )
            return {"class_path": class_path, "kwargs": parameters}
        return {str(key): _semantic_component_tree(item, manifest) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_component_tree(item, manifest) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _semantic_component_tree(value.value, manifest)
    if isinstance(value, (set, frozenset)):
        normalized = [_semantic_component_tree(item, manifest) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _semantic_component_tree(dump(mode="json"), manifest)
    return {"type": f"{type(value).__module__}:{type(value).__qualname__}"}


def _resolved_symbol(class_path: str) -> Any:
    module_name, _, symbol_name = class_path.partition(":")
    return getattr(importlib.import_module(module_name), symbol_name)


def _component_implementation(symbol: Any) -> dict[str, str | None]:
    module_name = getattr(symbol, "__module__", type(symbol).__module__)
    qualname = getattr(symbol, "__qualname__", type(symbol).__qualname__)
    try:
        source = inspect.getsourcefile(symbol)
    except TypeError:
        source = None
    source_hash = None
    if source is not None:
        source_bytes = Path(source).read_bytes().replace(b"\r\n", b"\n")
        source_hash = hashlib.sha256(source_bytes).hexdigest()
    return {
        "resolved_symbol": f"{module_name}:{qualname}",
        "source_sha256": source_hash,
    }


def _semantic_parameters(symbol: Any, provided: Mapping[str, Any]) -> dict[str, Any]:
    ignored = frozenset(getattr(symbol, "fingerprint_ignored_parameters", ()))
    parameters = {str(key): value for key, value in provided.items() if key not in ignored}
    try:
        signature = inspect.signature(symbol)
    except (TypeError, ValueError):
        return parameters
    for name, parameter in signature.parameters.items():
        if name in ignored or name in parameters:
            continue
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        if parameter.default is not inspect.Parameter.empty:
            parameters[name] = parameter.default
    return parameters


def _trackmania_contracts() -> dict[str, Any]:
    from trackmaniarl.trackmania.actions import build_brake_tap_action_table
    from trackmaniarl.trackmania.features import LidarFeaturePipeline

    _, action_table = build_brake_tap_action_table()
    action_bytes = b"".join(item.tobytes() for item in action_table)
    return {
        "action_table_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "feature_schema": LidarFeaturePipeline.schema_version,
        "feature_fields": LidarFeaturePipeline.source_fields,
    }


def _hash_geometry_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                _geometry_hash(item, base_dir)
                if key == "geometry_path"
                else _hash_geometry_paths(item, base_dir)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_hash_geometry_paths(item, base_dir) for item in value]
    return value


def _geometry_hash(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("geometry_path must be a string")
    geometry = (base_dir / value).resolve()
    return hashlib.sha256(geometry.read_bytes()).hexdigest()
