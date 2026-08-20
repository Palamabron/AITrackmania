"""Versioned gRPC methods and transition wire helpers."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import inspect
import ipaddress
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.data import Transition

PROTOCOL_VERSION = "1"
SERVICE = "trackmaniarl.Distributed"


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
    component_code: dict[str, str] = {}
    for component in _component_specs(components):
        class_path = component["class_path"]
        module_name = class_path.partition(":")[0]
        module = importlib.import_module(module_name)
        source = inspect.getsourcefile(module)
        if source is not None:
            source_bytes = Path(source).read_bytes().replace(b"\r\n", b"\n")
            component_code[class_path] = hashlib.sha256(source_bytes).hexdigest()
        if class_path.endswith(":MambaTemporalCore"):
            kwargs = component.get("kwargs")
            if isinstance(kwargs, dict):
                kwargs.pop("backend", None)
    config["component_code_sha256"] = component_code
    if any(
        class_path.partition(":")[0].startswith("trackmaniarl.trackmania")
        for class_path in component_code
    ):
        config["builtin_contracts"] = _trackmania_contracts()
    config = _hash_geometry_paths(config, base_dir)
    evaluation = config.get("evaluation")
    maps = evaluation.get("maps", []) if isinstance(evaluation, dict) else []
    for map_spec in maps:
        map_spec.pop("map_path", None)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _component_specs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        result = [value] if isinstance(value.get("class_path"), str) else []
        return result + [item for child in value.values() for item in _component_specs(child)]
    if isinstance(value, list):
        return [item for child in value for item in _component_specs(child)]
    return []


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
