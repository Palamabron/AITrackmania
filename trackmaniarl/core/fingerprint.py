"""Semantic identity for resumable training runs."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from trackmaniarl.core.spec import RunSpec

_FINGERPRINTED_ASSET_PATHS = frozenset({"geometry_path", "pace_reference_path"})


def run_fingerprint(spec: RunSpec, base_dir: Path) -> str:
    config, manifest = _fingerprint_config(spec)
    config["components"] = _semantic_component_tree(config["components"], manifest)
    source_digest = _trackmaniarl_source_digest()
    manifest = _bind_component_package_digests(manifest, {"trackmaniarl": source_digest})
    config["component_manifest"] = sorted(manifest, key=_canonical_sort_key)
    config["trackmaniarl_source_sha256"] = source_digest
    if _uses_trackmania(manifest):
        config["builtin_contracts"] = _trackmania_contracts()
    config = _hash_asset_paths(config, base_dir)
    _remove_evaluation_map_paths(config)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _fingerprint_config(spec: RunSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = spec.model_dump(mode="json")
    config.pop("run_id")
    config.pop("artifacts_dir")
    components = config["components"]
    components.pop("logger")
    components.pop("additional_loggers")
    config["components"] = components
    return config, []


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _trackmaniarl_source_digest() -> str:
    root = Path(__file__).parent.parent
    sources = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for source in sources:
        relative = source.relative_to(root).as_posix().encode()
        content = source.read_bytes().replace(b"\r\n", b"\n")
        digest.update(relative + b"\0" + content + b"\0")
    return digest.hexdigest()


def _uses_trackmania(manifest: list[dict[str, Any]]) -> bool:
    return any(
        str(item["resolved_symbol"]).partition(":")[0].startswith("trackmaniarl.trackmania")
        for item in manifest
    )


def _remove_evaluation_map_paths(config: dict[str, Any]) -> None:
    evaluation = config["evaluation"]
    if evaluation is None:
        return
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation fingerprint state must be a mapping")
    maps = evaluation["maps"]
    for map_spec in maps:
        map_spec.pop("map_path")


def _semantic_component_tree(value: Any, manifest: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        return _semantic_mapping(value, manifest)
    if isinstance(value, (list, tuple)):
        return [_semantic_component_tree(item, manifest) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _semantic_component_tree(value.value, manifest)
    if isinstance(value, (set, frozenset)):
        return _semantic_set(value, manifest)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _semantic_object(value, manifest)


def _semantic_mapping(value: dict[Any, Any], manifest: list[dict[str, Any]]) -> Any:
    class_path = value.get("class_path")
    if not isinstance(class_path, str):
        return {str(key): _semantic_component_tree(item, manifest) for key, item in value.items()}
    symbol = _resolved_symbol(class_path)
    provided = value.get("kwargs", {})
    if not isinstance(provided, Mapping):
        raise TypeError(f"Component {class_path!r} kwargs must be a mapping")
    parameters = _semantic_component_tree(_semantic_parameters(symbol, provided), manifest)
    implementation = _component_implementation(symbol)
    manifest.append(_component_manifest_entry(class_path, implementation, parameters))
    return {"class_path": class_path, "kwargs": parameters}


def _component_manifest_entry(
    class_path: str, implementation: Mapping[str, Any], parameters: Any
) -> dict[str, Any]:
    return {
        "class_path": class_path,
        "resolved_symbol": implementation["resolved_symbol"],
        "source_sha256": implementation["source_sha256"],
        "parameters": parameters,
    }


def _semantic_set(value: set[Any] | frozenset[Any], manifest: list[dict[str, Any]]) -> list[Any]:
    normalized = [_semantic_component_tree(item, manifest) for item in value]
    return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))


def _semantic_object(value: Any, manifest: list[dict[str, Any]]) -> Any:
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


def _bind_component_package_digests(
    manifest: list[dict[str, Any]], digests: dict[str, str | None]
) -> list[dict[str, Any]]:
    bound = []
    for item in manifest:
        entry = dict(item)
        packages = _component_packages(entry)
        for package in packages:
            if package not in digests:
                digests[package] = _package_source_digest(package)
        entry["package_sources_sha256"] = {package: digests[package] for package in packages}
        bound.append(entry)
    return bound


def _component_packages(entry: Mapping[str, Any]) -> tuple[str, ...]:
    component_paths = (str(entry["class_path"]), str(entry["resolved_symbol"]))
    packages = {path.partition(":")[0].partition(".")[0] for path in component_paths}
    return tuple(sorted(packages))


def _package_source_digest(package_name: str) -> str | None:
    sources = sorted(_package_python_sources(package_name))
    if not sources:
        return None
    digest = hashlib.sha256()
    for relative, content in sources:
        digest.update(relative.encode() + b"\0" + content + b"\0")
    return digest.hexdigest()


def _package_python_sources(package_name: str) -> list[tuple[str, bytes]]:
    module = importlib.import_module(package_name)
    roots = [Path(path) for path in getattr(module, "__path__", ())]
    sources = [
        (source.relative_to(root).as_posix(), _normalized_source(source))
        for root in roots
        for source in root.rglob("*.py")
        if source.is_file()
    ]
    if sources:
        return sources
    source_path = getattr(module, "__file__", None)
    if source_path is None:
        return []
    source = Path(source_path)
    return [(source.name, _normalized_source(source))] if source.is_file() else []


def _normalized_source(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


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


def _hash_asset_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                _asset_hash(item, base_dir, key)
                if key in _FINGERPRINTED_ASSET_PATHS
                else _hash_asset_paths(item, base_dir)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_hash_asset_paths(item, base_dir) for item in value]
    return value


def _asset_hash(value: Any, base_dir: Path, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    asset = (base_dir / value).resolve()
    return hashlib.sha256(asset.read_bytes()).hexdigest()
