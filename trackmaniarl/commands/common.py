"""Command-line entrypoint for the current TrackmaniaRL project workflow."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from time import time_ns
from typing import Any

from trackmaniarl.core.runtime import resolve_run, validate_resolved_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.project.scaffold import create_project


def _configure_process_logging() -> None:
    """Send library INFO logs (progress, episodes, demos) to the console."""

    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("trackmaniarl").setLevel(logging.INFO)


def _package_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_").lower()
    if not name or name[0].isdigit():
        raise ValueError("Project package name must start with a letter or underscore")
    return name


def _init(args: argparse.Namespace) -> None:
    package = _package_name(args.package or Path(args.directory).name)
    target = create_project(args.directory, package, template=args.template)
    print(f"Created {target}")
    print(f'Next: cd "{target}"')
    print("      uv sync")
    print("      uv run trackmaniarl validate run.yaml")


def _validate(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    # Validation writes a synthetic checkpoint and must never reserve the
    # artifact directory intended for the real live-training run.
    components = spec.components.model_copy(update={"additional_loggers": ()})
    validation_spec = spec.model_copy(
        update={"components": components, "run_id": f"{spec.run_id}-validate-{time_ns()}"}
    )
    run = resolve_run(validation_spec, base_dir=Path(args.config).parent)
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()
    print(f"Validated {spec.run_id}: {metrics}")
    print(f"Manifest: {run.run_dir / 'manifest.json'}")


def _iter_class_paths(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child_path = f"{path}.{key}" if path else str(key)
            child = value[key]
            if key == "class_path":
                yield child_path, str(child)
            yield from _iter_class_paths(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_class_paths(child, f"{path}[{index}]")


def _inspect_config(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    components = sorted(_iter_class_paths(spec.model_dump(mode="python")))
    print(
        "run.yaml is trusted executable configuration: inspect-config is not a sandbox "
        "and does not import components."
    )
    for path, class_path in components:
        module = class_path.partition(":")[0]
        origin = (
            "first_party"
            if module == "trackmaniarl" or module.startswith("trackmaniarl.")
            else "external"
        )
        print(f"{path}\t{origin}\t{class_path}")
    print(f"Validated {spec.run_id}: {len(components)} component path(s)")


def _load_env_value(config: Path, name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    dotenv = config.resolve().parent / ".env"
    if not dotenv.exists():
        return None
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = raw_line.partition("=")
        if separator and key.strip() == name:
            return raw_value.strip().strip("'\"") or None
    return None


def _required_token(config: Path) -> str:
    from trackmaniarl.distributed.protocol import require_distributed_token

    spec = RunSpec.from_yaml(config)
    token = _load_env_value(config, spec.distributed.token_env)
    if not token:
        raise ValueError(
            f"Set {spec.distributed.token_env} in the environment or {config.parent / '.env'}"
        )
    try:
        return require_distributed_token(token, name=spec.distributed.token_env)
    except ValueError as exc:
        raise ValueError(
            f"{exc}; "
            'generate a random token with `python -c "import secrets; '
            'print(secrets.token_urlsafe(32))"`'
        ) from exc


def _spawn_executable() -> str:
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        return sys.executable
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    candidate = Path(virtual_env) / scripts_dir / executable_name
    return str(candidate) if candidate.is_file() else sys.executable


def _spawn_context() -> multiprocessing.context.SpawnContext:
    """Start Windows children with the active virtual environment's interpreter."""

    multiprocessing.set_executable(_spawn_executable())
    return multiprocessing.get_context("spawn")


def _next_versioned_run_id(run_id: str, artifacts_dir: Path) -> str:
    """Return the first free local run identifier for a new training attempt."""

    index = 1
    while (artifacts_dir / f"{run_id}-{index}").exists():
        index += 1
    return f"{run_id}-{index}"


def _new_attempt_spec(config: Path, spec: RunSpec, args: argparse.Namespace) -> RunSpec:
    """Assign a distinct run ID when a fresh local attempt would reuse artifacts."""

    fresh_attempt = bool(getattr(args, "reset_replay", False)) or not bool(
        getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    )
    artifacts_dir = config.parent / spec.artifacts_dir
    if not fresh_attempt or not (artifacts_dir / spec.run_id).exists():
        return spec
    run_id = _next_versioned_run_id(spec.run_id, artifacts_dir)
    print(f"Run ID {spec.run_id!r} already exists; using {run_id!r} for this new attempt.")
    return spec.model_copy(update={"run_id": run_id})


def _resumed_attempt_spec(config: Path, spec: RunSpec, args: argparse.Namespace) -> RunSpec:
    run_dir = _resumed_run_dir(config, spec, args)
    if run_dir is None or not _matches_attempt(spec.run_id, run_dir.name):
        return spec
    if run_dir.name != spec.run_id:
        print(f"Resuming checkpoint run ID {run_dir.name!r}.")
    resumed_spec = spec.model_copy(update={"run_id": run_dir.name})
    return _restore_warm_start_identity(resumed_spec, run_dir)


def _resumed_run_dir(config: Path, spec: RunSpec, args: argparse.Namespace) -> Path | None:
    checkpoint = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    if checkpoint is None or bool(getattr(args, "reset_replay", False)):
        return None
    path = Path(checkpoint).resolve()
    artifacts_dir = (config.parent / spec.artifacts_dir).resolve()
    run_dir = path.parent.parent
    if path.parent.name != "checkpoints" or run_dir.parent != artifacts_dir:
        return None
    return run_dir


def _matches_attempt(configured_id: str, resumed_id: str) -> bool:
    if resumed_id == configured_id:
        return True
    return re.fullmatch(rf"{re.escape(configured_id)}-[1-9][0-9]*", resumed_id) is not None


def _restore_warm_start_identity(spec: RunSpec, run_dir: Path) -> RunSpec:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return spec
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initialization = _manifest_initialization(manifest)
    if (
        not isinstance(initialization, str)
        or "model_initialization_checkpoint" in spec.components.learner.kwargs
    ):
        return spec
    return _with_model_initialization_checkpoint(spec, Path(initialization))


def _manifest_initialization(manifest: Any) -> Any:
    if not isinstance(manifest, dict):
        raise TypeError("manifest.json must contain an object")
    config = manifest["config"]
    if not isinstance(config, dict):
        raise TypeError("manifest config must contain an object")
    components = config["components"]
    if not isinstance(components, dict):
        raise TypeError("manifest components must contain an object")
    learner = components["learner"]
    if not isinstance(learner, dict):
        raise TypeError("manifest learner must contain an object")
    kwargs = learner["kwargs"]
    if not isinstance(kwargs, dict):
        raise TypeError("manifest learner kwargs must contain an object")
    return kwargs.get("model_initialization_checkpoint")


def _with_model_initialization_checkpoint(spec: RunSpec, checkpoint: Path) -> RunSpec:
    learner = spec.components.learner
    kwargs = dict(learner.kwargs)
    kwargs["model_initialization_checkpoint"] = str(checkpoint)
    learner = learner.model_copy(update={"kwargs": kwargs})
    components = spec.components.model_copy(update={"learner": learner})
    return spec.model_copy(update={"components": components})
