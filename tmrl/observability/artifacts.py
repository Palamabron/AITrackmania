"""Reproducibility manifests and bounded asynchronous episode artifact writing."""

from __future__ import annotations

import gzip
import json
import platform
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tmrl.core.data import EpisodeArtifact

if TYPE_CHECKING:
    from tmrl.core.runtime import ResolvedRun


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        secret_tokens = ("key", "token", "secret", "password")
        return {
            key: "<redacted>"
            if any(token in key.lower() for token in secret_tokens)
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def write_run_manifest(run: ResolvedRun) -> Path:
    """Write a stable, redacted manifest before the first worker is started."""

    run.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        tmrl_version = version("tmrl")
    except PackageNotFoundError:
        tmrl_version = "uninstalled"
    evaluation_assets: list[dict[str, str]] = []
    if run.spec.evaluation is not None:
        from tmrl.trackmania.geometry import BoundaryGeometry
        from tmrl.trackmania.session import PLUGIN_PROTOCOL_VERSION

        for map_spec in run.spec.evaluation.maps:
            geometry = BoundaryGeometry(
                map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
            )
            evaluation_assets.append(
                {
                    "map_id": map_spec.id,
                    "map_uid": map_spec.expected_map_uid,
                    "geometry_sha256": geometry.sha256,
                    "plugin_protocol_version": PLUGIN_PROTOCOL_VERSION,
                }
            )
    try:
        import torch

        torch_environment: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "rocm_build": torch.version.hip,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except ImportError:
        torch_environment = {"version": None}
    execution = getattr(run.learner, "execution_manifest", None)
    manifest = {
        "api_version": run.spec.api_version,
        "run_id": run.spec.run_id,
        "config": _redact(run.spec.model_dump(mode="json")),
        "components": {
            "learner": type(run.learner).__module__ + ":" + type(run.learner).__name__,
            "replay_store": type(run.replay_store).__module__
            + ":"
            + type(run.replay_store).__name__,
            "sampler": type(run.sampler).__module__ + ":" + type(run.sampler).__name__,
            "feature_pipeline": type(run.feature_pipeline).__module__
            + ":"
            + type(run.feature_pipeline).__name__,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tmrl": tmrl_version,
            "git_revision": _git_revision(),
            "torch": torch_environment,
        },
        "torch_execution": dict(execution()) if callable(execution) else None,
        "evaluation_assets": evaluation_assets,
    }
    target = run.run_dir / "manifest.json"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        candidate = json.dumps(manifest, indent=2, sort_keys=True, default=str)
        if existing != candidate:
            message = (
                "Immutable manifest already exists for run_id "
                f"{run.spec.run_id!r}; choose a new run_id"
            )
            raise FileExistsError(message)
        return target
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(target)
    return target


class AsyncEpisodeWriter:
    """Bounded background writer for compressed, reference-only episode artifacts."""

    def __init__(
        self, directory: str | Path, *, max_pending: int = 8, max_artifacts: int = 100
    ) -> None:
        if max_pending < 1 or max_artifacts < 1:
            raise ValueError("max_pending and max_artifacts must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tmrl-artifact")
        self._pending: set[Future[Path]] = set()
        self._max_pending = max_pending
        self._max_artifacts = max_artifacts

    def submit(self, artifact: EpisodeArtifact) -> Future[Path]:
        """Queue an artifact or fail fast rather than allowing unbounded memory growth."""

        self._collect_completed()
        if len(self._pending) >= self._max_pending:
            raise RuntimeError(
                "Episode artifact queue is full; raise retention limits or consume artifacts"
            )
        future = self._executor.submit(self._write, artifact)
        self._pending.add(future)
        return future

    def _write(self, artifact: EpisodeArtifact) -> Path:
        target = self.directory / f"episode-{artifact.episode_id}.json.gz"
        with gzip.open(target, "wt", encoding="utf-8") as file:
            json.dump(asdict(artifact), file, default=str, separators=(",", ":"))
        artifacts = sorted(
            self.directory.glob("episode-*.json.gz"), key=lambda item: item.stat().st_mtime
        )
        for stale in artifacts[: -self._max_artifacts]:
            stale.unlink(missing_ok=True)
        return target

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._collect_completed()

    def _collect_completed(self) -> None:
        """Raise completed writer failures instead of silently losing artifacts."""

        completed = tuple(future for future in self._pending if future.done())
        for future in completed:
            self._pending.remove(future)
        failures: list[BaseException] = []
        for future in completed:
            try:
                future.result()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
