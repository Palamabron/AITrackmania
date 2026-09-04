"""Small, dependency-light built-ins used by the SDK template and tests."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast
from uuid import uuid4

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import TrainingBatch, Transition

_CHECKPOINT_COPY_CHUNK_BYTES = 8 * 1024**2
_DEFAULT_MAX_DECOMPRESSED_CHECKPOINT_BYTES = 8 * 1024**3


class IdentityFeaturePipeline:
    """Collates transitions into explicit lists without copying arbitrary PyTrees."""

    def transform_observation(self, observation: Any) -> Any:
        return observation

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return {
            "observations": [item.observation for item in transitions],
            "actions": [item.action for item in transitions],
            "rewards": [item.reward for item in transitions],
            "next_observations": [item.next_observation for item in transitions],
            "terminated": [item.terminated for item in transitions],
            "truncated": [item.truncated for item in transitions],
        }


class ZeroPolicy:
    """Safe policy used only by the synthetic validation path."""

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> float:
        del observation, mode
        return 0.0


class SmokeLearner:
    """Minimal learner proving an extension implements the TrackmaniaRL contract."""

    def __init__(self) -> None:
        self._updates = 0
        self._policy = ZeroPolicy()

    def setup(self, context: Mapping[str, Any]) -> None:
        del context

    def update(self, batch: TrainingBatch) -> Mapping[str, float]:
        rewards = batch.data["rewards"]
        self._updates += 1
        return {
            "train/mean_reward": float(sum(rewards) / len(rewards)),
            "train/updates": self._updates,
        }

    def policy(self) -> ZeroPolicy:
        return self._policy

    def state_dict(self) -> Mapping[str, Any]:
        return {"updates": self._updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._updates = int(state["updates"])


class JsonlRunLogger:
    """Always-on local run logger; remote adapters remain optional components."""

    def __init__(self, run_dir: str | Path = "artifacts", run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.run_dir / "events.jsonl"
        self._run_id = run_id
        self._segment_id = uuid4().hex
        self._started_at = datetime.now(UTC)
        self._write_lock = threading.Lock()
        self._file: TextIO | None = None

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        item = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_s": (datetime.now(UTC) - self._started_at).total_seconds(),
            "run_id": self._run_id,
            "segment_id": self._segment_id,
            "event": event,
            "payload": dict(payload),
            "step": step,
        }
        line = json.dumps(item, default=str, sort_keys=True) + "\n"
        with self._write_lock:
            if self._file is None:
                self._file = self._path.open("a", encoding="utf-8")
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        with self._write_lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class CompositeRunLogger:
    """Fan out neutral events while retaining local JSONL as the source of truth."""

    def __init__(self, *loggers: Any) -> None:
        self._loggers = tuple(loggers)

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        for logger in self._loggers:
            logger.log(event, payload, step=step)

    def close(self) -> None:
        errors: list[BaseException] = []
        for logger in reversed(self._loggers):
            try:
                logger.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def _checkpoint_safe_globals() -> list[Any]:
    """Non-executable values required by checkpoint payloads."""

    import numpy
    import numpy.dtypes

    reconstruct = cast(Any, numpy)._core.multiarray._reconstruct
    dtype_classes = [value for value in vars(numpy.dtypes).values() if isinstance(value, type)]
    return [bytes, reconstruct, numpy.ndarray, numpy.dtype, *dtype_classes]


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a checkpoint without executing pickle payloads."""

    import torch

    with torch.serialization.safe_globals(_checkpoint_safe_globals()):
        return cast(
            Mapping[str, Any],
            torch.load(path, map_location="cpu", weights_only=True),
        )


def _write_zstd_checkpoint(state: Mapping[str, Any], temporary: Path) -> None:
    import torch
    import zstandard

    with temporary.open("wb") as destination:
        with zstandard.ZstdCompressor(level=3).stream_writer(
            destination, closefd=False
        ) as compressed:
            # The weights-only unpickler cannot parse pickle protocol >= 4;
            # torch's default protocol keeps checkpoints loadable safely.
            torch.save(dict(state), compressed)
        destination.flush()
        os.fsync(destination.fileno())


class TorchCheckpointCodec:
    """Atomically replace zstd-streamed Torch checkpoints on successful saves."""

    def __init__(
        self,
        max_decompressed_bytes: int = _DEFAULT_MAX_DECOMPRESSED_CHECKPOINT_BYTES,
    ) -> None:
        if max_decompressed_bytes < 1:
            raise ValueError("max_decompressed_bytes must be positive")
        self.max_decompressed_bytes = max_decompressed_bytes

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            _write_zstd_checkpoint(state, temporary)
            os.replace(temporary, path)
        except BaseException:
            # Clean up failures reported inside Python. Abrupt process exit or
            # power loss can still leave this temporary behind.
            temporary.unlink(missing_ok=True)
            raise
        sync_checkpoint_path(path)

    def load(self, path: Path) -> Mapping[str, Any]:
        import zstandard

        with path.open("rb") as source:
            compressed = source.read(4) == b"\x28\xb5\x2f\xfd"
        if not compressed:
            raise ValueError(f"checkpoint is not a zstd stream: {path}")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.decompressed.tmp")
        try:
            with (
                path.open("rb") as source,
                temporary.open("wb") as destination,
                zstandard.ZstdDecompressor().stream_reader(source) as reader,
            ):
                _copy_checkpoint_limited(reader, destination, self.max_decompressed_bytes)
            return _load_torch_checkpoint(temporary)
        finally:
            temporary.unlink(missing_ok=True)


def _copy_checkpoint_limited(source: BinaryIO, destination: BinaryIO, limit: int) -> None:
    copied = 0
    while chunk := source.read(min(_CHECKPOINT_COPY_CHUNK_BYTES, limit - copied + 1)):
        copied += len(chunk)
        if copied > limit:
            raise ValueError(f"decompressed checkpoint exceeds configured limit of {limit} bytes")
        destination.write(chunk)


def sync_checkpoint_path(path: Path) -> None:
    """Flush a replaced checkpoint before any durability-dependent callback."""

    with path.open("rb+") as checkpoint:
        os.fsync(checkpoint.fileno())
    try:
        directory = os.open(path.parent, os.O_RDONLY)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


# JSON is retained as an explicit opt-in codec for non-tensor toy learners.
class JsonCheckpointCodec:
    """Portable JSON checkpoint codec for scalar-only learner state."""

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(dict(state), destination, sort_keys=True)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        sync_checkpoint_path(path)

    def load(self, path: Path) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], json.loads(path.read_text(encoding="utf-8")))
