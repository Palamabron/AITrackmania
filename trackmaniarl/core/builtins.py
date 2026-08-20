"""Small, dependency-light built-ins used by the SDK template and tests."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast
from uuid import uuid4

from trackmaniarl.core.data import SampleBatch, Transition


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

    def act(self, observation: Any, *, deterministic: bool = False) -> float:
        del observation, deterministic
        return 0.0


class SmokeLearner:
    """Minimal learner proving an extension implements the TrackmaniaRL contract."""

    def __init__(self) -> None:
        self._updates = 0
        self._policy = ZeroPolicy()

    def setup(self, context: Mapping[str, Any]) -> None:
        del context

    def update(self, batch: SampleBatch) -> Mapping[str, float]:
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


JsonlTracker = JsonlRunLogger


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


class TorchCheckpointCodec:
    """Atomic zstd-streamed Torch checkpoints with legacy uncompressed reads."""

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        import zstandard

        temporary = path.with_suffix(path.suffix + ".tmp")
        with (
            temporary.open("wb") as destination,
            zstandard.ZstdCompressor(level=3).stream_writer(
                destination, closefd=False
            ) as compressed,
        ):
            # The weights-only unpickler cannot parse pickle protocol >= 4;
            # torch's default protocol keeps checkpoints loadable safely.
            torch.save(dict(state), compressed)
        temporary.replace(path)

    def load(self, path: Path) -> Mapping[str, Any]:
        import zstandard

        with path.open("rb") as source:
            compressed = source.read(4) == b"\x28\xb5\x2f\xfd"
        if not compressed:
            return _load_torch_checkpoint(path)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.decompressed.tmp")
        try:
            with (
                path.open("rb") as source,
                temporary.open("wb") as destination,
                zstandard.ZstdDecompressor().stream_reader(source) as reader,
            ):
                shutil.copyfileobj(reader, destination, length=8 * 1024**2)
            return _load_torch_checkpoint(temporary)
        finally:
            temporary.unlink(missing_ok=True)


# JSON is retained as an explicit opt-in codec for non-tensor toy learners.
class JsonCheckpointCodec:
    """Portable JSON checkpoint codec for scalar-only learner state."""

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(state), sort_keys=True), encoding="utf-8")

    def load(self, path: Path) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], json.loads(path.read_text(encoding="utf-8")))
