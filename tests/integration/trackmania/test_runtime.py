"""Release contracts for observability, resume, and the optional game adapter."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

import trackmaniarl.core.builtins as core_builtins
from trackmaniarl.core.builtins import JsonlRunLogger, TorchCheckpointCodec
from trackmaniarl.observability.trackers import WandbTracker

_RUN_SPEC = """api_version: "2.0"
run_id: token-test
components:
  learner: {class_path: trackmaniarl.core.builtins:SmokeLearner}
  replay_store: {class_path: trackmaniarl.core.replay:InMemoryReplayStore}
  sampler: {class_path: trackmaniarl.core.replay:UniformSampler}
  feature_pipeline: {class_path: trackmaniarl.core.builtins:IdentityFeaturePipeline}
"""


@dataclass(slots=True)
class _WandbCapture:
    logged: list[dict[str, object]] = field(default_factory=list)
    definitions: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)


class _FakeRun:
    url = ""

    def __init__(self, capture: _WandbCapture) -> None:
        self.capture = capture

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.capture.definitions.append((name, kwargs))

    def log(self, values: dict[str, object]) -> None:
        self.capture.logged.append(values)

    def finish(self, *, exit_code: int) -> None:
        self.capture.finished.append(exit_code)


class _FakeWandb:
    capture = _WandbCapture()

    class Settings:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    @classmethod
    def init(cls, **kwargs: object) -> _FakeRun:
        del kwargs
        return _FakeRun(cls.capture)


def test_jsonl_events_have_release_envelope(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path, run_id="release")
    logger.log("train/update", {"loss": 1.0}, step=3)
    logger.close()
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["schema_version"] == "1.0"
    assert event["run_id"] == "release"
    assert event["timestamp_utc"]
    assert event["elapsed_s"] >= 0
    assert event["segment_id"]


def test_distributed_token_requires_at_least_32_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl import cli

    config = tmp_path / "run.yaml"
    config.write_text(_RUN_SPEC, encoding="utf-8")
    monkeypatch.setenv("TRACKMANIARL_DISTRIBUTED_TOKEN", "short")
    with pytest.raises(ValueError, match="at least 32 characters"):
        cli._required_token(config)


def test_torch_checkpoints_are_zstd_streamed_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    codec = TorchCheckpointCodec()
    state = {"tensor": torch.zeros(1024, dtype=torch.float32), "counter": 3}
    codec.save(state, path)
    restored = codec.load(path)
    assert path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    assert path.stat().st_size < state["tensor"].numel() * state["tensor"].element_size()
    assert torch.equal(restored["tensor"], state["tensor"])
    assert restored["counter"] == 3


def test_torch_checkpoint_removes_temporary_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    temporary = path.with_suffix(".pt.tmp")

    def fail_write(state: object, destination: Path) -> None:
        del state
        destination.write_bytes(b"incomplete")
        raise OSError("injected write failure")

    monkeypatch.setattr(core_builtins, "_write_zstd_checkpoint", fail_write)

    with pytest.raises(OSError, match="injected write failure"):
        TorchCheckpointCodec().save({}, path)

    assert not path.exists()
    assert not temporary.exists()


def test_torch_checkpoint_removes_temporary_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous checkpoint")
    temporary = path.with_suffix(".pt.tmp")

    def fail_replace(source: Path, destination: Path) -> None:
        assert (source, destination) == (temporary, path)
        raise PermissionError("injected replace failure")

    monkeypatch.setattr(core_builtins.os, "replace", fail_replace)

    with pytest.raises(PermissionError, match="injected replace failure"):
        TorchCheckpointCodec().save({"counter": 1}, path)

    assert path.read_bytes() == b"previous checkpoint"
    assert not temporary.exists()


def _assert_wandb_capture(capture: _WandbCapture) -> None:
    assert [event["episode/index"] for event in capture.logged] == [1, 2]
    assert [event["env/episode"] for event in capture.logged] == [1, 2]
    assert all("trainer/update" not in event for event in capture.logged)
    assert ("episode/*", {"step_metric": "env/episode"}) in capture.definitions
    assert capture.finished == [0]


def test_wandb_tracker_queues_remote_logging_without_reusing_global_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = _WandbCapture()
    monkeypatch.setattr(_FakeWandb, "capture", capture)
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    tracker = WandbTracker("project", run_dir=str(tmp_path))
    tracker.log("train/episode", {"index": 1}, step=10)
    tracker.log("train/episode", {"index": 2}, step=10)
    tracker.close()
    _assert_wandb_capture(capture)
