from __future__ import annotations

import multiprocessing
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from queue import Queue
from typing import Any, cast

import grpc
import pytest

from tests.integration.runtime.distributed_runtime_support import (
    _persist_spool_then_exit,
    _RpcFailure,
)
from trackmaniarl.distributed.actor import (
    ActorBackgroundError,
    ActorRuntime,
)
from trackmaniarl.distributed.actor_transport import is_retryable_rpc_error
from trackmaniarl.distributed.codec import (
    WireCodec,
)


def _recovery_actor(tmp_path: Path) -> tuple[ActorRuntime, bytes, bytes]:
    codec = WireCodec(1024 * 1024)
    existing = tmp_path / "00000000000000000000.rollout"
    orphan = tmp_path / "00000000000000000000.tmp"
    existing_payload = codec.encode({"sequence": 100})
    orphan_payload = codec.encode({"sequence": 0})
    existing.write_bytes(existing_payload)
    orphan.write_bytes(orphan_payload)
    (tmp_path / "00000000000000000002.tmp").write_bytes(b"incomplete")
    actor = object.__new__(ActorRuntime)
    actor.spool_dir = tmp_path
    actor.codec = codec
    return actor, existing_payload, orphan_payload


def test_actor_recovers_valid_numeric_temporary_without_overwriting_spool(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "00000000000000000000.rollout"
    orphan = tmp_path / "00000000000000000000.tmp"
    invalid = tmp_path / "00000000000000000002.tmp"
    actor, existing_payload, orphan_payload = _recovery_actor(tmp_path)

    actor._recover_spool_temporaries()

    recovered = tmp_path / "00000000000000000003.rollout"
    assert existing.read_bytes() == existing_payload
    assert recovered.read_bytes() == orphan_payload
    assert not orphan.exists()
    assert invalid.read_bytes() == b"incomplete"
    assert actor._scan_spool_bytes() == sum(
        path.stat().st_size for path in (existing, recovered, invalid)
    )
    assert actor._next_sequence() == 4


def _patch_durability_calls(monkeypatch: pytest.MonkeyPatch, path: Path, events: list[str]) -> None:
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync(descriptor: int) -> None:
        events.append("file-fsync")
        real_fsync(descriptor)

    def replace(source: Path, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr("trackmaniarl.distributed.actor.os.fsync", fsync)
    monkeypatch.setattr("trackmaniarl.distributed.actor.os.replace", replace)
    monkeypatch.setattr(
        "trackmaniarl.distributed.actor.sync_checkpoint_path",
        lambda replaced: events.append("directory-sync") if replaced == path else None,
    )


def test_actor_spool_write_fsyncs_before_replace_and_syncs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    path = tmp_path / "00000000000000000000.rollout"
    _patch_durability_calls(monkeypatch, path, events)

    ActorRuntime._persist_spool_payload(path, b"durable-payload")

    assert path.read_bytes() == b"durable-payload"
    assert events == ["file-fsync", "replace", "directory-sync"]


class _AcknowledgingClient:
    def __init__(self, path: Path, stop: threading.Event) -> None:
        self.path = path
        self.stop = stop
        self.calls = 0

    def call(self, method: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        assert method == "Submit"
        assert value["identity"] == "durable-rollout"
        assert self.path.exists()
        self.calls += 1
        self.stop.set()
        return {
            "accepted": True,
            "force_refresh": False,
            "evaluate": False,
            "stop": False,
        }


def _sender_actor(path: Path, codec: WireCodec, client: Any) -> ActorRuntime:
    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = client.stop
    actor.force_refresh = threading.Event()
    actor.stop_reason = "running"
    actor.queue = Queue()
    actor.queue.put(path)
    actor.codec = codec
    actor.client = client
    actor._spool_lock = threading.Lock()
    actor._spool_bytes_total = path.stat().st_size
    actor._background_failure_lock = threading.Lock()
    actor._background_failure = None
    return actor


def _persisted_spool(tmp_path: Path) -> tuple[Path, bytes, WireCodec]:
    codec = WireCodec(1024 * 1024)
    path = tmp_path / "00000000000000000000.rollout"
    payload = codec.encode({"sequence": 0, "identity": "durable-rollout"})
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_persist_spool_then_exit, args=(str(path), payload))
    process.start()
    process.join(timeout=15.0)
    assert not process.is_alive()
    assert process.exitcode == 24
    assert path.read_bytes() == payload
    return path, payload, codec


def test_fsynced_actor_spool_survives_exit_and_is_removed_only_after_ack(
    tmp_path: Path,
) -> None:
    path, _, codec = _persisted_spool(tmp_path)
    stop = threading.Event()
    actor = _sender_actor(path, codec, _AcknowledgingClient(path, stop))

    actor._sender_loop()

    assert actor.client.calls == 1
    assert not path.exists()
    assert actor._spool_bytes_total == 0


def test_actor_rpc_retry_classifier_accepts_transient_codes() -> None:
    for code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
        assert is_retryable_rpc_error(_RpcFailure(code))


def test_actor_rpc_retry_classifier_rejects_permanent_codes() -> None:
    permanent = (
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.FAILED_PRECONDITION,
        grpc.StatusCode.INVALID_ARGUMENT,
    )
    for code in permanent:
        assert not is_retryable_rpc_error(_RpcFailure(code))


class _ImmediateStop:
    def __init__(self) -> None:
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def set(self) -> None:
        self.stopped = True

    def wait(self, timeout: float) -> bool:
        del timeout
        return self.stopped


class _FailingSubmitClient:
    def __init__(self) -> None:
        self.calls = 0
        self.stop = _ImmediateStop()

    def call(self, method: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        assert method == "Submit"
        assert value["sequence"] == 0
        self.calls += 1
        code = grpc.StatusCode.UNAVAILABLE
        if self.calls > 1:
            code = grpc.StatusCode.UNAUTHENTICATED
        raise _RpcFailure(code)


def test_actor_sender_retries_transient_rpc_but_preserves_spool_on_permanent_rpc(
    tmp_path: Path,
) -> None:
    codec = WireCodec(1024 * 1024)
    path = tmp_path / "00000000000000000000.rollout"
    path.write_bytes(codec.encode({"sequence": 0}))
    client = _FailingSubmitClient()
    actor = _sender_actor(path, codec, cast(Any, client))

    actor._sender_loop()

    assert client.calls == 2
    assert path.exists()
    assert actor._spool_bytes_total == path.stat().st_size
    with pytest.raises(ActorBackgroundError, match="rollout sender failed"):
        actor._raise_background_failure()
