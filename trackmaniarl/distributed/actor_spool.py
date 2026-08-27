"""Durable rollout spool operations for distributed actors."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from queue import Queue
from threading import Event
from time import monotonic
from typing import Any, Protocol

from trackmaniarl.core.builtins import sync_checkpoint_path
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor_requests import SpoolRequest
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.protocol import transition_to_wire

logger = logging.getLogger(__name__)

_SPOOL_WAIT_POLL_S = 0.5
_SPOOL_WAIT_WARN_S = 10.0


class SpoolRuntime(Protocol):
    spec: RunSpec
    actor_id: str
    stop: Event
    codec: WireCodec
    spool_dir: Path
    queue: Queue[Path]
    sequence: int
    _spool_lock: Any
    _spool_bytes_total: int

    def _request_base(self) -> dict[str, Any]: ...

    def _persist_spool_payload(self, path: Path, payload: bytes) -> None: ...

    def _wait_for_spool_capacity(self, payload_bytes: int) -> None: ...

    def _current_spool_bytes(self) -> int: ...

    def _valid_spool_temporary(self, path: Path) -> bool: ...


def spool(runtime: SpoolRuntime, request: SpoolRequest) -> None:
    if not request.transitions and not request.summaries and not request.evaluations:
        return
    value = _spool_value(runtime, request)
    payload = runtime.codec.encode(value)
    runtime._wait_for_spool_capacity(len(payload))
    if runtime.stop.is_set():
        return
    _persist_rollout(runtime, payload)


def _spool_value(runtime: SpoolRuntime, request: SpoolRequest) -> dict[str, Any]:
    value = {
        **runtime._request_base(),
        "sequence": runtime.sequence,
        "policy_version": _chunk_policy_version(request),
        "transitions": [transition_to_wire(item) for item in request.transitions],
        "episodes": request.summaries,
        "evaluations": request.evaluations,
        "evaluation_snapshot": request.evaluation_snapshot,
    }
    return value


def _persist_rollout(runtime: SpoolRuntime, payload: bytes) -> None:
    path = runtime.spool_dir / f"{runtime.sequence:020d}.rollout"
    runtime._persist_spool_payload(path, payload)
    with runtime._spool_lock:
        runtime._spool_bytes_total += len(payload)
    runtime.sequence += 1
    runtime.queue.put(path)


def _chunk_policy_version(request: SpoolRequest) -> int:
    if not request.transitions:
        return request.policy_version
    versions: list[int] = []
    for transition in request.transitions:
        version = transition.info["policy_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("transition policy_version must be an integer")
        versions.append(version)
    return min(versions)


def wait_for_spool_capacity(runtime: SpoolRuntime, payload_bytes: int) -> None:
    limit = runtime.spec.distributed.spool_max_bytes
    if payload_bytes > limit:
        raise ValueError(f"rollout payload is {payload_bytes} bytes; spool limit is {limit}")
    warned_at = float("-inf")
    while not runtime.stop.is_set():
        if runtime._current_spool_bytes() + payload_bytes <= limit:
            return
        if monotonic() - warned_at >= _SPOOL_WAIT_WARN_S:
            logger.warning(
                "Actor %s rollout spool is full (%d bytes); pausing collection until "
                "the learner drains it",
                runtime.actor_id,
                runtime._current_spool_bytes(),
            )
            warned_at = monotonic()
        runtime.stop.wait(_SPOOL_WAIT_POLL_S)


def discard_spooled(runtime: SpoolRuntime, path: Path, size: int) -> None:
    path.unlink(missing_ok=True)
    with runtime._spool_lock:
        runtime._spool_bytes_total = max(0, runtime._spool_bytes_total - size)


def current_spool_bytes(runtime: SpoolRuntime) -> int:
    with runtime._spool_lock:
        return runtime._spool_bytes_total


def recover_spool_temporaries(runtime: SpoolRuntime) -> None:
    temporaries = sorted(
        (path for path in runtime.spool_dir.glob("*.tmp") if path.stem.isdigit()),
        key=lambda path: (int(path.stem), path.name),
    )
    occupied = {
        int(path.stem) for path in runtime.spool_dir.glob("*.rollout") if path.stem.isdigit()
    }
    reserved = occupied | {int(path.stem) for path in temporaries}
    next_sequence = max(reserved, default=-1) + 1
    for temporary in temporaries:
        if not runtime._valid_spool_temporary(temporary):
            continue
        sequence, next_sequence = _available_sequence(temporary, occupied, next_sequence)
        path = runtime.spool_dir / f"{sequence:020d}.rollout"
        os.replace(temporary, path)
        sync_checkpoint_path(path)
        occupied.add(sequence)
        logger.info("Recovered orphaned actor spool file %s", path.name)


def _available_sequence(temporary: Path, occupied: set[int], next_sequence: int) -> tuple[int, int]:
    sequence = int(temporary.stem)
    if sequence not in occupied:
        return sequence, next_sequence
    return next_sequence, next_sequence + 1


def valid_spool_temporary(runtime: SpoolRuntime, path: Path) -> bool:
    try:
        value = _decode_spool_temporary(runtime, path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Leaving invalid actor spool temporary %s in place: %s: %s",
            path.name,
            type(exc).__name__,
            exc,
        )
        return False
    if isinstance(value, Mapping):
        return True
    logger.warning("Leaving non-mapping actor spool temporary %s in place", path.name)
    return False


def _decode_spool_temporary(runtime: SpoolRuntime, path: Path) -> Any:
    size = path.stat().st_size
    if size > runtime.codec.max_message_bytes:
        raise ValueError(
            f"compressed payload is {size} bytes; limit is {runtime.codec.max_message_bytes}"
        )
    return runtime.codec.decode(path.read_bytes())


def scan_spool_bytes(runtime: SpoolRuntime) -> int:
    total = 0
    paths = (
        path
        for path in runtime.spool_dir.iterdir()
        if path.stem.isdigit() and path.suffix in {".rollout", ".tmp"}
    )
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def next_sequence(runtime: SpoolRuntime) -> int:
    existing = [
        int(path.stem)
        for path in runtime.spool_dir.iterdir()
        if path.stem.isdigit() and path.suffix in {".rollout", ".tmp"}
    ]
    return max(existing, default=-1) + 1
