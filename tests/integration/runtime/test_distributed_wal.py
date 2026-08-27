from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from google.protobuf.wrappers_pb2 import BytesValue

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
    _ephemeral_port,
    _resolved_run,
    _transition,
    _TransitionSpec,
    _TransitionState,
)
from trackmaniarl.distributed.coordinator import (
    Coordinator,
)
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    transition_to_wire,
)


def _coordinator(tmp_path: Path, run_id: str, total: int = 10) -> Coordinator:
    training = {
        "total_transitions": total,
        "warmup_transitions": total,
        "checkpoint_interval_updates": None,
    }
    run = _resolved_run(tmp_path, run_id, training)
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def _close(coordinator: Coordinator) -> None:
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def _payload(sequence: int, state: _TransitionState) -> dict[str, Any]:
    spec = _TransitionSpec("actor", sequence, float(sequence), state)
    return {
        "actor_id": "actor",
        "session_id": "session",
        "sequence": sequence,
        "policy_version": 0,
        "transitions": [transition_to_wire(_transition(spec))],
        "episodes": [],
        "evaluations": [],
        "evaluation_snapshot": b"",
    }


def _empty_payload(sequence: int) -> dict[str, Any]:
    return {
        "actor_id": "actor",
        "session_id": "session",
        "sequence": sequence,
        "policy_version": 0,
        "transitions": [],
        "episodes": [],
        "evaluations": [],
        "evaluation_snapshot": b"",
    }


def _request(coordinator: Coordinator, payload: dict[str, Any]) -> BytesValue:
    wire = {"protocol_version": PROTOCOL_VERSION, "fingerprint": "fingerprint", **payload}
    return BytesValue(value=coordinator.codec.encode(wire))


def _submit(coordinator: Coordinator, request: BytesValue) -> BytesValue:
    context = cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}"))
    return coordinator._submit(request, context)


@dataclass(slots=True)
class _DelayedAppend:
    append: Any
    committed: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def __call__(self, session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
        result = self.append(session_id, sequence, payload)
        if sequence == 0:
            self.committed.set()
            assert self.release.wait(timeout=2.0)
        return result


def _submit_capture(
    coordinator: Coordinator, request: BytesValue, failures: list[BaseException]
) -> None:
    try:
        _submit(coordinator, request)
    except BaseException as exc:
        failures.append(exc)


def _run_concurrent_submissions(
    coordinator: Coordinator, delayed: _DelayedAppend
) -> tuple[list[BaseException], threading.Thread, BytesValue]:
    failures: list[BaseException] = []
    first_request = _request(coordinator, _payload(0, _TransitionState.CONTINUES))
    first = threading.Thread(target=_submit_capture, args=(coordinator, first_request, failures))
    first.start()
    assert delayed.committed.wait(timeout=2.0)
    second_request = _request(coordinator, _payload(1, _TransitionState.TERMINATES))
    response = _submit(coordinator, second_request)
    coordinator._drain_rollouts(2)
    delayed.release.set()
    first.join(timeout=2.0)
    return failures, first, response


def test_concurrent_submit_wakes_do_not_reorder_journal_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path, "ordered-wal")
    delayed = _DelayedAppend(coordinator.journal.append)
    monkeypatch.setattr(coordinator.journal, "append", delayed)
    failures, first, response = _run_concurrent_submissions(coordinator, delayed)

    try:
        assert coordinator.codec.decode(response.value)["accepted"]
        assert not failures
        assert not first.is_alive()
        assert [item.action for item in coordinator.run.replay_store.get([0, 1])] == [0, 1]
        assert coordinator.counters.journal_applied_frontier == 2
    finally:
        delayed.release.set()
        _close(coordinator)


def test_journal_recovery_crosses_internal_batch_boundary(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "large-recovery", 400)
    _append_journal_transitions(coordinator, 300)

    try:
        coordinator._recover_journal(0)
        run = coordinator.run
        assert len(run.replay_store) == 300
        assert coordinator.counters.transitions == 300
        assert coordinator.counters.journal_applied_frontier == 300
        assert run.logger.events.count("distributed/wal_recovery") == 1
        recovery = dict(run.logger.records)["distributed/wal_recovery"]
        assert recovery["rows"] == 300
        assert recovery["to_frontier"] == 300
    finally:
        _close(coordinator)


def _append_journal_transitions(coordinator: Coordinator, count: int) -> None:
    for sequence in range(count):
        state = _TransitionState.TERMINATES if sequence == count - 1 else _TransitionState.CONTINUES
        value = _payload(sequence, state)
        value["transitions"][0]["reward"] = 1.0
        coordinator.journal.append("session", sequence, coordinator.codec.encode(value))


def test_corrupt_recovery_row_emits_wal_error(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "corrupt-recovery")
    coordinator.journal.append("session", 0, b"not-a-wire-payload")

    try:
        with pytest.raises(ValueError, match="wire payload"):
            coordinator._recover_journal(0)

        incident = dict(coordinator.run.logger.records)["distributed/wal_error"]
        assert coordinator.run.logger.events.count("distributed/wal_error") == 1
        assert incident["operation"] == "recovery_decode"
        assert "distributed/wal_recovery" not in coordinator.run.logger.events
    finally:
        _close(coordinator)


def test_checkpoint_prunes_only_applied_frontier_and_recovers_tail(tmp_path: Path) -> None:
    first = _coordinator(tmp_path, "crash-recovery")
    _append_journal_transitions(first, 2)
    first._drain_rollouts(1)
    checkpoint = first._checkpoint()
    first._checkpoint_writer.wait()
    assert first.journal.pruned_through == 1
    assert [row_id for row_id, _ in first.journal.rows_after(0)] == [2]
    assert first.run.logger.events.count("train/checkpoint_completed") == 1
    _close(first)
    _assert_restored_tail(tmp_path, checkpoint)


def _assert_restored_tail(tmp_path: Path, checkpoint: Path) -> None:
    resumed = _coordinator(tmp_path, "crash-recovery")
    try:
        resumed.restore_checkpoint(checkpoint)
        assert len(resumed.run.replay_store) == 2
        assert resumed.counters.transitions == 2
        assert resumed.counters.journal_applied_frontier == 2
        assert [item.step for item in resumed.run.replay_store.get([0, 1])] == [0, 1]
    finally:
        _close(resumed)


def test_submit_retry_after_lost_response_is_ingested_exactly_once(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "lost-submit-response")
    payload = _payload(0, _TransitionState.CONTINUES)
    payload["transitions"][0]["reward"] = 1.0
    request = _request(coordinator, payload)
    try:
        _submit(coordinator, request)
        retried = coordinator.codec.decode(_submit(coordinator, request).value)
        coordinator._drain_rollouts(10)
        assert retried["accepted"] is True
        assert retried["duplicate"] is True
        assert len(list(coordinator.journal.rows_after(0))) == 1
        assert len(coordinator.run.replay_store) == 1
        assert coordinator.counters.transitions == 1
    finally:
        _close(coordinator)
