from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _append_journal_then_exit,
    _ephemeral_port,
    _resolved_run,
    _submit_rollout_then_exit,
)
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_support import _AsyncCheckpointWriter, _CheckpointWrite
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.journal import JournalPayloadConflictError, RolloutJournal


def _training() -> dict[str, int | None]:
    return {
        "total_transitions": 10,
        "warmup_transitions": 10,
        "checkpoint_interval_updates": None,
    }


def _coordinator(run: Any) -> Coordinator:
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def _close(coordinator: Coordinator) -> None:
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def test_rollout_journal_is_idempotent_and_recovers_rows(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.sqlite3"
    journal = RolloutJournal(path)
    first_id, inserted = journal.append("session", 0, b"first")
    duplicate_id, duplicate_inserted = journal.append("session", 0, b"first")
    with pytest.raises(JournalPayloadConflictError, match="different payload"):
        journal.append("session", 0, b"ignored")
    second_id, second_inserted = journal.append("session", 1, b"second")
    profile = journal.actor_profile("PC-1", 4)
    journal.close()
    reopened = RolloutJournal(path)
    _assert_reopened_journal(
        reopened,
        (first_id, second_id, duplicate_id),
        (inserted, second_inserted, duplicate_inserted),
    )
    assert reopened.actor_profile("PC-1", 4) == profile
    assert reopened.identity == journal.identity
    reopened.close()


def _assert_reopened_journal(
    journal: RolloutJournal, row_ids: tuple[int, int, int], insertions: tuple[bool, bool, bool]
) -> None:
    first_id, second_id, duplicate_id = row_ids
    inserted, second_inserted, duplicate_inserted = insertions
    assert inserted
    assert second_inserted
    assert not duplicate_inserted
    assert duplicate_id == first_id
    assert list(journal.rows_after(first_id)) == [(second_id, b"second")]


def test_committed_journal_row_survives_abrupt_process_exit(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.sqlite3"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_append_journal_then_exit,
        args=(str(path), b"committed-before-crash"),
    )

    process.start()
    process.join(timeout=15.0)

    assert not process.is_alive()
    assert process.exitcode == 23
    journal = RolloutJournal(path)
    try:
        assert list(journal.rows_after(0)) == [(1, b"committed-before-crash")]
    finally:
        journal.close()


def test_coordinator_recovers_submit_committed_before_learner_process_exit(
    tmp_path: Path,
) -> None:
    _assert_crashed_submit(tmp_path)
    run = _resolved_run(tmp_path, "learner-crash-after-submit", _training())
    coordinator = Coordinator(
        run, CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, "fingerprint")
    )
    try:
        assert len(run.replay_store) == 0
        coordinator._recover_journal(0)
        _assert_crash_recovery(coordinator)
    finally:
        _close(coordinator)


def _assert_crashed_submit(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_submit_rollout_then_exit, args=(str(tmp_path),))
    process.start()
    process.join(timeout=15.0)
    assert not process.is_alive()
    assert process.exitcode == 25


def _assert_crash_recovery(coordinator: Coordinator) -> None:
    run = coordinator.run
    assert len(run.replay_store) == 1
    assert coordinator.counters.transitions == 1
    assert coordinator.counters.journal_applied_frontier == 1
    assert run.replay_store.get([0])[0].step == 7
    recovery = dict(run.logger.records)["distributed/wal_recovery"]
    assert recovery["rows"] == 1
    assert recovery["transitions"] == 1


def test_rollout_receipt_survives_prune_and_rejects_unsafe_rollback(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.sqlite3"
    journal = RolloutJournal(path)
    foreign = RolloutJournal(tmp_path / "foreign.sqlite3")
    try:
        row_id, inserted = journal.append("session", 0, b"payload")
        assert inserted

        journal.prune(row_id)
        journal.close()
        journal = RolloutJournal(path)

        assert journal.pruned_through == row_id
        assert not journal.has_rows()
        assert journal.append("session", 0, b"payload") == (row_id, False)
        _assert_unsafe_rollbacks(journal, foreign, row_id)
    finally:
        foreign.close()
        journal.close()


def _assert_unsafe_rollbacks(journal: RolloutJournal, foreign: RolloutJournal, row_id: int) -> None:
    with pytest.raises(ValueError, match="predates data already pruned"):
        journal.validate_checkpoint(journal.identity, row_id - 1)
    with pytest.raises(ValueError, match="different rollout journal"):
        foreign.validate_checkpoint(journal.identity, 0)
    with pytest.raises(ValueError, match="ahead of durable WAL history"):
        journal.validate_checkpoint(journal.identity, row_id + 1)


class _PartialWriteCodec:
    def save(self, state: Mapping[str, Any], path: Path) -> None:
        del state
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
        raise OSError("simulated checkpoint crash")


def test_failed_checkpoint_never_advances_the_pruned_frontier(tmp_path: Path) -> None:
    journal = RolloutJournal(tmp_path / "rollouts.sqlite3")
    row_id, _ = journal.append("session", 0, b"payload")
    writer = _AsyncCheckpointWriter(_PartialWriteCodec())
    try:
        writer.submit(
            _CheckpointWrite({}, tmp_path / "checkpoint.pt", lambda: journal.prune(row_id), None)
        )
        with pytest.raises(OSError, match="simulated checkpoint crash"):
            writer.wait()

        assert journal.pruned_through == 0
        assert [stored_id for stored_id, _ in journal.rows_after(0)] == [row_id]
    finally:
        writer.close()
        journal.close()


@dataclass(slots=True)
class _CheckpointCapture:
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    saved: dict[str, Any] = field(default_factory=dict)


class _DelayedCodec:
    def __init__(self, capture: _CheckpointCapture) -> None:
        self.capture = capture

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        self.capture.started.set()
        if not self.capture.release.wait(timeout=5.0):
            raise TimeoutError("checkpoint test did not release the codec")
        self.capture.saved["state"] = state
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")


class _MutableState:
    def __init__(self, value: str) -> None:
        self.value: dict[str, Any] = {
            "nested": {"value": value},
            "tensor": torch.tensor([1.0]),
        }

    def state_dict(self) -> Mapping[str, Any]:
        return self.value


def _snapshot_coordinator(
    tmp_path: Path, capture: _CheckpointCapture
) -> tuple[Coordinator, _MutableState, _MutableState]:
    replay = _MutableState("replay-before")
    sampler = _MutableState("sampler-before")
    run = replace(
        _resolved_run(tmp_path, "checkpoint-snapshot", _training()),
        replay_store=cast(Any, replay),
        sampler=cast(Any, sampler),
        checkpoint_codec=cast(Any, _DelayedCodec(capture)),
    )
    return _coordinator(run), replay, sampler


def _mutate_state(state: _MutableState, name: str) -> None:
    state.value["nested"]["value"] = name
    state.value["tensor"].fill_(2.0)


def _assert_snapshot(checkpoint: Mapping[str, Any]) -> None:
    assert checkpoint["replay_store"]["nested"]["value"] == "replay-before"
    assert torch.equal(checkpoint["replay_store"]["tensor"], torch.tensor([1.0]))
    assert checkpoint["sampler"]["nested"]["value"] == "sampler-before"
    assert torch.equal(checkpoint["sampler"]["tensor"], torch.tensor([1.0]))


def test_async_checkpoint_snapshots_mutable_replay_and_sampler_state(tmp_path: Path) -> None:
    capture = _CheckpointCapture()
    coordinator, replay, sampler = _snapshot_coordinator(tmp_path, capture)

    try:
        coordinator._checkpoint()
        assert capture.started.wait(timeout=5.0)
        _mutate_state(replay, "replay-after")
        _mutate_state(sampler, "sampler-after")
        capture.release.set()
        coordinator._checkpoint_writer.wait()
        _assert_snapshot(capture.saved["state"])
    finally:
        capture.release.set()
        _close(coordinator)
