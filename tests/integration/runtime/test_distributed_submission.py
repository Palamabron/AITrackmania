from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from google.protobuf.wrappers_pb2 import BytesValue

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
    _ephemeral_port,
    _resolved_run,
    _transition,
    _TransitionSpec,
)
from trackmaniarl.distributed.coordinator import (
    Coordinator,
)
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    transition_to_wire,
)


def _coordinator(tmp_path: Path, run_id: str) -> Coordinator:
    training = {
        "total_transitions": 10,
        "warmup_transitions": 10,
        "checkpoint_interval_updates": None,
    }
    run = _resolved_run(tmp_path, run_id, training)
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def _base_payload(sequence: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
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
    return BytesValue(value=coordinator.codec.encode(payload))


def _submit(coordinator: Coordinator, request: BytesValue) -> BytesValue:
    context = cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}"))
    return coordinator._submit(request, context)


def _close(coordinator: Coordinator) -> None:
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def _malformed_payload(coordinator: Coordinator, malformation: str) -> dict[str, Any]:
    transition = transition_to_wire(_transition(_TransitionSpec("actor", 0, 1.0)))
    payload = _base_payload(0)
    payload["transitions"] = [transition]
    match malformation:
        case "transitions_not_list":
            payload["transitions"] = {}
        case "non_finite_reward":
            transition["reward"] = torch.tensor(float("nan"))
        case "invalid_action":
            transition["action"] = b"not-a-numeric-pytree"
        case "incomplete_episode":
            _set_incomplete_episode(payload)
        case _:
            _set_invalid_snapshot(coordinator, payload)
    return payload


def _set_incomplete_episode(payload: dict[str, Any]) -> None:
    payload["transitions"] = []
    payload["episodes"] = [{"finished": True, "finish_time_s": 1.0}]


def _set_invalid_snapshot(coordinator: Coordinator, payload: dict[str, Any]) -> None:
    payload["transitions"] = []
    payload["evaluations"] = [
        {"finished": True, "finish_time_s": 1.0, "policy_version": 1, "steps": 1}
    ]
    payload["evaluation_snapshot"] = coordinator.codec.encode(["not", "a", "mapping"])


def _empty_request(coordinator: Coordinator, sequence: int) -> BytesValue:
    return _request(coordinator, _base_payload(sequence))


def _transition_request(coordinator: Coordinator, sequence: int) -> BytesValue:
    payload = _base_payload(sequence)
    spec = _TransitionSpec("actor", sequence, 1.0)
    payload["transitions"] = [transition_to_wire(_transition(spec))]
    return _request(coordinator, payload)


def _fail_append(session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
    del session_id, sequence, payload
    raise OSError("simulated WAL I/O failure")


def _induce_incidents(coordinator: Coordinator, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    lag = coordinator.run.spec.distributed.hard_policy_lag_updates
    coordinator.counters.updates = lag + 1
    rejected = _submit(coordinator, _transition_request(coordinator, 0))
    coordinator.counters.updates = 0
    monkeypatch.setattr(coordinator.journal, "append", _fail_append)
    with pytest.raises(OSError, match="simulated WAL I/O failure"):
        _submit(coordinator, _transition_request(coordinator, 1))
    return coordinator.codec.decode(rejected.value)


def test_malformed_rollout_is_rejected_before_wal_append(tmp_path: Path) -> None:
    malformations = (
        "transitions_not_list",
        "non_finite_reward",
        "invalid_action",
        "incomplete_episode",
        "invalid_evaluation_snapshot",
    )
    for malformation in malformations:
        coordinator = _coordinator(tmp_path, f"invalid-before-wal-{malformation}")
        request = _request(coordinator, _malformed_payload(coordinator, malformation))
        try:
            with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
                _submit(coordinator, request)
            assert not coordinator.journal.has_rows()
        finally:
            _close(coordinator)


def test_evaluation_waits_for_the_first_trained_policy_snapshot(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "evaluation-after-warmup")
    codec = coordinator.codec
    coordinator._evaluation_due.add("actor")

    try:
        before_training = codec.decode(_submit(coordinator, _empty_request(coordinator, 0)).value)
        assert before_training["evaluate"] is False
        assert "actor" in coordinator._evaluation_due

        coordinator.counters.policy_version = 1
        coordinator._policy_payload = codec.encode({"model": {}})
        after_training = codec.decode(_submit(coordinator, _empty_request(coordinator, 1)).value)
        assert after_training["evaluate"] is True
        assert "actor" not in coordinator._evaluation_due
    finally:
        _close(coordinator)


def test_rollout_rejection_and_wal_failure_emit_reasoned_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path, "runtime-incidents")
    try:
        _assert_incidents(coordinator, _induce_incidents(coordinator, monkeypatch))
    finally:
        _close(coordinator)


def _assert_incidents(coordinator: Coordinator, rejected: dict[str, Any]) -> None:
    assert rejected["reason"] == "hard_policy_lag"
    logger = coordinator.run.logger
    assert logger.events.count("distributed/rollout_rejected") == 1
    assert logger.events.count("distributed/wal_error") == 1
    incidents = dict(logger.records)
    assert incidents["distributed/rollout_rejected"]["reason"] == "hard_policy_lag"
    assert incidents["distributed/wal_error"]["operation"] == "append"


def _stale_evaluation_payload(coordinator: Coordinator) -> dict[str, Any]:
    maximum = coordinator.run.spec.distributed.hard_policy_lag_updates
    coordinator.counters.updates = maximum + 1
    payload = _base_payload(0)
    payload["evaluations"] = [
        {"finished": True, "finish_time_s": 36.0, "steps": 1, "policy_version": 0}
    ]
    return payload


def test_stale_evaluation_only_payload_bypasses_hard_policy_lag(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "stale-evaluation")
    run = coordinator.run
    codec = coordinator.codec
    try:
        payload = _stale_evaluation_payload(coordinator)
        response = _submit(coordinator, _request(coordinator, payload))
        decoded = codec.decode(response.value)
        assert decoded["accepted"] is True
        assert decoded["force_refresh"] is True
        assert coordinator.journal.has_rows()
        assert run.logger.events.count("distributed/rollout_rejected") == 0
    finally:
        _close(coordinator)
