from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.integration.runtime.distributed_runtime_support import (
    _transition,
    _TransitionSpec,
    _TransitionState,
)
from tests.integration.runtime.test_distributed_submission import (
    _base_payload,
    _close,
    _coordinator,
    _request,
    _submit,
)
from trackmaniarl.core.replay import PrioritizedSampler
from trackmaniarl.distributed.actor_metrics import summarize_episode
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.protocol import transition_to_wire

_EPISODE_ID = "actor/session/episode"


class _EpisodeOutcome(StrEnum):
    FINISHED = "finished"
    FAILED = "time_limit"


class _SummaryIdentity(StrEnum):
    IDENTIFIED = "identified"
    LEGACY = "legacy"


_INVALID_EPISODE_UPDATES: tuple[dict[str, object], ...] = (
    {"episode_id": ""},
    {"episode_id": 7},
    {"episode_id": "actor/session/"},
    {"episode_id": "other/session/episode"},
    {"finish_time_s": 0.0},
    {"termination": "time_limit"},
    {"race_time_s": 35.0},
    {"finished": 0.0, "termination": "time_limit"},
    {"finish_time_s": 1e100, "race_time_s": 1e100},
    {"finish_time_s": 1e-50, "race_time_s": 1e-50},
)


def _episode_summary(
    episode_id: str, outcome: _EpisodeOutcome, finish_time_s: float
) -> dict[str, Any]:
    summary = summarize_episode(
        1.0,
        {
            "policy_version": 0,
            "termination_reason": outcome.value,
            "race_time_ms": finish_time_s * 1_000.0,
        },
        2,
    )
    return {**summary, "episode_id": episode_id}


def _summary_payload(
    sequence: int, outcome: _EpisodeOutcome, finish_time_s: float
) -> dict[str, Any]:
    payload = _base_payload(sequence)
    payload["episodes"] = [_episode_summary(_EPISODE_ID, outcome, finish_time_s)]
    return payload


def _elite_coordinator(tmp_path: Path, run_id: str) -> Coordinator:
    coordinator = _coordinator(tmp_path, run_id)
    sampler = PrioritizedSampler(coordinator.run.feature_pipeline, elite_time_s=37.0, seed=0)
    object.__setattr__(coordinator.run, "sampler", sampler)
    return coordinator


def _episode_event(coordinator: Coordinator) -> dict[str, Any]:
    event = dict(coordinator.run.logger.records)["train/episode"]
    return dict(event)


def _episode_paces(coordinator: Coordinator) -> list[float]:
    store = coordinator.run.replay_store
    return [store.sampling_pace_s(index) for index in (0, 1)]


def _assert_paces(coordinator: Coordinator, expected: float | None) -> None:
    paces = _episode_paces(coordinator)
    if expected is None:
        assert all(np.isinf(pace) for pace in paces)
        return
    assert paces == pytest.approx([expected, expected])


def _submit_episode_chunks(coordinator: Coordinator, sequences: tuple[int, int]) -> None:
    for step, sequence in enumerate(sequences):
        payload = _base_payload(sequence)
        state = _TransitionState.TERMINATES if step == 1 else _TransitionState.CONTINUES
        transition = _transition(_TransitionSpec("actor", step, 1.0, state))
        payload["transitions"] = [transition_to_wire(transition)]
        _submit(coordinator, _request(coordinator, payload))


@pytest.mark.parametrize("updates", _INVALID_EPISODE_UPDATES)
def test_invalid_episode_outcome_is_rejected_before_wal(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    coordinator = _coordinator(tmp_path, "invalid-finished-summary")
    payload = _summary_payload(0, _EpisodeOutcome.FINISHED, 36.0)
    payload["episodes"][0].update(updates)
    try:
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            _submit(coordinator, _request(coordinator, payload))
        assert not coordinator.journal.has_rows()
    finally:
        _close(coordinator)


def test_legacy_summary_without_identity_is_accepted_without_relabel(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "legacy-finished-summary")
    payload = _summary_payload(0, _EpisodeOutcome.FINISHED, 36.0)
    payload["episodes"][0].pop("episode_id")
    try:
        response = _submit(coordinator, _request(coordinator, payload))
        assert coordinator.codec.decode(response.value)["accepted"]
        coordinator._drain_rollouts(1)
        assert _episode_event(coordinator)["replay/labeled_transitions"] == 0
    finally:
        _close(coordinator)


def test_finished_summary_relabels_transitions_from_earlier_chunks(tmp_path: Path) -> None:
    coordinator = _elite_coordinator(tmp_path, "completed-episode-pace")
    payload = _summary_payload(2, _EpisodeOutcome.FINISHED, 36.5)
    try:
        _submit_episode_chunks(coordinator, (0, 1))
        _submit(coordinator, _request(coordinator, payload))
        coordinator._drain_rollouts(10)
        _assert_paces(coordinator, 36.5)
        event = _episode_event(coordinator)
        assert event["episode_id"] == _EPISODE_ID
        assert event["replay/labeled_transitions"] == 2
    finally:
        _close(coordinator)


def test_disabled_elite_sampling_skips_episode_relabel(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path, "disabled-episode-pace")
    payload = _summary_payload(2, _EpisodeOutcome.FINISHED, 36.5)
    try:
        _submit_episode_chunks(coordinator, (0, 1))
        _submit(coordinator, _request(coordinator, payload))
        coordinator._drain_rollouts(10)
        _assert_paces(coordinator, None)
        assert _episode_event(coordinator)["replay/labeled_transitions"] == 0
    finally:
        _close(coordinator)


@pytest.mark.parametrize("outcome", tuple(_EpisodeOutcome))
def test_summary_first_only_labels_finished(tmp_path: Path, outcome: _EpisodeOutcome) -> None:
    finish_time_s = 36.5 if outcome is _EpisodeOutcome.FINISHED else 0.0
    coordinator = _elite_coordinator(tmp_path, f"summary-first-{outcome}")
    payload = _summary_payload(0, outcome, finish_time_s)
    try:
        _submit(coordinator, _request(coordinator, payload))
        _submit_episode_chunks(coordinator, (1, 2))
        coordinator._drain_rollouts(10)
        expected = 36.5 if outcome is _EpisodeOutcome.FINISHED else None
        _assert_paces(coordinator, expected)
        assert _episode_event(coordinator)["replay/labeled_transitions"] == 0
    finally:
        _close(coordinator)


def _failed_evaluation_payload(coordinator: Coordinator) -> dict[str, Any]:
    payload = _summary_payload(2, _EpisodeOutcome.FAILED, 0.0)
    payload["evaluations"] = [
        {"finished": True, "finish_time_s": 35.0, "policy_version": 0, "steps": 1}
    ]
    payload["evaluation_snapshot"] = coordinator.codec.encode({"model": {}})
    return payload


def test_failed_and_eval_summaries_do_not_relabel(tmp_path: Path) -> None:
    coordinator = _elite_coordinator(tmp_path, "non-finish-pace")
    coordinator._best_evaluation = (1.0, 0.0, 0.0)
    coordinator._fastest_evaluation = (0.0, 1.0, 0.0)
    try:
        _submit_episode_chunks(coordinator, (0, 1))
        _submit(coordinator, _request(coordinator, _failed_evaluation_payload(coordinator)))
        coordinator._drain_rollouts(10)
        _assert_paces(coordinator, None)
        assert _episode_event(coordinator)["replay/labeled_transitions"] == 0
    finally:
        _close(coordinator)


def _append_recovery_rows(coordinator: Coordinator, identity: _SummaryIdentity) -> None:
    transition_payload = _base_payload(0)
    transition = _transition(_TransitionSpec("actor", 0, 1.0))
    transition_payload["transitions"] = [transition_to_wire(transition)]
    summary_payload = _summary_payload(1, _EpisodeOutcome.FINISHED, 36.25)
    if identity is _SummaryIdentity.LEGACY:
        summary_payload["episodes"][0].pop("episode_id")
    codec = coordinator.codec
    coordinator.journal.append("session", 0, codec.encode(transition_payload))
    coordinator.journal.append("session", 1, codec.encode(summary_payload))


@pytest.mark.parametrize("identity", tuple(_SummaryIdentity))
def test_wal_recovery_requires_episode_id(tmp_path: Path, identity: _SummaryIdentity) -> None:
    coordinator = _elite_coordinator(tmp_path, f"recovered-episode-pace-{identity}")
    _append_recovery_rows(coordinator, identity)
    try:
        coordinator._recover_journal(0)
        pace = coordinator.run.replay_store.sampling_pace_s(0)
        expected = pytest.approx(36.25) if identity is _SummaryIdentity.IDENTIFIED else np.inf
        assert pace == expected
        assert "train/episode" not in coordinator.run.logger.events
        assert coordinator.counters.journal_applied_frontier == 2
    finally:
        _close(coordinator)
