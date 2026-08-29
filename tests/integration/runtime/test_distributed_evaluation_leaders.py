from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.integration.runtime.distributed_evaluation_leader_support import (
    _candidate,
    _DeferredLeaderCheckpointProbe,
    _delayed_failure_scenario,
    _FailingLeaderProbe,
    _StepRecordingLogger,
)
from tests.integration.runtime.distributed_evaluation_support import (
    _evaluation_coordinator,
    _EvaluationIngestor,
    _finished_evaluation,
)
from trackmaniarl.distributed.coordinator_checkpoint import EvaluationCheckpointKind
from trackmaniarl.distributed.coordinator_leaders import record_evaluation_leaders

_RELIABLE = EvaluationCheckpointKind.RELIABLE
_FASTEST = EvaluationCheckpointKind.FASTEST


class _PartialPromotionProbe:
    def __init__(self, directory: Path, coordinator: Any) -> None:
        self.directory = directory
        self.coordinator = coordinator
        self.ranks: list[tuple[object, object]] = []

    def save(self, evaluated: Any) -> Path:
        self.ranks.append((self.coordinator._best_evaluation, self.coordinator._fastest_evaluation))
        path = self.directory / f"{evaluated.kind.value}.pt"
        if evaluated.kind is _RELIABLE:
            assert evaluated.on_saved is not None
            evaluated.on_saved(path)
        else:
            assert evaluated.on_failed is not None
            evaluated.on_failed(OSError("disk full"))
        return path


def test_fastest_leader_does_not_replace_the_reliable_leader(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    checkpoints: list[EvaluationCheckpointKind] = []
    ingestor = _EvaluationIngestor(_evaluation_coordinator(tmp_path, events, checkpoints))
    ingestor.ingest([_finished_evaluation(40.0), _finished_evaluation(60.0)])
    ingestor.ingest([_finished_evaluation(39.0), _finished_evaluation(70.0)])
    assert checkpoints == [_RELIABLE, _FASTEST, _FASTEST]
    reliable = [payload for event, payload in events if event == "eval/best_checkpoint"]
    fastest = [payload for event, payload in events if event == "eval/fastest_checkpoint"]
    assert [item["finish_time_median_s"] for item in reliable] == [50.0]
    assert [item["finish_time_best_s"] for item in fastest] == [40.0, 39.0]
    assert [item["shared_with_reliable"] for item in fastest] == [0.0, 0.0]


def test_reliable_leader_prioritizes_finish_rate_before_median(tmp_path: Path) -> None:
    checkpoints: list[EvaluationCheckpointKind] = []
    coordinator = _evaluation_coordinator(tmp_path, [], checkpoints)
    coordinator.run.spec.evaluation.min_finish_rate = 0.6
    coordinator._evaluation_policy_states[42] = {"weight": 2.0}
    record_evaluation_leaders(coordinator, _candidate(41, 1.0, (42.0, 41.0)))
    record_evaluation_leaders(coordinator, _candidate(42, 0.6, (40.0, 39.0)))
    assert checkpoints.count(_RELIABLE) == 1
    assert checkpoints.count(_FASTEST) == 2


def test_leader_checkpoint_requires_the_exact_evaluated_policy(tmp_path: Path) -> None:
    checkpoints: list[EvaluationCheckpointKind] = []
    coordinator = _evaluation_coordinator(tmp_path, [], checkpoints)
    coordinator._evaluation_policy_states.clear()
    with pytest.raises(RuntimeError, match="exact snapshot missing"):
        record_evaluation_leaders(coordinator, _candidate(41, 1.0, (42.0, 41.0)))
    assert checkpoints == []
    assert coordinator._best_evaluation is None
    assert coordinator._fastest_evaluation is None


def test_failed_leader_checkpoint_can_retry_the_same_candidate(tmp_path: Path) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    coordinator.run.spec.evaluation.min_finish_rate = 1.0
    probe = _FailingLeaderProbe(tmp_path)
    coordinator._checkpoint = probe.save
    candidate = _candidate(41, 0.8, (42.0, 41.0))
    record_evaluation_leaders(coordinator, candidate)
    record_evaluation_leaders(coordinator, candidate)
    assert probe.attempts == 2
    assert coordinator._fastest_evaluation is None


def test_partial_dual_promotion_persists_only_the_durable_rank(tmp_path: Path) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    probe = _PartialPromotionProbe(tmp_path, coordinator)
    coordinator._checkpoint = probe.save

    record_evaluation_leaders(coordinator, _candidate(41, 1.0, (42.0, 41.0)))

    reliable_rank, fastest_rank = probe.ranks[0]
    assert reliable_rank is not None
    assert fastest_rank is None
    assert coordinator._best_evaluation == reliable_rank
    assert coordinator._fastest_evaluation is None

    retried: list[EvaluationCheckpointKind] = []
    restarted = _evaluation_coordinator(tmp_path, [], retried)
    restarted._best_evaluation = reliable_rank
    restarted._fastest_evaluation = fastest_rank
    record_evaluation_leaders(restarted, _candidate(41, 1.0, (42.0, 41.0)))
    assert retried == [_FASTEST]


def test_delayed_callback_logs_the_update_captured_when_queued(tmp_path: Path) -> None:
    coordinator = _evaluation_coordinator(tmp_path, [], [])
    logger = _StepRecordingLogger()
    coordinator.run.logger = logger
    coordinator.run.spec.evaluation.min_finish_rate = 1.0
    coordinator.counters.updates = 17
    probe = _DeferredLeaderCheckpointProbe(tmp_path)
    coordinator._checkpoint = probe.save
    record_evaluation_leaders(coordinator, _candidate(41, 0.8, (42.0, 41.0)))
    coordinator.counters.updates = 99
    probe.complete()
    assert logger.records == [("eval/fastest_checkpoint", 17)]


def test_pending_failure_resolves_before_the_next_candidate(tmp_path: Path) -> None:
    scenario = _delayed_failure_scenario(tmp_path)
    try:
        thread, first_rank = scenario.begin()
        scenario.resolve_failure(thread, first_rank)
        scenario.retry()
    finally:
        scenario.close()
