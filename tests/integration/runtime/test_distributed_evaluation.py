from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from tests.integration.runtime.distributed_evaluation_support import (
    _evaluation_coordinator,
    _EvaluationIngestor,
    _failed_evaluation,
    _finished_evaluation,
    _queue_backlog,
    _stopped_coordinator,
)
from tests.integration.runtime.distributed_runtime_support import _Logger
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.distributed.actor_collection import summary as actor_summary
from trackmaniarl.distributed.actor_evaluation import EvaluationPlan, _evaluation_trials
from trackmaniarl.distributed.actor_requests import EvaluationEpisodeRequest
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_checkpoint import EvaluationCheckpointKind
from trackmaniarl.distributed.coordinator_support import _Counters
from trackmaniarl.distributed.coordinator_validation import _validate_evaluation_summary


def test_external_stop_does_not_ingest_or_train_a_queued_backlog(tmp_path: Path) -> None:
    events: list[str] = []
    coordinator = _stopped_coordinator(tmp_path, events)
    ingest = coordinator._ingest

    def tracking_ingest(value: Mapping[str, Any], row_id: int) -> None:
        events.append("ingest")
        ingest(value, row_id)

    coordinator._ingest = tracking_ingest
    _queue_backlog(coordinator)
    coordinator.run_forever()

    assert events == []
    assert coordinator.counters.transitions == 0
    assert coordinator.counters.updates == 0
    assert coordinator._rollouts.qsize() == 3


class _EvaluationPrewarmPolicy:
    def __init__(self) -> None:
        self.state = 0
        self.modes: list[PolicyMode] = []
        self.states: list[int] = []

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation
        self.modes.append(mode)
        self.states.append(self.state)
        self.state += 1
        return 0

    def reset_episode(self) -> None:
        self.state = 0


class _EvaluationPrewarmPipeline:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_episode(self) -> None:
        self.reset_count += 1

    def transform_observation(self, observation: Any) -> Any:
        return observation


class _EvaluationPrewarmEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *, seed: int) -> tuple[list[int], dict[str, Any]]:
        del seed
        self.reset_count += 1
        return [self.reset_count], {}

    def step(self, action: int) -> tuple[list[int], float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.step_count += 1
        return (
            [0],
            1.0,
            True,
            False,
            {
                "termination_reason": "finished",
                "finished": True,
                "race_time_ms": 1_000.0,
            },
        )


class _EvaluationPrewarmRuntime:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(training=SimpleNamespace(max_episode_steps=1))
        self.stop = Event()
        self._evaluation_index = 0

    @staticmethod
    def _reset_environment(request: Any) -> Any:
        observation, _ = request.environment.reset(seed=request.episode)
        return observation

    def _evaluate_episode(self, request: EvaluationEpisodeRequest) -> dict[str, Any]:
        from trackmaniarl.distributed.actor_evaluation import evaluate_episode

        return evaluate_episode(self, request)

    @staticmethod
    def _summary(reward: float, info: Mapping[str, Any], transitions: int) -> dict[str, Any]:
        return actor_summary(reward, info, transitions)

    def _evaluation_telemetry_failure(self, failure: Any) -> dict[str, Any]:
        self._evaluation_index += 1
        return {"termination": "telemetry_error", "policy_version": failure.version}


def test_distributed_evaluation_prewarms_before_a_fresh_measured_trial() -> None:
    runtime = _EvaluationPrewarmRuntime()
    environment = _EvaluationPrewarmEnvironment()
    pipeline = _EvaluationPrewarmPipeline()
    policy = _EvaluationPrewarmPolicy()

    summaries = _evaluation_trials(EvaluationPlan(runtime, environment, pipeline, policy, 4), 1)

    assert summaries[0]["termination"] == "finished"
    assert environment.reset_count == 2
    assert environment.step_count == 1
    assert pipeline.reset_count == 3
    assert policy.modes == [PolicyMode.EVALUATION, PolicyMode.EVALUATION]
    assert policy.states == [0, 0]


def test_ingest_aggregates_evaluation_batches_and_checkpoints_best(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    checkpoints: list[EvaluationCheckpointKind] = []
    ingestor = _EvaluationIngestor(_evaluation_coordinator(tmp_path, events, checkpoints))
    ingestor.ingest([_finished_evaluation(52.0), _failed_evaluation()])
    ingestor.ingest([_failed_evaluation(), _failed_evaluation()])
    ingestor.ingest([_finished_evaluation(50.0), _finished_evaluation(46.0)])

    summaries = [payload for event, payload in events if event == "eval/summary"]
    _assert_evaluation_outcomes(summaries)
    _assert_evaluation_observability(summaries[0])
    _assert_best_evaluation(events, checkpoints)


def _assert_evaluation_outcomes(summaries: list[dict[str, Any]]) -> None:
    assert [item["finish_rate"] for item in summaries] == [0.5, 0.0, 1.0]
    assert summaries[0]["finish_time_mean_s"] == pytest.approx(52.0)
    assert summaries[0]["policy_version"] == 41.0
    assert summaries[0]["q_margin_start_mean"] == pytest.approx(0.5)
    assert summaries[2]["finish_time_median_s"] == pytest.approx(48.0)
    assert summaries[2]["finish_time_best_s"] == pytest.approx(46.0)
    assert summaries[2]["sub_40_rate"] == 0.0


def _assert_evaluation_observability(summary: dict[str, Any]) -> None:
    assert summary["control_gas_fraction_mean"] == pytest.approx(0.75)
    assert summary["control_brake_fraction_mean"] == pytest.approx(0.2)
    assert summary["control_brake_tap_fraction_mean"] == pytest.approx(0.1)
    assert summary["control_steer_abs_mean"] == pytest.approx(0.6)
    assert summary["action_latency_ms"] == pytest.approx(2.5)
    assert summary["controller_apply_ms"] == pytest.approx(3.5)
    assert summary["telemetry_wait_ms"] == pytest.approx(6.5)
    assert summary["telemetry_skipped_frames_total"] == 5.0
    assert summary["telemetry_skipped_frames_mean"] == pytest.approx(0.125)
    assert summary["telemetry_skipped_frames_max"] == 3.0
    assert summary["telemetry_steps_with_skipped_frames_fraction"] == pytest.approx(0.125)
    assert summary["progress_bin/90_100/action_count"] == 4.0
    assert summary["progress_bin/90_100/q_margin_mean"] == 1.5


def _assert_best_evaluation(
    events: list[tuple[str, dict[str, Any]]], checkpoints: list[EvaluationCheckpointKind]
) -> None:
    progress = [payload for event, payload in events if event == "eval/progress_bin"]
    assert progress[0]["90_100/q_max_mean"] == 3.0
    assert checkpoints == [
        EvaluationCheckpointKind.FASTEST,
        EvaluationCheckpointKind.RELIABLE,
        EvaluationCheckpointKind.FASTEST,
    ]
    best = [payload for event, payload in events if event == "eval/best_checkpoint"]
    assert [item["finish_rate"] for item in best] == [1.0]
    assert best[0]["finish_time_median_s"] == pytest.approx(48.0)
    fastest = [payload for event, payload in events if event == "eval/fastest_checkpoint"]
    assert [item["finish_time_best_s"] for item in fastest] == [52.0, 46.0]
    assert [item["shared_with_reliable"] for item in fastest] == [0.0, 0.0]


def test_evaluation_summary_rejects_invalid_observability_metrics() -> None:
    invalid_metrics = (
        {"steps": -1},
        {"steps": 1.5},
        {"steps": 1, "controller_apply_ms_mean": -0.1},
        {"steps": 1, "telemetry_steps_with_skipped_frames_fraction": 1.1},
        {"steps": 1, "telemetry_skipped_frames_total": 0.5},
        {"steps": 1, "telemetry_skipped_frames_total": 1.0, "telemetry_skipped_frames_max": 2.0},
    )
    for invalid in invalid_metrics:
        summary = {"finished": 0.0, "finish_time_s": 0.0, "policy_version": 0, **invalid}
        with pytest.raises((TypeError, ValueError)):
            _validate_evaluation_summary(summary)


def test_evaluation_stop_requires_consecutive_successful_batches() -> None:
    coordinator = _evaluation_stop_coordinator()

    coordinator._record_evaluation_stop({"finish_rate": 0.2, "finish_time_median_s": 58.0})
    coordinator._record_evaluation_stop({"finish_rate": 0.9, "finish_time_median_s": 35.9})
    coordinator._record_evaluation_stop({"finish_rate": 0.8, "finish_time_median_s": 35.0})
    assert coordinator._evaluation_stop_reason is None
    coordinator._record_evaluation_stop({"finish_rate": 0.9, "finish_time_median_s": 36.0})
    coordinator._record_evaluation_stop({"finish_rate": 1.0, "finish_time_median_s": 35.8})
    assert coordinator._evaluation_stop_reason is not None
    assert "evaluation target passed 2 consecutive times" in coordinator._evaluation_stop_reason


def _evaluation_stop_coordinator() -> Coordinator:
    coordinator = object.__new__(Coordinator)
    training = SimpleNamespace(
        evaluation_stop_min_finish_rate=0.9,
        evaluation_stop_median_s=36.0,
        evaluation_stop_consecutive_batches=2,
    )
    coordinator.run = SimpleNamespace(spec=SimpleNamespace(training=training), logger=_Logger())
    coordinator.counters = _Counters(updates=10)
    coordinator._consecutive_evaluation_passes = 0
    coordinator._evaluation_stop_reason = None
    return coordinator
