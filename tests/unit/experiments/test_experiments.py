"""Tests for deterministic evaluation and autonomous-strategy fallbacks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trackmaniarl.experiments.evaluation import (
    STANDARD_METRICS,
    EvaluationResult,
    aggregate_results,
)
from trackmaniarl.experiments.orchestration import (
    FallbackStrategy,
    GridStrategy,
    Proposal,
    StudySpec,
)


class FailingStrategy:
    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal:
        del study, history
        raise RuntimeError("provider unavailable")


def _finished_result() -> EvaluationResult:
    return EvaluationResult(
        True,
        10.0,
        False,
        5.0,
        1.0,
        60.0,
        steps=2,
        controller_apply_ms=2.0,
        telemetry_wait_ms=8.0,
        telemetry_skipped_frames_total=3,
        telemetry_skipped_frames_max=2,
        telemetry_steps_with_skipped_frames_fraction=0.5,
    )


def _failed_result() -> EvaluationResult:
    return EvaluationResult(
        False,
        None,
        True,
        1.0,
        3.0,
        30.0,
        steps=1,
        controller_apply_ms=5.0,
        telemetry_wait_ms=11.0,
        telemetry_skipped_frames_total=0,
    )


def _evaluation_metrics() -> dict[str, float]:
    return aggregate_results([_finished_result(), _failed_result()])


def test_evaluation_suite_always_reports_the_standard_metric_set() -> None:
    assert tuple(_evaluation_metrics()) == STANDARD_METRICS


def test_evaluation_suite_aggregates_timing_and_telemetry() -> None:
    metrics = _evaluation_metrics()
    assert metrics["eval/finish_rate"] == 0.5
    assert metrics["eval/finish_time_s"] == 10.0
    assert metrics["eval/action_latency_ms"] == 5 / 3
    assert metrics["eval/controller_apply_ms"] == 3.0
    assert metrics["eval/telemetry_wait_ms"] == 9.0
    assert metrics["eval/telemetry_skipped_frames_total"] == 3.0
    assert metrics["eval/telemetry_skipped_frames_mean"] == 1.0
    assert metrics["eval/telemetry_skipped_frames_max"] == 2.0
    assert metrics["eval/telemetry_steps_with_skipped_frames_fraction"] == 1 / 3


def test_provider_failure_falls_back_to_a_valid_grid_proposal() -> None:
    study = StudySpec(
        name="fallback",
        max_trials=2,
        evaluation_suite="smoke",
        search_space={"algorithm.lr": [0.001, 0.01]},
    )
    proposal = FallbackStrategy(FailingStrategy(), GridStrategy()).propose(study, [])
    assert isinstance(proposal, Proposal)
    assert proposal.source == "grid"
    assert proposal.patch == {"algorithm.lr": 0.001}


def test_grid_strategy_selects_late_combinations_without_materializing_the_grid() -> None:
    study = StudySpec(
        name="large-grid",
        max_trials=1,
        evaluation_suite="smoke",
        search_space={f"parameter_{index:02d}": [0, 1] for index in range(24)},
    )
    proposal = GridStrategy().propose(study, [{}] * 5)
    assert proposal.patch == {
        **{f"parameter_{index:02d}": 0 for index in range(21)},
        "parameter_21": 1,
        "parameter_22": 0,
        "parameter_23": 1,
    }
