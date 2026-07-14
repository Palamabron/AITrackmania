"""Tests for deterministic evaluation and autonomous-strategy fallbacks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tmrl.experiments.evaluation import STANDARD_METRICS, EvaluationResult, aggregate_results
from tmrl.experiments.orchestration import FallbackStrategy, GridStrategy, Proposal, StudySpec


class FailingStrategy:
    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal:
        del study, history
        raise RuntimeError("provider unavailable")


def test_evaluation_suite_always_reports_the_standard_metric_set() -> None:
    metrics = aggregate_results(
        [
            EvaluationResult(True, 10.0, False, 5.0, 1.0, 60.0),
            EvaluationResult(False, None, True, 1.0, 3.0, 30.0),
        ]
    )
    assert tuple(metrics) == STANDARD_METRICS
    assert metrics["eval/finish_rate"] == 0.5
    assert metrics["eval/finish_time_s"] == 10.0


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
