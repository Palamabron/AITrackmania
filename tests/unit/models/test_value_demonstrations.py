from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from tests.unit._composite_value_fixtures import _batch, _scalar_model, _sequence_batch
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.objectives import (
    DemonstrationCrossEntropyObjective,
    DemonstrationMarginObjective,
    PolicyAnchorObjective,
    ValueObjective,
    ValueObjectiveContext,
)
from trackmaniarl.core.data import TrainingBatch


def _assert_rejects_masked_action(objective: ValueObjective) -> None:
    context = ValueObjectiveContext(
        expected_values=torch.tensor([[[1.0, 2.0, 3.0]]]),
        actions=torch.tensor([[2]]),
        valid=torch.ones((1, 1), dtype=torch.bool),
        metadata={"demo_flags": (True,)},
        action_mask=torch.tensor([True, True, False]),
    )
    with pytest.raises(ValueError, match="excluded by policy_action_ids"):
        objective.loss(context)


def test_demonstration_objectives_reject_masked_expert_actions() -> None:
    objectives = (DemonstrationMarginObjective(), DemonstrationCrossEntropyObjective())
    for objective in objectives:
        _assert_rejects_masked_action(objective)


def _switch_objective_context() -> ValueObjectiveContext:
    return ValueObjectiveContext(
        expected_values=torch.tensor([[[4.0, 0.0], [0.0, 4.0]]]),
        actions=torch.tensor([[0, 0]]),
        valid=torch.ones((1, 2), dtype=torch.bool),
        metadata={
            "demo_flags": (True, True),
            "demonstration_steering_switches": (False, True),
            "demonstration_steering_switch_distances": (1, 0),
        },
    )


def _assert_switch_weight_increases_loss(
    objectives: tuple[ValueObjective, ValueObjective], context: ValueObjectiveContext
) -> None:
    uniform = objectives[0].loss(context)
    weighted = objectives[1].loss(context)
    assert uniform is not None
    assert weighted is not None
    assert weighted > uniform


def test_demonstration_objectives_can_emphasize_steering_switches() -> None:
    objectives = (
        (
            DemonstrationMarginObjective(),
            DemonstrationMarginObjective(steering_switch_weight=4.0),
        ),
        (
            DemonstrationCrossEntropyObjective(),
            DemonstrationCrossEntropyObjective(steering_switch_weight=4.0),
        ),
    )
    for pair in objectives:
        _assert_switch_weight_increases_loss(pair, _switch_objective_context())


def _switch_window_context() -> ValueObjectiveContext:
    return ValueObjectiveContext(
        expected_values=torch.tensor([[[0.0, 4.0], [4.0, 0.0], [4.0, 0.0]]]),
        actions=torch.tensor([[0, 0, 0]]),
        valid=torch.ones((1, 3), dtype=torch.bool),
        metadata={
            "demo_flags": (True, True, True),
            "demonstration_steering_switches": (False, True, False),
            "demonstration_steering_switch_distances": (1, 0, 1),
        },
    )


def test_demonstration_switch_weight_can_cover_neighboring_steps() -> None:
    context = _switch_window_context()
    exact = DemonstrationCrossEntropyObjective(steering_switch_weight=4.0)
    window = DemonstrationCrossEntropyObjective(
        steering_switch_weight=4.0, steering_switch_radius_steps=1
    )

    exact_loss = exact.loss(context)
    window_loss = window.loss(context)
    assert exact_loss is not None
    assert window_loss is not None
    assert window_loss > exact_loss


def test_demonstration_objectives_reject_negative_steering_switch_weights() -> None:
    constructors = (DemonstrationMarginObjective, DemonstrationCrossEntropyObjective)
    for constructor in constructors:
        with pytest.raises(ValueError, match="weights must be finite and non-negative"):
            constructor(steering_switch_weight=-1.0)


def test_demonstration_objectives_reject_invalid_switch_radius() -> None:
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        DemonstrationCrossEntropyObjective(steering_switch_radius_steps=-1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_value_objectives_reject_non_finite_parameters(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        DemonstrationMarginObjective(margin=value)
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        DemonstrationCrossEntropyObjective(weight=value)
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        PolicyAnchorObjective(weight=value)


def test_demonstration_objectives_require_demo_metadata() -> None:
    context = _switch_objective_context()
    context = ValueObjectiveContext(
        context.expected_values,
        context.actions,
        context.valid,
        {},
    )
    objectives = (DemonstrationMarginObjective(), DemonstrationCrossEntropyObjective())
    for objective in objectives:
        with pytest.raises(ValueError, match="requires demo_flags metadata"):
            objective.loss(context)


def test_steering_switch_weight_requires_switch_metadata() -> None:
    context = _switch_objective_context()
    context = ValueObjectiveContext(
        context.expected_values,
        context.actions,
        context.valid,
        {"demo_flags": (True, True)},
    )
    objectives = (
        DemonstrationMarginObjective(steering_switch_weight=4.0),
        DemonstrationCrossEntropyObjective(steering_switch_weight=4.0),
    )
    for objective in objectives:
        with pytest.raises(ValueError, match="requires demonstration switch metadata"):
            objective.loss(context)


def _demo_diagnostic_learner(switch_weight: float = 1.0) -> DiscreteValueLearner:
    learner = DiscreteValueLearner(
        _scalar_model(),
        objectives=(DemonstrationCrossEntropyObjective(steering_switch_weight=switch_weight),),
        diagnostics_interval_updates=1,
    )
    learner.setup({"seed": 7})
    return learner


def _demo_diagnostic_batch() -> TrainingBatch:
    batch = _batch(batch_size=4)
    batch.metadata.update(
        {
            "demo_flags": (True, True, False, False),
            "demonstration_steering_switches": (False, True, False, False),
            "replay/demo_sample_fraction": 0.5,
            "replay/expert_demo_active_fraction": 0.25,
            "replay/expert_demo_sample_fraction": 0.25,
            "replay/expert_demo_target_fraction": 0.2,
        }
    )
    return batch


def _assert_demo_metric_bounds(metrics: Mapping[str, float]) -> None:
    for key in (
        "debug/demo_accuracy",
        "debug/demo_steering_switch_accuracy",
        "debug/demo_steady_accuracy",
    ):
        assert 0.0 <= metrics[key] <= 1.0


def _assert_demo_replay_metrics(metrics: Mapping[str, float]) -> None:
    expected = {
        "replay/demo_sample_fraction": 0.5,
        "replay/expert_demo_active_fraction": 0.25,
        "replay/expert_demo_sample_fraction": 0.25,
        "replay/expert_demo_target_fraction": 0.2,
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)


def test_value_learner_reports_demonstration_switch_diagnostics() -> None:
    metrics, _ = _demo_diagnostic_learner().update(_demo_diagnostic_batch())
    assert metrics["debug/demo_sample_fraction"] == pytest.approx(0.5)
    assert metrics["debug/demo_steering_switch_fraction"] == pytest.approx(0.5)
    _assert_demo_metric_bounds(metrics)
    _assert_demo_replay_metrics(metrics)


def _add_switch_metadata(batch: TrainingBatch) -> None:
    count = len(batch.transition_ids)
    batch.metadata.update(
        {
            "demo_flags": (True,) * count,
            "demonstration_steering_switches": tuple(index % 2 == 0 for index in range(count)),
        }
    )


def test_sequence_demo_objective_aligns_switch_metadata_to_training_positions() -> None:
    objective = DemonstrationCrossEntropyObjective(steering_switch_weight=4.0)
    learner = DiscreteValueLearner(
        _scalar_model(), objectives=(objective,), diagnostics_interval_updates=1
    )
    learner.setup({"seed": 7})
    batch = _sequence_batch()
    _add_switch_metadata(batch)
    metrics, _ = learner.update(batch)
    assert metrics["loss/objectives"] > 0.0
    assert 0.0 < metrics["debug/demo_steering_switch_fraction"] < 1.0
