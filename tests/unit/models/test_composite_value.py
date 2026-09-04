from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

import pytest
import torch

from tests.unit._composite_value_fixtures import (
    CountingScalarHead,
    _assert_nested_state_equal,
    _batch,
    _recurrent_core,
    _recurrent_value_model,
    _scalar_model,
    _sequence_batch,
    _value_model,
)
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.batches import ValueBatchView
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.models.composite import BatchLayout, FrameBatchAdapter
from trackmaniarl.models.contracts import (
    FractionLossContext,
    ValuePhase,
    ValueSupport,
)
from trackmaniarl.models.heads import ImplicitQuantileHead, ImplicitQuantileHeadConfig
from trackmaniarl.models.strategies import (
    LearnedFractionStrategy,
)

_VALUE_DIAGNOSTICS = {
    "debug/q_selected_mean",
    "debug/q_target_mean",
    "debug/td_abs_mean",
    "debug/n_step_return_mean",
    "debug/bootstrap_zero_fraction",
    "debug/action_batch_entropy",
}


def _assert_sequence_update(
    metrics: Mapping[str, float], priorities: PriorityUpdate, batch: TrainingBatch
) -> None:
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics.keys() >= _VALUE_DIAGNOSTICS
    assert priorities.transition_ids == list(batch.metadata["priority_transition_ids"])
    assert len(priorities.priorities) == batch.rewards.shape[0]


def _parameters_changed(module: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> bool:
    return any(not torch.equal(value, before[name]) for name, value in module.state_dict().items())


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _composed_returns(batch: TrainingBatch, positions: Sequence[int]) -> torch.Tensor:
    return torch.stack(
        [
            batch.rewards[:, position] + 0.9 * batch.rewards[:, position + 1]
            for position in positions[:-1]
        ],
        dim=1,
    )


def test_frame_batch_adapter_preserves_pytree_frame_order() -> None:
    observation = {
        "x": torch.arange(24).reshape(2, 3, 4),
        "nested": (torch.arange(6).reshape(2, 3, 1),),
    }
    batch = FrameBatchAdapter.flatten(observation, BatchLayout.SEQUENCE)
    frames = cast(Mapping[str, object], batch.frames)
    assert torch.equal(cast(torch.Tensor, frames["x"])[:, 0], torch.tensor([0, 4, 8, 12, 16, 20]))
    restored = batch.restore(cast(torch.Tensor, frames["x"]).float())
    assert restored.shape == (2, 3, 4)
    assert torch.equal(restored.long(), observation["x"])


def _assert_recurrent_burn_in_gradient(kind: str) -> None:
    core = _recurrent_core(kind, 3, 5)
    features = torch.randn(2, 4, 3, requires_grad=True)
    outputs = core.unroll(features, burn_in=2)
    weights = torch.arange(1, outputs.numel() + 1, dtype=outputs.dtype).reshape_as(outputs)
    (outputs * weights).sum().backward()
    assert torch.count_nonzero(features.grad[:, :2]) == 0
    assert torch.count_nonzero(features.grad[:, 2:]) > 0


def test_recurrent_burn_in_detaches_context_but_trains_suffix() -> None:
    for kind in ("gru", "mamba-torch"):
        _assert_recurrent_burn_in_gradient(kind)


def test_dueling_selected_quantiles_equal_all_action_gather() -> None:
    head = ImplicitQuantileHead(ImplicitQuantileHeadConfig(5, 4, 8, True))
    features = torch.randn(2, 3, 5)
    strategy = LearnedFractionStrategy(5, fraction_count=6)
    support = strategy.support(features, ValuePhase.TRAIN, None)
    actions = torch.randint(0, 4, (2, 3))
    selected = head.evaluate_actions(features, support, actions)
    gathered = head.evaluate_all(features, support).gather(
        -1, actions.unsqueeze(-1).unsqueeze(-1).expand(2, 3, 6, 1)
    )
    torch.testing.assert_close(selected, gathered.squeeze(-1))


def _assert_fraction_loss_updates_only_fraction_proposal() -> None:
    strategy = LearnedFractionStrategy(4, fraction_count=4, entropy_coefficient=0.0)
    features = torch.randn(2, 4, requires_grad=True)
    support = strategy.support(features, ValuePhase.TRAIN, None)
    boundaries = torch.randn(2, 3, requires_grad=True)
    midpoints = torch.randn(2, 4, requires_grad=True)
    auxiliary = strategy.auxiliary_loss(FractionLossContext(support, boundaries, midpoints))
    assert auxiliary is not None
    auxiliary.loss.backward()
    assert strategy.proposal.weight.grad is not None
    assert features.grad is None
    assert boundaries.grad is None
    assert midpoints.grad is None


def _assert_fraction_boundary_gradient_matches_fqf_objective() -> None:
    strategy = LearnedFractionStrategy(3, fraction_count=4, entropy_coefficient=0.0)
    boundaries = torch.tensor([[0.0, 0.2, 0.5, 0.8, 1.0]], requires_grad=True)
    support = ValueSupport(
        points=torch.tensor([[0.1, 0.35, 0.65, 0.9]]),
        weights=boundaries[:, 1:] - boundaries[:, :-1],
        boundaries=boundaries,
        entropy=torch.zeros(1),
    )
    boundary_values = torch.tensor([[2.0, 4.0, 7.0]])
    midpoint_values = torch.tensor([[1.0, 3.0, 6.0, 10.0]])
    auxiliary = strategy.auxiliary_loss(
        FractionLossContext(support, boundary_values, midpoint_values)
    )
    assert auxiliary is not None
    auxiliary.loss.backward()
    expected = 2 * boundary_values - midpoint_values[:, :-1] - midpoint_values[:, 1:]
    torch.testing.assert_close(boundaries.grad[:, 1:-1], expected)


def test_fraction_proposal_gradient_contracts() -> None:
    _assert_fraction_loss_updates_only_fraction_proposal()
    _assert_fraction_boundary_gradient_matches_fqf_objective()


def test_double_dqn_target_never_evaluates_all_actions() -> None:
    head = CountingScalarHead(6, 3)
    learner = DiscreteValueLearner(_scalar_model(head), target_tau=0.0)
    learner.setup({"seed": 7})
    learner.update(_batch())
    online_head = cast(CountingScalarHead, learner.model.head)
    target_head = cast(CountingScalarHead, learner.target_model.head)
    assert online_head.all_calls == 1
    assert online_head.selected_calls == 1
    assert target_head.all_calls == 0
    assert target_head.selected_calls == 1


def _assert_rejects_invalid_policy_action_ids(
    policy_action_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="policy_action_ids"):
        DiscreteValueLearner(_scalar_model(), policy_action_ids=policy_action_ids)


def _assert_rejects_policy_action_outside_model() -> None:
    learner = DiscreteValueLearner(_scalar_model(), policy_action_ids=(0, 3))

    with pytest.raises(ValueError, match="action_count"):
        learner.setup({"seed": 7})


def test_value_learner_rejects_invalid_policy_action_contracts() -> None:
    for policy_action_ids in ((), (0, 0), (-1, 0)):
        _assert_rejects_invalid_policy_action_ids(policy_action_ids)
    _assert_rejects_policy_action_outside_model()


def _assert_value_strategy_updates_from_sequence_batch(kind: str) -> None:
    learner = DiscreteValueLearner(
        _value_model(kind), target_tau=0.0, diagnostics_interval_updates=1
    )
    learner.setup({"seed": 7})
    batch = _sequence_batch()

    metrics, priorities = learner.update(batch)

    _assert_sequence_update(metrics, priorities, batch)


def test_all_value_strategies_update_from_sequence_sampler_batches() -> None:
    for kind in ("scalar", "qr", "iqn", "fqf"):
        _assert_value_strategy_updates_from_sequence_batch(kind)


def _assert_recurrent_value_learner_updates_after_burn_in(kind: str) -> None:
    model = _recurrent_value_model(kind)
    temporal = model.temporal
    learner = DiscreteValueLearner(
        model,
        burn_in=1,
        target_tau=0.0,
        diagnostics_interval_updates=1,
    )
    learner.setup({"seed": 7})
    temporal_before = _parameter_snapshot(temporal)
    batch = _sequence_batch()

    metrics, priorities = learner.update(batch)

    _assert_sequence_update(metrics, priorities, batch)
    assert metrics["debug/trained_positions"] == 2.0
    assert _parameters_changed(temporal, temporal_before)


def test_recurrent_value_learner_updates_after_burn_in() -> None:
    for kind in ("gru", "mamba-torch"):
        _assert_recurrent_value_learner_updates_after_burn_in(kind)


def _assert_recurrent_policy_reset_restores_first_step_behavior(kind: str) -> None:
    learner = DiscreteValueLearner(_recurrent_value_model(kind), exploration_epsilon=0.0)
    learner.setup({"seed": 7})
    policy = learner.policy()
    first_observation = torch.tensor([0.5, -1.0, 2.0, 0.25])

    first_action = policy.act(first_observation, mode=PolicyMode.EVALUATION)
    first_state = deepcopy(policy._state)
    policy.act(torch.tensor([-2.0, 1.5, 0.0, 3.0]), mode=PolicyMode.EVALUATION)
    policy.reset_episode()

    initial_state = policy.model.initial_policy_state(1, policy.device)
    torch.testing.assert_close(policy._state, initial_state, rtol=0.0, atol=0.0)
    repeated_action = policy.act(first_observation, mode=PolicyMode.EVALUATION)

    assert repeated_action == first_action
    torch.testing.assert_close(policy._state, first_state, rtol=0.0, atol=0.0)


def test_recurrent_policy_reset_restores_first_step_behavior() -> None:
    for kind in ("gru", "mamba-torch"):
        _assert_recurrent_policy_reset_restores_first_step_behavior(kind)


def _assert_recurrent_learner_resume_continues_deterministically(kind: str) -> None:
    batch = _sequence_batch()
    learner = DiscreteValueLearner(_recurrent_value_model(kind), burn_in=1)
    learner.setup({"seed": 7})
    learner.update(batch)
    checkpoint = deepcopy(learner.state_dict())

    learner.update(batch)
    expected = deepcopy(learner.state_dict())

    restored = DiscreteValueLearner(_recurrent_value_model(kind), burn_in=1)
    restored.setup({"seed": 99})
    restored.load_state_dict(checkpoint)
    restored.update(batch)

    _assert_nested_state_equal(restored.state_dict(), expected)


def test_recurrent_learner_resume_continues_deterministically() -> None:
    for kind in ("gru", "mamba-torch"):
        _assert_recurrent_learner_resume_continues_deterministically(kind)


def test_sequence_view_does_not_compose_the_final_n_step_return_twice() -> None:
    batch = _sequence_batch()
    view = ValueBatchView.from_batch(batch)
    positions = view.training_positions(burn_in=0)

    returns, discounts = view.returns_and_discounts(positions)

    torch.testing.assert_close(
        returns[:, :-1],
        _composed_returns(batch, positions),
    )
    torch.testing.assert_close(returns[:, -1], batch.rewards[:, -1])
    torch.testing.assert_close(discounts[:, :-1], torch.full_like(discounts[:, :-1], 0.81))
    torch.testing.assert_close(discounts[:, -1], batch.bootstrap_discounts[:, -1])
