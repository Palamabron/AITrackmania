from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import cast

import pytest
import torch
from torch import nn

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.composite import CompositeValueModel, FrameBatchAdapter
from trackmaniarl.models.contracts import FractionLossContext, ValuePhase, ValueSupport
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import ImplicitQuantileHead, ScalarQHead
from trackmaniarl.models.strategies import LearnedFractionStrategy, ScalarValueStrategy
from trackmaniarl.models.temporal import GruTemporalCore, IdentityTemporalCore, MambaTemporalCore


class CountingScalarHead(ScalarQHead):
    def __init__(self, feature_dim: int, action_count: int) -> None:
        super().__init__(feature_dim, action_count, dueling=True)
        self.all_calls = 0
        self.selected_calls = 0

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        self.all_calls += 1
        return super().evaluate_all(features, support)

    def evaluate_actions(
        self, features: torch.Tensor, support: ValueSupport, actions: torch.Tensor
    ) -> torch.Tensor:
        self.selected_calls += 1
        return super().evaluate_actions(features, support, actions)


class FailingNativeMamba(MambaTemporalCore):
    @staticmethod
    def _native_scan() -> object:
        raise ImportError("native kernel unavailable in test")


def _batch(batch_size: int = 3) -> TrainingBatch:
    observations = torch.randn(batch_size, 4)
    return TrainingBatch(
        data={},
        observations=observations,
        actions=torch.randint(0, 3, (batch_size,)),
        rewards=torch.randn(batch_size),
        next_observations=torch.randn(batch_size, 4),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        bootstrap_discounts=torch.full((batch_size,), 0.99),
        transition_ids=list(range(batch_size)),
    )


def _scalar_model(head: nn.Module | None = None) -> CompositeValueModel:
    return CompositeValueModel(
        MlpSensorEncoder(4, 6, 8),
        IdentityTemporalCore(6),
        head or ScalarQHead(6, 3, dueling=True),
        ScalarValueStrategy(),
    )


def test_frame_batch_adapter_preserves_pytree_frame_order() -> None:
    observation = {
        "x": torch.arange(24).reshape(2, 3, 4),
        "nested": (torch.arange(6).reshape(2, 3, 1),),
    }
    batch = FrameBatchAdapter.flatten(observation, sequence=True)
    frames = cast(Mapping[str, object], batch.frames)
    assert torch.equal(cast(torch.Tensor, frames["x"])[:, 0], torch.tensor([0, 4, 8, 12, 16, 20]))
    restored = batch.restore(cast(torch.Tensor, frames["x"]).float())
    assert restored.shape == (2, 3, 4)
    assert torch.equal(restored.long(), observation["x"])


def test_gru_burn_in_detaches_context_but_trains_suffix() -> None:
    core = GruTemporalCore(3, 5)
    features = torch.randn(2, 4, 3, requires_grad=True)
    core.unroll(features, burn_in=2).sum().backward()
    assert torch.count_nonzero(features.grad[:, :2]) == 0
    assert torch.count_nonzero(features.grad[:, 2:]) > 0


def test_dueling_selected_quantiles_equal_all_action_gather() -> None:
    head = ImplicitQuantileHead(5, 4, cosine_count=8, dueling=True)
    features = torch.randn(2, 3, 5)
    strategy = LearnedFractionStrategy(5, fraction_count=6)
    support = strategy.support(features, ValuePhase.TRAIN, None)
    actions = torch.randint(0, 4, (2, 3))
    selected = head.evaluate_actions(features, support, actions)
    gathered = head.evaluate_all(features, support).gather(
        -1, actions.unsqueeze(-1).unsqueeze(-1).expand(2, 3, 6, 1)
    )
    torch.testing.assert_close(selected, gathered.squeeze(-1))


def test_fraction_loss_updates_only_fraction_proposal() -> None:
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


def test_fraction_boundary_gradient_matches_fqf_analytic_objective() -> None:
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


def test_fqf_uses_dedicated_fraction_optimizer() -> None:
    model = CompositeValueModel(
        MlpSensorEncoder(4, 6, 8),
        IdentityTemporalCore(6),
        ImplicitQuantileHead(6, 3, cosine_count=8, dueling=True),
        LearnedFractionStrategy(6, fraction_count=4),
    )
    learner = DiscreteValueLearner(model, fraction_learning_rate=1e-4)
    learner.setup({"seed": 7})
    proposal_before = model.strategy.proposal.weight.detach().clone()
    encoder_before = model.encoder.network[0].weight.detach().clone()
    learner.update(_batch())
    assert learner.fraction_optimizer is not None
    assert not torch.equal(model.strategy.proposal.weight, proposal_before)
    assert not torch.equal(model.encoder.network[0].weight, encoder_before)


def test_checkpoint_rejects_architecture_change() -> None:
    learner = DiscreteValueLearner(_scalar_model())
    learner.setup({"seed": 7})
    state = deepcopy(learner.state_dict())
    changed = DiscreteValueLearner(
        CompositeValueModel(
            MlpSensorEncoder(4, 7, 8),
            IdentityTemporalCore(7),
            ScalarQHead(7, 3),
            ScalarValueStrategy(),
        )
    )
    changed.setup({"seed": 7})
    with pytest.raises(ValueError, match="fingerprint"):
        changed.load_state_dict(state)


def test_mamba_torch_unroll_matches_streaming_step() -> None:
    core = MambaTemporalCore(4, d_state=3, d_conv=2, expand=1, backend="torch").eval()
    features = torch.randn(2, 5, 4)
    unrolled = core.unroll(features, burn_in=0)
    state = core.initial_state(2, torch.device("cpu"))
    outputs = []
    for step in features.unbind(dim=1):
        output, state = core.step(step, state)
        outputs.append(output)
    torch.testing.assert_close(torch.stack(outputs, dim=1), unrolled)


def test_mamba_auto_records_pure_torch_fallback_without_changing_fingerprint() -> None:
    automatic = FailingNativeMamba(4, d_state=3, expand=1, backend="auto")
    automatic.resolve_backend(torch.device("cpu"))
    pure = FailingNativeMamba(4, d_state=3, expand=1, backend="torch")
    pure.load_state_dict(automatic.state_dict())
    assert automatic.resolved_backend == "torch"
    assert automatic.fallback_reason == "ImportError: native kernel unavailable in test"
    assert automatic.state_dict().keys() == pure.state_dict().keys()
    automatic_model = CompositeValueModel(
        MlpSensorEncoder(4, 4, 6),
        automatic,
        ScalarQHead(4, 2),
        ScalarValueStrategy(),
    )
    pure_model = CompositeValueModel(
        MlpSensorEncoder(4, 4, 6), pure, ScalarQHead(4, 2), ScalarValueStrategy()
    )
    assert automatic_model.architecture_fingerprint() == pure_model.architecture_fingerprint()
