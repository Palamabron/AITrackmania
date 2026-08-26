from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from trackmaniarl.algorithms import AdaptiveGradientClipper
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.batches import ValueBatchView
from trackmaniarl.algorithms.value_based.objectives import (
    DemonstrationCrossEntropyObjective,
    DemonstrationMarginObjective,
    ValueObjective,
    ValueObjectiveContext,
)
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, SequenceSampler
from trackmaniarl.models.backbones import HypersphericalLinear, SimbaV2Backbone
from trackmaniarl.models.composite import CompositeValueModel, FrameBatchAdapter
from trackmaniarl.models.contracts import (
    FractionLossContext,
    RiskDistortion,
    RiskSpec,
    ValuePhase,
    ValueSupport,
)
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import FixedQuantileHead, ImplicitQuantileHead, ScalarQHead
from trackmaniarl.models.strategies import (
    FixedQuantileStrategy,
    LearnedFractionStrategy,
    RandomQuantileStrategy,
    ScalarValueStrategy,
)
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


class StatefulTestScaler:
    def __init__(self, scale: float = 1.0) -> None:
        self.current_scale = scale

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs * self.current_scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        self.current_scale += 1.0

    def state_dict(self) -> dict[str, float]:
        return {"current_scale": self.current_scale}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.current_scale = float(state["current_scale"])


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


def _value_model(kind: str) -> CompositeValueModel:
    encoder = MlpSensorEncoder(4, 6, 8)
    temporal = IdentityTemporalCore(6)
    if kind == "scalar":
        return CompositeValueModel(
            encoder,
            temporal,
            ScalarQHead(6, 3, dueling=True),
            ScalarValueStrategy(),
        )
    if kind == "qr":
        return CompositeValueModel(
            encoder,
            temporal,
            FixedQuantileHead(6, 3, quantile_count=4, dueling=True),
            FixedQuantileStrategy(quantile_count=4),
        )
    strategy = (
        RandomQuantileStrategy(
            train_quantile_count=4,
            target_quantile_count=5,
            evaluation_quantile_count=6,
        )
        if kind == "iqn"
        else LearnedFractionStrategy(6, fraction_count=4)
    )
    return CompositeValueModel(
        encoder,
        temporal,
        ImplicitQuantileHead(6, 3, cosine_count=8, dueling=True),
        strategy,
    )


def _sequence_batch() -> TrainingBatch:
    store = InMemoryReplayStore()
    for step in range(8):
        store.append(
            Transition(
                observation=torch.tensor([float(step), 1.0, -1.0, 0.5]),
                action=step % 3,
                reward=float(step + 1) / 10.0,
                next_observation=torch.tensor([float(step + 1), 1.0, -1.0, 0.5]),
                terminated=step == 7,
                truncated=False,
                episode_id="episode-0",
                step=step,
            )
        )
    return SequenceSampler(IdentityFeaturePipeline(), sequence_length=4, seed=2).sample(
        store,
        BatchRequest(batch_size=2, sequence_length=4, n_step=2, gamma=0.9),
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


@pytest.mark.parametrize("policy_action_ids", [(), (0, 0), (-1, 0)])
def test_value_learner_rejects_invalid_policy_action_ids(
    policy_action_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="policy_action_ids"):
        DiscreteValueLearner(_scalar_model(), policy_action_ids=policy_action_ids)


def test_value_learner_rejects_policy_action_outside_model() -> None:
    learner = DiscreteValueLearner(_scalar_model(), policy_action_ids=(0, 3))

    with pytest.raises(ValueError, match="action_count"):
        learner.setup({"seed": 7})


@pytest.mark.parametrize(
    "objective",
    [DemonstrationMarginObjective(), DemonstrationCrossEntropyObjective()],
)
def test_demonstration_objectives_reject_masked_expert_actions(objective: ValueObjective) -> None:
    context = ValueObjectiveContext(
        expected_values=torch.tensor([[[1.0, 2.0, 3.0]]]),
        actions=torch.tensor([[2]]),
        valid=torch.ones((1, 1), dtype=torch.bool),
        metadata={"demo_flags": (True,)},
        action_mask=torch.tensor([True, True, False]),
    )

    with pytest.raises(ValueError, match="excluded by policy_action_ids"):
        objective.loss(context)


@pytest.mark.parametrize("kind", ["scalar", "qr", "iqn", "fqf"])
def test_all_value_strategies_update_from_sequence_sampler_batches(kind: str) -> None:
    learner = DiscreteValueLearner(
        _value_model(kind), target_tau=0.0, diagnostics_interval_updates=1
    )
    learner.setup({"seed": 7})
    batch = _sequence_batch()

    metrics, priorities = learner.update(batch)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert {
        "debug/q_selected_mean",
        "debug/q_target_mean",
        "debug/td_abs_mean",
        "debug/n_step_return_mean",
        "debug/bootstrap_zero_fraction",
        "debug/action_batch_entropy",
    } <= metrics.keys()
    assert priorities.transition_ids == list(batch.metadata["priority_transition_ids"])
    assert len(priorities.priorities) == batch.rewards.shape[0]


def test_sequence_view_does_not_compose_the_final_n_step_return_twice() -> None:
    batch = _sequence_batch()
    view = ValueBatchView.from_batch(batch)
    positions = view.training_positions(burn_in=0)

    returns, discounts = view.returns_and_discounts(positions)

    torch.testing.assert_close(
        returns[:, :-1],
        torch.stack(
            [
                batch.rewards[:, position] + 0.9 * batch.rewards[:, position + 1]
                for position in positions[:-1]
            ],
            dim=1,
        ),
    )
    torch.testing.assert_close(returns[:, -1], batch.rewards[:, -1])
    torch.testing.assert_close(discounts[:, :-1], torch.full_like(discounts[:, :-1], 0.81))
    torch.testing.assert_close(discounts[:, -1], batch.bootstrap_discounts[:, -1])


@pytest.mark.parametrize("kind", ["scalar", "qr", "iqn", "fqf"])
def test_selected_action_priorities_use_the_quantile_axis_for_every_strategy(kind: str) -> None:
    learner = DiscreteValueLearner(_value_model(kind))
    learner.setup({"seed": 7})
    features = torch.zeros(2, 3, 6, device=learner.device)
    current_support = learner.model.support(features, ValuePhase.TRAIN)
    target_support = learner.target_model.support(features, ValuePhase.TARGET)
    predictions = torch.arange(
        current_support.points.numel(),
        dtype=torch.float32,
        device=learner.device,
    ).reshape_as(current_support.points)
    targets = 0.5 * torch.arange(
        target_support.points.numel(),
        dtype=torch.float32,
        device=learner.device,
    ).reshape_as(target_support.points)
    valid = torch.tensor(
        [[True, True, False], [True, False, True]],
        device=learner.device,
    )

    priorities = learner._priorities(
        predictions,
        current_support,
        targets,
        target_support,
        valid,
    )

    predicted_means = (predictions * current_support.weights).sum(dim=-1)
    target_means = (targets * target_support.weights).sum(dim=-1)
    errors = (predicted_means - target_means).abs() * valid
    expected = 0.9 * errors.max(dim=1).values + 0.1 * errors.sum(dim=1) / valid.sum(dim=1)
    assert priorities == pytest.approx(expected.tolist())


def test_invalid_priority_ids_abort_before_backward_or_optimizer_step() -> None:
    learner = DiscreteValueLearner(_scalar_model())
    learner.setup({"seed": 7})
    batch = _sequence_batch()
    invalid = replace(
        batch,
        metadata={**batch.metadata, "priority_transition_ids": (batch.transition_ids[-1],)},
    )
    online_before = deepcopy(learner.model.state_dict())
    target_before = deepcopy(learner.target_model.state_dict())

    with pytest.raises(ValueError, match="equal length"):
        learner.update(invalid)

    assert learner.update_count == 0
    assert not learner.optimizer.state
    assert all(parameter.grad is None for parameter in learner.model.parameters())
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, online_before[name])
    for name, value in learner.target_model.state_dict().items():
        torch.testing.assert_close(value, target_before[name])


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


def test_random_quantile_upper_cvar_uses_partial_interval_mass() -> None:
    strategy = RandomQuantileStrategy(
        train_quantile_count=4,
        target_quantile_count=4,
        evaluation_quantile_count=4,
    )
    support = strategy.support(torch.zeros(1, 6), ValuePhase.EVALUATE, None)
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    result = strategy.expectation(
        values,
        support,
        RiskSpec(RiskDistortion.UPPER_CVAR, alpha=0.1),
    )

    torch.testing.assert_close(result, torch.tensor([[4.0]]))


def test_strategy_quantile_counts_change_the_architecture_fingerprint() -> None:
    first = _value_model("iqn")
    second = CompositeValueModel(
        MlpSensorEncoder(4, 6, 8),
        IdentityTemporalCore(6),
        ImplicitQuantileHead(6, 3, cosine_count=8, dueling=True),
        RandomQuantileStrategy(
            train_quantile_count=8,
            target_quantile_count=7,
            evaluation_quantile_count=6,
        ),
    )

    assert first.architecture_fingerprint() != second.architecture_fingerprint()


def test_value_learner_projects_simba_weights_after_update() -> None:
    backbone = SimbaV2Backbone(4, 6, block_count=1, expansion=2)
    model = CompositeValueModel(
        backbone,
        IdentityTemporalCore(6),
        ScalarQHead(6, 3, dueling=True),
        ScalarValueStrategy(),
    )
    learner = DiscreteValueLearner(model)
    learner.setup({"seed": 7})

    learner.update(_batch())

    for module in backbone.modules():
        if isinstance(module, HypersphericalLinear):
            torch.testing.assert_close(
                module.weight.norm(dim=1), module.weight.new_ones(module.weight.shape[0])
            )


def test_value_learner_persists_adaptive_gradient_clipper_state() -> None:
    learner = DiscreteValueLearner(
        _scalar_model(),
        adaptive_gradient_clipper=AdaptiveGradientClipper(
            decay=0.5,
            warmup_steps=0,
            clip_factor=1.0,
        ),
    )
    learner.setup({"seed": 7})

    metrics, _ = learner.update(_batch())
    state = deepcopy(learner.state_dict())
    restored = DiscreteValueLearner(
        _scalar_model(),
        adaptive_gradient_clipper=AdaptiveGradientClipper(
            decay=0.5,
            warmup_steps=0,
            clip_factor=1.0,
        ),
    )
    restored.setup({"seed": 7})
    restored.load_state_dict(state)

    assert metrics["gradients/adaptive_ema_norm"] > 0.0
    assert metrics["gradients/adaptive_coefficient"] <= 1.0
    assert restored.adaptive_gradient_clipper is not None
    assert learner.adaptive_gradient_clipper is not None
    assert restored.adaptive_gradient_clipper.state_dict().keys() == (
        learner.adaptive_gradient_clipper.state_dict().keys()
    )
    for name, value in learner.adaptive_gradient_clipper.state_dict().items():
        torch.testing.assert_close(restored.adaptive_gradient_clipper.state_dict()[name], value)


def test_value_learner_resume_restores_scaler_before_deterministic_continuation() -> None:
    batch = _batch()
    learner = DiscreteValueLearner(_scalar_model())
    learner.setup({"seed": 7})
    learner.scaler = StatefulTestScaler()
    learner.update(batch)
    checkpoint = deepcopy(learner.state_dict())

    learner.update(batch)
    expected = deepcopy(learner.model.state_dict())

    restored = DiscreteValueLearner(_scalar_model())
    restored.setup({"seed": 99})
    restored.scaler = StatefulTestScaler(scale=99.0)
    restored.load_state_dict(checkpoint)
    assert isinstance(restored.scaler, StatefulTestScaler)
    assert restored.scaler.current_scale == 2.0

    restored.update(batch)

    assert restored.scaler.current_scale == 3.0
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)


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


def test_policy_loading_accepts_a_legacy_incomplete_training_state() -> None:
    source = DiscreteValueLearner(_value_model("iqn"))
    source.setup({"seed": 7})
    source.update(_batch())
    checkpoint = deepcopy(source.state_dict())
    checkpoint["architecture_fingerprint"] = source.model.legacy_architecture_fingerprint()
    del checkpoint["training"]["scaler"]
    checkpoint["runtime"] = {}

    restored = DiscreteValueLearner(_value_model("iqn"))
    restored.setup({"seed": 11})
    restored.load_policy_state_dict(checkpoint)

    assert restored.update_count == 0
    for name, value in source.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], value)
        torch.testing.assert_close(restored.target_model.state_dict()[name], value)


def test_offline_pretraining_temporarily_freezes_warm_started_submodules(
    tmp_path: Path,
) -> None:
    source = _scalar_model()
    checkpoint = tmp_path / "warm-start.pt"
    torch.save(
        {
            "learner": {
                "online": {
                    "encoder": source.encoder.state_dict(),
                    "temporal": source.temporal.state_dict(),
                    "head": source.head.state_dict(),
                    "strategy": source.strategy.state_dict(),
                }
            }
        },
        checkpoint,
    )
    model = _scalar_model()
    initially_frozen = model.encoder.network[0].bias
    initially_frozen.requires_grad_(False)
    learner = DiscreteValueLearner(
        model,
        model_initialization_checkpoint=checkpoint,
        warm_start_submodules=("encoder",),
        freeze_warm_start_during_offline_pretraining=True,
    )
    learner.setup({"seed": 7})
    optimizer = learner.optimizer
    optimizer_parameter_ids = tuple(
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    )
    encoder_before = deepcopy(model.encoder.state_dict())
    head_before = deepcopy(model.head.state_dict())

    learner.begin_offline_pretraining()
    learner.update(_batch())

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert all(
        torch.equal(value, encoder_before[name])
        for name, value in model.encoder.state_dict().items()
    )
    assert any(
        not torch.equal(value, head_before[name]) for name, value in model.head.state_dict().items()
    )
    assert learner.optimizer is optimizer
    assert (
        tuple(
            id(parameter)
            for group in learner.optimizer.param_groups
            for parameter in group["params"]
        )
        == optimizer_parameter_ids
    )

    learner.end_offline_pretraining()

    assert model.encoder.network[0].weight.requires_grad
    assert not initially_frozen.requires_grad
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_offline_pretraining_freeze_requires_a_warm_start_checkpoint() -> None:
    model = _scalar_model()
    learner = DiscreteValueLearner(
        model,
        freeze_warm_start_during_offline_pretraining=True,
    )
    learner.setup({"seed": 7})

    learner.begin_offline_pretraining()

    assert all(parameter.requires_grad for parameter in model.parameters())


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
