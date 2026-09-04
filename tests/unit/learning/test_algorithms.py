"""CPU contract checks for the first-class RunSpec 2.0 learners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.unit.learning._algorithm_fixtures import (
    BatchKind,
    CheckpointScaler,
    ConstantTqcModel,
    ContinuousModel,
    ContinuousPpoModel,
    DiscreteSacModel,
    RedqModel,
    StructuredPpoModel,
    _assert_policy_batches_single_chw_image,
    _assert_update,
    _batch,
    _mapping_continuous_case,
    _sequence_batch,
    _with_column_log_probabilities,
    _with_column_scalar_outputs,
)
from trackmaniarl.algorithms import (
    ProximalPolicyOptimization,
    RandomizedEnsembleSAC,
    SoftActorCritic,
    StableDiscreteSoftActorCritic,
    TruncatedQuantileCritic,
)
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.algorithms.ppo_support import AdvantageInputs, generalized_advantage_estimate
from trackmaniarl.algorithms.sac_support import quantile_batch_output, scalar_batch_output
from trackmaniarl.algorithms.truncated_quantile_critic import _truncate_quantile_mixture
from trackmaniarl.core.data import TrainingBatch


def _assert_dense_continuous_updates() -> None:
    batch = _batch(BatchKind.CONTINUOUS)
    _assert_update(SoftActorCritic(ContinuousModel()), _batch(BatchKind.CONTINUOUS))
    _assert_update(
        RandomizedEnsembleSAC(RedqModel(), policy_update_interval=1),
        batch,
    )
    _assert_update(TruncatedQuantileCritic(ContinuousModel(quantiles=5)), batch)
    _assert_update(
        SoftActorCritic(_with_column_scalar_outputs(ContinuousModel())),
        batch,
    )
    _assert_update(
        RandomizedEnsembleSAC(_with_column_scalar_outputs(RedqModel()), policy_update_interval=1),
        batch,
    )


def _assert_structured_continuous_updates() -> None:
    _assert_update(
        TruncatedQuantileCritic(_with_column_log_probabilities(ContinuousModel(quantiles=5))),
        _batch(BatchKind.CONTINUOUS),
    )
    model, batch = _mapping_continuous_case()
    _assert_update(SoftActorCritic(model), batch)
    model, batch = _mapping_continuous_case(quantiles=5)
    _assert_update(TruncatedQuantileCritic(model), batch)


def test_continuous_learners_update_with_supported_observation_structures() -> None:
    _assert_dense_continuous_updates()
    _assert_structured_continuous_updates()
    _assert_policy_batches_single_chw_image()


def test_continuous_learners_reject_ambiguous_model_output_shapes() -> None:
    with pytest.raises(ValueError, match=r"critic output must have shape \(8,\)"):
        scalar_batch_output(torch.zeros(8, 2), "critic output", 8)
    with pytest.raises(ValueError, match=r"log probabilities must have shape \(8,\)"):
        scalar_batch_output(torch.zeros(8, 1, 1), "log probabilities", 8)
    with pytest.raises(ValueError, match=r"critic output must have shape \(8, quantiles\)"):
        quantile_batch_output(torch.zeros(8), "critic output", 8)
    with pytest.raises(ValueError, match=r"critic output must have shape \(8, quantiles\)"):
        quantile_batch_output(torch.zeros(7, 5), "critic output", 8)


@dataclass(frozen=True)
class _TqcCase:
    batch_size: int
    critic_count: int
    quantile_count: int
    drop_count: int
    learn_alpha: bool


_Observations = torch.Tensor | dict[str, torch.Tensor]


@dataclass(frozen=True)
class _BehaviorSample:
    actions: torch.Tensor
    metadata: dict[str, torch.Tensor]


def _behavior_sample(
    learner: ProximalPolicyOptimization, observations: _Observations
) -> _BehaviorSample:
    with torch.no_grad():
        actions, probabilities, latent = learner.model.actor.sample_with_latent(observations)
        values = learner.model.value(observations)
    metadata = {
        "behavior_log_probabilities": probabilities,
        "behavior_values": values,
        "behavior_latent_actions": latent,
    }
    return _BehaviorSample(actions, metadata)


def _sequence_shape(observations: _Observations) -> tuple[int, int]:
    reference = (
        observations
        if isinstance(observations, torch.Tensor)
        else next(iter(observations.values()))
    )
    return int(reference.shape[0]), int(reference.shape[1])


def _ppo_batch(
    learner: ProximalPolicyOptimization,
    observations: _Observations,
    next_observations: _Observations,
) -> TrainingBatch:
    behavior = _behavior_sample(learner, observations)
    shape = _sequence_shape(observations)
    return TrainingBatch(
        data=observations,
        observations=observations,
        actions=behavior.actions,
        rewards=torch.randn(shape),
        next_observations=next_observations,
        terminated=torch.zeros(shape, dtype=torch.bool),
        truncated=torch.zeros(shape, dtype=torch.bool),
        bootstrap_discounts=torch.full(shape, 0.99),
        transition_ids=list(range(shape[0] * shape[1])),
        metadata=behavior.metadata,
    )


def _assert_ppo_metrics(metrics: Mapping[str, float]) -> None:
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert 0.0 <= metrics["state/clip_fraction"] <= 1.0
    assert metrics["state/learning_rate"] == pytest.approx(3e-4)


@pytest.fixture
def ppo_sequence_case() -> tuple[ProximalPolicyOptimization, TrainingBatch]:
    learner = ProximalPolicyOptimization(
        ContinuousPpoModel(),
        update_epochs=2,
        minibatch_size=8,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0, "total_transitions": 32})
    observations = torch.randn(4, 4, 4)
    return learner, _ppo_batch(learner, observations, torch.randn(4, 4, 4))


_TQC_CASES = (
    _TqcCase(1, 2, 3, 0, False),
    _TqcCase(3, 2, 5, 2, True),
    _TqcCase(7, 3, 7, 3, True),
    _TqcCase(11, 4, 4, 1, False),
)


def _assert_tqc_update(case: _TqcCase) -> None:
    model = ContinuousModel(quantiles=case.quantile_count, critic_count=case.critic_count)
    learner = TruncatedQuantileCritic(
        model,
        top_quantiles_to_drop_per_critic=case.drop_count,
        learn_entropy_coefficient=case.learn_alpha,
    )
    _assert_update(learner, _batch(BatchKind.CONTINUOUS, case.batch_size))
    assert (learner.alpha_optimizer is not None) is case.learn_alpha


def test_tqc_updates_across_target_quantile_shapes() -> None:
    for case in _TQC_CASES:
        _assert_tqc_update(case)


_ACTOR_CRITIC_LEARNERS = (
    SoftActorCritic(ContinuousModel()),
    RandomizedEnsembleSAC(RedqModel()),
    TruncatedQuantileCritic(ContinuousModel(quantiles=5)),
    StableDiscreteSoftActorCritic(DiscreteSacModel()),
)


def _assert_resumable_policy_state(learner: Any) -> None:
    learner.setup({"seed": 0})
    policy_state = {
        name: value.detach().clone() + (1.0 if value.is_floating_point() else 0)
        for name, value in learner.policy().export_state().items()
    }
    state = learner.state_dict_for_policy(policy_state)
    for name, value in policy_state.items():
        torch.testing.assert_close(state["model"][f"actor.{name}"], value)
        torch.testing.assert_close(state["target_model"][f"actor.{name}"], value)
    assert state["actor_optimizer"]["state"] == {}


def test_actor_critic_builds_resumable_exact_evaluated_policy_state() -> None:
    for learner in _ACTOR_CRITIC_LEARNERS:
        _assert_resumable_policy_state(learner)


_SCALER_LEARNER_PAIRS = (
    (SoftActorCritic(ContinuousModel()), SoftActorCritic(ContinuousModel())),
    (RandomizedEnsembleSAC(RedqModel()), RandomizedEnsembleSAC(RedqModel())),
    (
        TruncatedQuantileCritic(ContinuousModel(quantiles=5)),
        TruncatedQuantileCritic(ContinuousModel(quantiles=5)),
    ),
    (
        StableDiscreteSoftActorCritic(DiscreteSacModel()),
        StableDiscreteSoftActorCritic(DiscreteSacModel()),
    ),
    (
        ProximalPolicyOptimization(ContinuousPpoModel()),
        ProximalPolicyOptimization(ContinuousPpoModel()),
    ),
)


def _assert_scaler_restore(source: Any, restored: Any) -> None:
    source.execution = TorchExecutionConfig(device="cpu", precision="float32")
    restored.execution = TorchExecutionConfig(device="cpu", precision="float32")
    source.setup({"seed": 0})
    restored.setup({"seed": 1})
    source.scaler = CheckpointScaler(17.0)
    restored.scaler = CheckpointScaler(99.0)

    restored.load_state_dict(source.state_dict())
    assert restored.scaler.current_scale == 17.0


def test_torch_learners_restore_gradient_scaler_state() -> None:
    for source, restored in _SCALER_LEARNER_PAIRS:
        _assert_scaler_restore(source, restored)


def _assert_quantile_drop(critic_count: int, quantile_count: int, drop_count: int) -> None:
    total = critic_count * quantile_count
    quantiles = torch.arange(2 * total, dtype=torch.float32).reshape(2, total)

    truncated = _truncate_quantile_mixture(
        quantiles,
        critic_count=critic_count,
        top_quantiles_to_drop_per_critic=drop_count,
    )

    expected_count = critic_count * (quantile_count - drop_count)
    torch.testing.assert_close(truncated, quantiles[:, :expected_count])


def test_tqc_drops_the_configured_number_of_quantiles_per_critic() -> None:
    for case in ((2, 5, 0), (2, 5, 2), (3, 7, 3)):
        _assert_quantile_drop(*case)


def test_tqc_rejects_a_drop_that_removes_every_target_quantile() -> None:
    quantiles = torch.zeros(2, 6)

    with pytest.raises(ValueError, match="removes every target quantile"):
        _truncate_quantile_mixture(
            quantiles,
            critic_count=2,
            top_quantiles_to_drop_per_critic=3,
        )


def test_tqc_actor_uses_the_mean_of_every_critic_quantile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = TruncatedQuantileCritic(
        ConstantTqcModel(),
        learn_entropy_coefficient=False,
        top_quantiles_to_drop_per_critic=0,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})

    def skip_optimization(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        del loss, optimizer

    monkeypatch.setattr(learner, "_optimize", skip_optimization)

    metrics, _ = learner.update(_batch(BatchKind.CONTINUOUS))

    assert metrics["loss/actor"] == pytest.approx(-2.0)


def test_nonrecurrent_actor_critic_learners_reject_sequence_batches_before_forward() -> None:
    learners_and_batches = (
        (SoftActorCritic(ContinuousModel()), _sequence_batch(BatchKind.CONTINUOUS)),
        (RandomizedEnsembleSAC(RedqModel()), _sequence_batch(BatchKind.CONTINUOUS)),
        (
            TruncatedQuantileCritic(ContinuousModel(quantiles=5)),
            _sequence_batch(BatchKind.CONTINUOUS),
        ),
        (StableDiscreteSoftActorCritic(DiscreteSacModel()), _sequence_batch(BatchKind.DISCRETE)),
    )
    for learner, batch in learners_and_batches:
        learner.execution = TorchExecutionConfig(device="cpu", precision="float32")
        learner.setup({"seed": 0})

        with pytest.raises(ValueError, match=r"requires sequence_length=1"):
            learner.update(batch)


def test_ppo_updates_from_behavior_policy_sequences(
    ppo_sequence_case: tuple[ProximalPolicyOptimization, TrainingBatch],
) -> None:
    learner, batch = ppo_sequence_case

    metrics = learner.update(batch)

    _assert_ppo_metrics(metrics)


def test_ppo_sequence_updates_anneal_and_persist_runtime_state(
    ppo_sequence_case: tuple[ProximalPolicyOptimization, TrainingBatch],
) -> None:
    learner, batch = ppo_sequence_case
    learner.update(batch)

    annealed = learner.update(batch)

    assert annealed["state/learning_rate"] == pytest.approx(1.5e-4)
    assert learner.state_dict()["observation_normalizer"]["moments"]
    assert "scaler" in learner.state_dict()


def test_ppo_gae_stops_recursion_at_episode_end_but_bootstraps_truncation() -> None:
    advantages, returns = generalized_advantage_estimate(
        AdvantageInputs(
            rewards=torch.tensor([[1.0, 2.0]]),
            values=torch.zeros(1, 2),
            next_values=torch.tensor([[0.0, 5.0]]),
            bootstrap_discounts=torch.tensor([[0.9, 0.9]]),
            episode_ends=torch.tensor([[False, True]]),
            gae_lambda=1.0,
        )
    )

    assert torch.allclose(advantages, torch.tensor([[6.85, 6.5]]))
    assert torch.equal(returns, advantages)


def test_ppo_policy_records_behavior_statistics() -> None:
    learner = ProximalPolicyOptimization(
        ContinuousPpoModel(), execution={"device": "cpu", "precision": "float32"}
    )
    learner.setup({"seed": 0})

    action, info = learner.policy().act_with_info(torch.randn(4))

    assert action.shape == (2,)
    assert set(info) == {
        "_trackmaniarl_behavior_log_probability",
        "_trackmaniarl_behavior_value",
        "_trackmaniarl_behavior_latent_action",
    }


def test_ppo_updates_from_structured_observation_sequences() -> None:
    learner = ProximalPolicyOptimization(
        StructuredPpoModel(),
        update_epochs=1,
        minibatch_size=4,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    observations = {
        "track": torch.randn(2, 3, 2),
        "telemetry": torch.randn(2, 3, 2),
    }
    next_observations = {key: torch.randn_like(value) for key, value in observations.items()}
    batch = _ppo_batch(learner, observations, next_observations)

    metrics = learner.update(batch)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
