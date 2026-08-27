from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _ephemeral_port,
    _Logger,
    _Pipeline,
    _Policy,
    _SlowLearner,
    _transition,
    _TransitionSpec,
    _TransitionState,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.data import BatchRequest
from trackmaniarl.core.replay import (
    InMemoryReplayStore,
    UniformSampler,
    _make_batch,
)
from trackmaniarl.core.replay.batches import _BatchBuild
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor_transport import PolicyReference
from trackmaniarl.distributed.codec import (
    WireCodec,
)
from trackmaniarl.distributed.coordinator import (
    Coordinator,
)
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig


def _interleaved_store() -> tuple[InMemoryReplayStore, list[int], list[int]]:
    store = InMemoryReplayStore()
    actor_a_ids = []
    actor_b_ids = []
    for step in range(3):
        actor_a_ids.append(store.append(_transition(_TransitionSpec("a", step, float(step + 1)))))
        actor_b_ids.append(store.append(_transition(_actor_b_transition(step))))
    return store, actor_a_ids, actor_b_ids


def _actor_b_transition(step: int) -> _TransitionSpec:
    state = _TransitionState.TERMINATES if step == 2 else _TransitionState.CONTINUES
    return _TransitionSpec("b", step, float(10 + step), state)


def test_interleaved_actors_build_episode_local_n_step_returns() -> None:
    store, actor_a_ids, actor_b_ids = _interleaved_store()

    assert store.n_step_ids(actor_a_ids[0], 3) == actor_a_ids
    assert store.n_step_ids(actor_b_ids[0], 3) == actor_b_ids
    batch = _make_batch(
        _BatchBuild(
            store,
            _Pipeline(),
            [actor_a_ids[0], actor_b_ids[0]],
            BatchRequest(batch_size=2, n_step=3, gamma=0.5),
        )
    )

    np.testing.assert_allclose(batch.rewards, [2.75, 18.5])
    np.testing.assert_allclose(batch.bootstrap_discounts, [0.125, 0.0])


def test_policy_snapshot_is_identical_and_replacement_is_atomic() -> None:
    codec = WireCodec(1024 * 1024)
    original = {"weight": torch.randn(4, 3), "bias": torch.randn(4)}
    restored = codec.decode(codec.encode(original))
    assert torch.equal(restored["weight"], original["weight"])
    assert torch.equal(restored["bias"], original["bias"])

    reference = PolicyReference(_Policy(1), epsilon=1.0, version=0)
    before = reference.get()
    reference.replace(_Policy(2), epsilon=0.1, version=7)
    after = reference.get()
    assert (before[0].act(None), before[1:]) == (1, (1.0, 0))
    assert (after[0].act(None), after[1:]) == (2, (0.1, 7))


def _epsilon_components() -> dict[str, dict[str, str]]:
    return {
        "learner": {"class_path": "tests.fake:SlowLearner"},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
    }


def _epsilon_coordinator(tmp_path: Path, config: dict[str, object]) -> Coordinator:
    values = {
        "api_version": "2.0",
        "run_id": config["run_id"],
        "artifacts_dir": str(tmp_path),
        "components": _epsilon_components(),
        "training": config["training"],
        "distributed": config["distributed"],
    }
    return _coordinator_for_spec(RunSpec.model_validate(values), tmp_path)


def _coordinator_for_spec(spec: RunSpec, tmp_path: Path) -> Coordinator:
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / spec.run_id,
        learner=_SlowLearner(),
        environment_factory=None,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
    )
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def _close_coordinator(coordinator: Coordinator) -> None:
    coordinator._checkpoint_writer.close()
    coordinator.journal.close()


def _transition_epsilon_config() -> dict[str, object]:
    return {
        "run_id": "epsilon",
        "training": {"warmup_transitions": 10},
        "distributed": {
            "epsilon_profiles": [1.0, 0.5],
            "epsilon_start": 0.6,
            "epsilon_final": 0.2,
            "epsilon_decay_transitions": 100,
        },
    }


def _update_epsilon_config() -> dict[str, object]:
    return {
        "run_id": "epsilon-updates",
        "training": {},
        "distributed": {
            "epsilon_start": 0.6,
            "epsilon_final": 0.2,
            "epsilon_decay_transitions": 10,
            "epsilon_decay_updates": 100,
        },
    }


def test_distributed_epsilon_uses_transition_schedule_and_profile_multiplier(
    tmp_path: Path,
) -> None:
    coordinator = _epsilon_coordinator(tmp_path, _transition_epsilon_config())
    try:
        assert coordinator._epsilon(0) == pytest.approx(0.6)
        coordinator.counters.transitions = 50
        assert coordinator._epsilon(0) == pytest.approx(0.4)
        assert coordinator._epsilon(1) == pytest.approx(0.2)
    finally:
        _close_coordinator(coordinator)


def test_distributed_epsilon_can_follow_completed_learner_updates(tmp_path: Path) -> None:
    coordinator = _epsilon_coordinator(tmp_path, _update_epsilon_config())
    try:
        coordinator.counters.transitions = 10_000
        coordinator.counters.updates = 50
        assert coordinator._epsilon(0) == pytest.approx(0.4)
    finally:
        _close_coordinator(coordinator)
