"""Contract and smoke tests for the isolated TrackmaniaRL SDK runtime."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trackmaniarl.builtins.algorithms import algorithm_class
from trackmaniarl.builtins.features import TransitionFeaturePipeline
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import EpisodeArtifact, Transition
from trackmaniarl.core.runtime import (
    _redact_config,
    _validate_model_contract,
    resolve_run,
    validate_resolved_run,
)
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.observability.artifacts import AsyncEpisodeWriter
from trackmaniarl.trackmania.baseline import TelemetryPpoModelFactory


class FakeEnvironment:
    def __init__(self) -> None:
        self.step_index = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, float], dict[str, object]]:
        del seed
        self.step_index = 0
        return {"speed": 0.0}, {}

    def step(self, action: object) -> tuple[dict[str, float], float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        return {"speed": float(self.step_index)}, 1.0, self.step_index == 2, False, {}

    def close(self) -> None:
        self.closed = True


class FakeEnvironmentFactory:
    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        return FakeEnvironment()


class PpoFakeEnvironment:
    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, *, seed: int | None = None) -> tuple[torch.Tensor, dict[str, object]]:
        del seed
        self.step_index = 0
        return torch.zeros(4), {}

    def step(self, action: object) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        return torch.full((4,), float(self.step_index)), 1.0, self.step_index == 2, False, {}

    def close(self) -> None:
        return


class PpoFakeEnvironmentFactory:
    def create(self, *, seed: int) -> PpoFakeEnvironment:
        del seed
        return PpoFakeEnvironment()


class RecordingEvaluator:
    def __init__(self) -> None:
        self.checkpoints: list[Path] = []

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        self.checkpoints.append(Path(checkpoint))

    def evaluate(self, policy: object) -> dict[str, float]:
        del policy
        return {"eval/finish_rate": 1.0}


def _spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "run_id": "smoke",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": {
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
        }
    )


def test_resolved_run_writes_manifest_and_smoke_checkpoint(tmp_path: Path) -> None:
    run = resolve_run(_spec(tmp_path))
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()
    manifest = run.run_dir / "manifest.json"
    assert manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["run_id"] == "smoke"
    assert metrics["train/updates"] == 1.0
    assert (run.run_dir / "checkpoints" / "validation.json").is_file()


def test_remote_tracker_config_redacts_nested_secrets() -> None:
    config = {
        "token": "private",
        "metadata": {"api_key": "private", "label": "safe"},
        "items": [{"password": "private"}],
    }

    assert _redact_config(config) == {
        "token": "<redacted>",
        "metadata": {"api_key": "<redacted>", "label": "safe"},
        "items": [{"password": "<redacted>"}],
    }


def test_model_contract_rejects_incompatible_algorithm() -> None:
    class DiscreteQuantileFactory:
        model_contract = ModelContract.DISCRETE_QUANTILE

        def build(self) -> object:
            return object()

    class ContinuousLearner:
        accepted_model_contracts = frozenset({ModelContract.CONTINUOUS_ACTOR_CRITIC})

    with pytest.raises(ValueError, match=r"cannot train.*discrete_quantile"):
        _validate_model_contract(ContinuousLearner(), DiscreteQuantileFactory())


def test_telemetry_ppo_factory_declares_actor_value_contract() -> None:
    factory = TelemetryPpoModelFactory(input_dim=4, hidden_dim=8)

    model = factory.build()

    assert factory.model_contract is ModelContract.CONTINUOUS_ACTOR_VALUE
    assert hasattr(model, "actor")
    assert hasattr(model, "value")


def test_ppo_stack_validates_with_trackmania_control_bounds(tmp_path: Path) -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "ppo-smoke",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": {
                "learner": {
                    "class_path": (
                        "trackmaniarl.algorithms.proximal_policy_optimization:"
                        "ProximalPolicyOptimization"
                    ),
                    "kwargs": {"update_epochs": 1, "minibatch_size": 4},
                },
                "model_factory": {
                    "class_path": "trackmaniarl.trackmania.baseline:TelemetryPpoModelFactory",
                    "kwargs": {"input_dim": 4, "hidden_dim": 8},
                },
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {
                    "class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler",
                },
                "feature_pipeline": {
                    "class_path": "trackmaniarl.trackmania.features:TelemetryFeaturePipeline",
                    "kwargs": {"field_count": 4},
                },
            },
            "training": {"batch_size": 2, "sequence_length": 2},
        }
    )
    run = resolve_run(spec)
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()

    assert "loss/policy" in metrics


def test_local_trainer_updates_ppo_once_per_fresh_episode(tmp_path: Path) -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "ppo-train",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": {
                "learner": {
                    "class_path": (
                        "trackmaniarl.algorithms.proximal_policy_optimization:"
                        "ProximalPolicyOptimization"
                    ),
                    "kwargs": {"update_epochs": 1, "minibatch_size": 2},
                },
                "environment": {
                    "class_path": (
                        "tests.integration.runtime.test_core_runtime:PpoFakeEnvironmentFactory"
                    )
                },
                "model_factory": {
                    "class_path": "trackmaniarl.trackmania.baseline:TelemetryPpoModelFactory",
                    "kwargs": {"input_dim": 4, "hidden_dim": 8},
                },
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.trackmania.features:TelemetryFeaturePipeline",
                    "kwargs": {"field_count": 4},
                },
            },
            "training": {
                "total_transitions": 4,
                "max_episode_steps": 2,
                "batch_size": 2,
                "sequence_length": 2,
                "checkpoint_interval_updates": None,
            },
        }
    )
    run = resolve_run(spec)
    try:
        result = Trainer(run).train()
    finally:
        run.logger.close()

    assert result.episodes == 2
    assert result.updates == 2


def test_manifest_allows_resume_when_only_environment_fields_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.observability import artifacts

    run = resolve_run(_spec(tmp_path))
    try:
        first = artifacts.write_run_manifest(run)
        monkeypatch.setattr(artifacts.platform, "platform", lambda: "hypothetical-os")
        monkeypatch.setattr(artifacts, "_git_revision", lambda: "deadbeef")
        second = artifacts.write_run_manifest(run)
    finally:
        run.logger.close()

    assert first == second
    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert "environment" not in manifest
    attempts = (run.run_dir / "manifest-attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempts) >= 2
    latest = json.loads(attempts[-1])
    assert latest["environment"]["git_revision"] == "deadbeef"
    assert latest["environment"]["platform"] == "hypothetical-os"


def test_episode_artifacts_are_compressed_and_background_written(tmp_path: Path) -> None:
    writer = AsyncEpisodeWriter(tmp_path, max_pending=1)
    path = writer.submit(
        EpisodeArtifact(
            episode_id="one",
            telemetry=[{"speed": 1.0}],
            actions=[0.0],
            rewards=[1.0],
            observation_refs=["frames/1.jpg"],
        )
    ).result(timeout=2)
    writer.close()
    assert path.suffix == ".gz"
    assert path.is_file()


def test_episode_artifact_write_failure_is_reported(tmp_path: Path, monkeypatch) -> None:
    writer = AsyncEpisodeWriter(tmp_path)

    def fail_write(artifact: EpisodeArtifact) -> Path:
        del artifact
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write", fail_write)
    writer.submit(EpisodeArtifact("one", [], [], [], []))
    with pytest.raises(OSError, match="disk full"):
        writer.close()


def test_trainer_collects_updates_and_checkpoints(tmp_path: Path) -> None:
    payload = _spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.integration.runtime.test_core_runtime:FakeEnvironmentFactory"
    }
    payload["training"] = {
        "total_transitions": 8,
        "max_episode_steps": 2,
        "batch_size": 4,
        "warmup_transitions": 4,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": 2,
    }
    run = resolve_run(RunSpec.model_validate(payload))
    try:
        result = Trainer(run).train()
    finally:
        run.logger.close()
    assert result.transitions == 8
    # The first four transitions are pure warm-up, rather than deferred updates.
    assert result.updates == 4
    assert result.checkpoints
    assert len(result.checkpoints) == len(set(result.checkpoints))
    resumed = resolve_run(RunSpec.model_validate(payload))
    try:
        resumed_result = Trainer(resumed, resume_checkpoint=result.checkpoints[-1]).train()
    finally:
        resumed.logger.close()
    assert resumed_result.transitions == result.transitions
    assert resumed_result.updates == result.updates


def test_trainer_without_periodic_checkpoints_keeps_only_the_final_state(tmp_path: Path) -> None:
    payload = _spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.integration.runtime.test_core_runtime:FakeEnvironmentFactory"
    }
    payload["training"] = {
        "total_transitions": 8,
        "max_episode_steps": 2,
        "batch_size": 4,
        "warmup_transitions": 4,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": None,
    }
    run = resolve_run(RunSpec.model_validate(payload))
    try:
        result = Trainer(run).train()
    finally:
        run.logger.close()

    assert result.updates == 4
    assert len(result.checkpoints) == 1


def test_training_spec_can_disable_the_final_checkpoint() -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "best-only",
            "components": {
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
            "training": {
                "checkpoint_interval_updates": None,
                "save_final_checkpoint": False,
            },
        }
    )

    assert spec.training.checkpoint_interval_updates is None
    assert not spec.training.save_final_checkpoint


def test_trainer_evaluation_artifact_is_bound_to_the_current_checkpoint(tmp_path: Path) -> None:
    payload = _spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.integration.runtime.test_core_runtime:FakeEnvironmentFactory"
    }
    payload["training"] = {
        "total_transitions": 8,
        "max_episode_steps": 2,
        "batch_size": 4,
        "warmup_transitions": 4,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": 100,
        "evaluate_every_episodes": 1,
    }
    unresolved = resolve_run(RunSpec.model_validate(payload))
    evaluator = RecordingEvaluator()
    run = replace(unresolved, evaluator=evaluator)
    try:
        result = Trainer(run).train()
    finally:
        run.logger.close()

    assert evaluator.checkpoints
    assert evaluator.checkpoints[-1] == result.checkpoints[-1]
    assert all(path.is_file() for path in evaluator.checkpoints)


def test_training_spec_controls_the_replay_request() -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "request-options",
            "components": {
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
            "training": {"batch_size": 3, "n_step": 2, "gamma": 0.8, "beta": 0.5},
        }
    )
    request = spec.training.batch_request()
    assert (request.batch_size, request.n_step, request.gamma, request.beta) == (3, 2, 0.8, 0.5)


def test_training_spec_anneals_prioritized_replay_beta() -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "beta-schedule",
            "components": {
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
            "training": {
                "total_transitions": 100,
                "beta": 0.4,
                "per_beta_final": 1.0,
                "per_beta_anneal_transitions": 100,
            },
        }
    )

    assert spec.training.replay_beta(0) == pytest.approx(0.4)
    assert spec.training.replay_beta(50) == pytest.approx(0.7)
    assert spec.training.replay_beta(100) == pytest.approx(1.0)


def test_builtins_catalogue_resolves_without_eager_optional_model_imports() -> None:
    batch = TransitionFeaturePipeline().collate([Transition(1.0, 0.0, 1.0, 2.0, False, False)])
    assert tuple(batch["observations"].shape) == (1,)
    assert algorithm_class("soft_actor_critic").__name__ == "SoftActorCritic"
