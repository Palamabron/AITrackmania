"""Contract and smoke tests for the isolated TMRL SDK runtime."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tmrl.builtins.algorithms import algorithm_class
from tmrl.builtins.features import TransitionFeaturePipeline
from tmrl.core.data import EpisodeArtifact, Transition
from tmrl.core.runtime import resolve_run, validate_resolved_run
from tmrl.core.spec import RunSpec
from tmrl.core.training import Trainer
from tmrl.observability.artifacts import AsyncEpisodeWriter


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
                "learner": {"class_path": "tmrl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
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
        "class_path": "tests.test_core_runtime:FakeEnvironmentFactory"
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


def test_trainer_evaluation_artifact_is_bound_to_the_current_checkpoint(tmp_path: Path) -> None:
    payload = _spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.test_core_runtime:FakeEnvironmentFactory"
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
                "learner": {"class_path": "tmrl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
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
                "learner": {"class_path": "tmrl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
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
