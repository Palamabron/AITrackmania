"""Checkpoint, resume, and artifact tests for the isolated runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.integration.runtime.core_runtime_support import RecordingEvaluator, runtime_spec
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.checkpoints import CheckpointRetentionResult, prune_checkpoint_family
from trackmaniarl.core.data import EpisodeArtifact, Transition
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.core.training_support import TrainingCounters, TrainingResult
from trackmaniarl.observability.artifacts import AsyncEpisodeWriter


@dataclass(frozen=True, slots=True)
class _ResumeResult:
    state: Mapping[str, Any]
    counters: Mapping[str, Any]
    replay_size: int


@dataclass(frozen=True, slots=True)
class _EvaluationRun:
    run: ResolvedRun
    evaluator: RecordingEvaluator


def _fake_environment_spec(tmp_path: Path, training: Mapping[str, object]) -> RunSpec:
    payload = runtime_spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:FakeEnvironmentFactory"
    }
    payload["training"] = dict(training)
    return RunSpec.model_validate(payload)


def _evaluation_run(tmp_path: Path, training: Mapping[str, object]) -> _EvaluationRun:
    payload = runtime_spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:FakeEnvironmentFactory"
    }
    payload["components"]["evaluator"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:RecordingEvaluator"
    }
    payload["training"] = dict(training)
    run = resolve_run(RunSpec.model_validate(payload))
    evaluator = run.evaluator
    assert isinstance(evaluator, RecordingEvaluator)
    return _EvaluationRun(run, evaluator)


def _train_and_close(run: ResolvedRun, resume_checkpoint: Path | None = None) -> TrainingResult:
    try:
        return Trainer(run, resume_checkpoint=resume_checkpoint).train()
    finally:
        run.logger.close()


def _initialized_run(run: ResolvedRun) -> ResolvedRun:
    learner = run.spec.components.learner.model_copy(
        update={"kwargs": {"model_initialization_checkpoint": "historic.pt"}}
    )
    components = run.spec.components.model_copy(update={"learner": learner})
    return replace(run, spec=run.spec.model_copy(update={"components": components}))


def _add_warm_start(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["warm_start"] = {"source": "historic.pt", "matched": ["encoder"]}
    manifest["torch_execution"] = {"resolved": True, "backend": "cpu"}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


_PPO_COMPONENTS = {
    "learner": {
        "class_path": (
            "trackmaniarl.algorithms.proximal_policy_optimization:ProximalPolicyOptimization"
        ),
        "kwargs": {
            "update_epochs": 1,
            "minibatch_size": 2,
            "execution": {"device": "cpu"},
        },
    },
    "environment": {
        "class_path": ("tests.integration.runtime.core_runtime_support:PpoFakeEnvironmentFactory")
    },
    "model_factory": {
        "class_path": "trackmaniarl.trackmania.baseline:TelemetryPpoModelFactory",
        "kwargs": {"input_dim": 33, "hidden_dim": 8},
    },
    "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
    "sampler": {"class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler"},
    "feature_pipeline": {"class_path": "trackmaniarl.trackmania.features:TelemetryFeaturePipeline"},
}
_PPO_TRAINING = {
    "total_transitions": 4,
    "max_episode_steps": 10,
    "batch_size": 2,
    "sequence_length": 2,
    "checkpoint_interval_updates": None,
}


def _ppo_resume_spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "ppo-resume",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": _PPO_COMPONENTS,
            "training": _PPO_TRAINING,
        }
    )


def _partial_transition() -> Transition:
    return Transition(
        observation=torch.zeros(33),
        action=torch.zeros(2),
        reward=1.0,
        next_observation=torch.ones(33),
        terminated=False,
        truncated=False,
        episode_id="episode-00000000",
        step=0,
    )


def _checkpoint_partial_replay(spec: RunSpec) -> Mapping[str, Any]:
    source = resolve_run(spec)
    source.learner.setup({"seed": spec.seed, "model_factory": source.model_factory})
    source.replay_store.append(_partial_transition())
    counters = TrainingCounters(transitions=1, updates=0, episodes=0, fractional_updates=0.0)
    try:
        checkpoint = Trainer(source)._write_checkpoint(counters)
        return source.checkpoint_codec.load(checkpoint)
    finally:
        source.logger.close()


def _restore_partial_replay(spec: RunSpec, state: Mapping[str, Any]) -> _ResumeResult:
    restored = resolve_run(spec)
    restored.learner.setup({"seed": spec.seed, "model_factory": restored.model_factory})
    try:
        counters = Trainer(restored)._restore_checkpoint(state)
        return _ResumeResult(state, counters, len(restored.replay_store))
    finally:
        restored.logger.close()


def test_manifest_allows_resume_without_the_original_warm_start(tmp_path: Path) -> None:
    from trackmaniarl.observability import artifacts

    run = resolve_run(runtime_spec(tmp_path))
    try:
        first = artifacts.write_run_manifest(_initialized_run(run))
        _add_warm_start(first)
        second = artifacts.write_run_manifest(run)
    finally:
        run.logger.close()
    assert first == second


def test_episode_artifacts_are_compressed_and_background_written(tmp_path: Path) -> None:
    writer = AsyncEpisodeWriter(tmp_path, max_pending=1)
    artifact = EpisodeArtifact("one", [{"speed": 1.0}], [0.0], [1.0], ["frames/1.jpg"])
    path = writer.submit(artifact).result(timeout=2)
    writer.close()
    assert path.suffix == ".gz"
    assert path.is_file()


def test_episode_artifact_write_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = AsyncEpisodeWriter(tmp_path)

    def fail_write(artifact: EpisodeArtifact) -> Path:
        del artifact
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write", fail_write)
    writer.submit(EpisodeArtifact("one", [], [], [], []))
    with pytest.raises(OSError, match="disk full"):
        writer.close()


def test_trainer_respects_disabled_final_checkpoint(tmp_path: Path) -> None:
    training = {
        "total_transitions": 4,
        "max_episode_steps": 2,
        "batch_size": 2,
        "warmup_transitions": 2,
        "checkpoint_interval_updates": None,
        "save_final_checkpoint": False,
    }
    run = resolve_run(_fake_environment_spec(tmp_path, training))
    result = _train_and_close(run)
    checkpoint_dir = run.run_dir / "checkpoints"
    assert result.checkpoints == ()
    assert not checkpoint_dir.exists() or not list(checkpoint_dir.iterdir())


def test_on_policy_checkpoint_discards_partial_replay_on_resume(tmp_path: Path) -> None:
    spec = _ppo_resume_spec(tmp_path)
    result = _restore_partial_replay(spec, _checkpoint_partial_replay(spec))
    assert result.state["replay_store"] is None
    assert result.counters["next_episode_index"] == 1
    assert result.replay_size == 0
    trained = _train_and_close(resolve_run(spec))
    resumed = _train_and_close(resolve_run(spec), trained.checkpoints[-1])
    attempts_path = spec.artifacts_dir / spec.run_id / "manifest-attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(
        (spec.artifacts_dir / spec.run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert resumed.transitions == trained.transitions
    assert "torch_execution" not in manifest
    assert all(attempt["torch_execution"]["resolved"] is True for attempt in attempts)


def test_training_checkpoint_requires_current_resume_fields(tmp_path: Path) -> None:
    spec = _ppo_resume_spec(tmp_path)
    complete = _checkpoint_partial_replay(spec)
    for field in ("run_fingerprint", "replay_store", "sampler", "next_episode_index"):
        incomplete = dict(complete)
        if field == "next_episode_index":
            incomplete["counters"] = {"updates": 0}
        else:
            incomplete.pop(field)
        with pytest.raises(ValueError, match=field):
            _restore_partial_replay(spec, incomplete)


def test_training_checkpoint_rejects_changed_training_config(tmp_path: Path) -> None:
    spec = _ppo_resume_spec(tmp_path)
    checkpoint = _checkpoint_partial_replay(spec)
    training = spec.training.model_copy(update={"gamma": 0.5})
    changed = spec.model_copy(update={"training": training})

    with pytest.raises(ValueError, match="run fingerprint mismatch"):
        _restore_partial_replay(changed, checkpoint)


def test_training_checkpoint_rejects_outdated_schema(tmp_path: Path) -> None:
    spec = _ppo_resume_spec(tmp_path)
    checkpoint = dict(_checkpoint_partial_replay(spec))
    checkpoint["schema_version"] = "1.0"

    with pytest.raises(ValueError, match="Unsupported training checkpoint schema"):
        _restore_partial_replay(spec, checkpoint)


def test_training_checkpoint_identity_ignores_run_and_artifact_paths(tmp_path: Path) -> None:
    spec = _ppo_resume_spec(tmp_path)
    checkpoint = _checkpoint_partial_replay(spec)
    relocated = spec.model_copy(
        update={"run_id": "ppo-relocated", "artifacts_dir": tmp_path / "relocated"}
    )

    result = _restore_partial_replay(relocated, checkpoint)

    run_identity = checkpoint["run_fingerprint"]
    assert checkpoint["schema_version"] == "2.0"
    assert isinstance(run_identity, str)
    assert len(run_identity) == 64
    assert result.counters["transitions"] == 1


def test_off_policy_checkpoint_rejects_null_component_state(tmp_path: Path) -> None:
    for field in ("replay_store", "sampler"):
        run = resolve_run(_fake_environment_spec(tmp_path, {}))
        try:
            run.learner.setup({"seed": run.spec.seed})
            trainer = Trainer(run)
            state = trainer._checkpoint_state(TrainingCounters())
            state[field] = None
            with pytest.raises(ValueError, match="replay or sampler"):
                trainer._restore_checkpoint(state)
        finally:
            run.logger.close()


def test_checkpoint_decompression_has_a_configured_limit(tmp_path: Path) -> None:
    path = tmp_path / "limited.pt"
    TorchCheckpointCodec().save({"tensor": torch.zeros(256)}, path)
    with pytest.raises(ValueError, match="decompressed checkpoint exceeds"):
        TorchCheckpointCodec(max_decompressed_bytes=64).load(path)


_RETENTION_CHECKPOINTS = (
    "update-00000001.pt",
    "update-00000002.pt",
    "update-00000003.pt",
    "distributed-update-00000001.pt",
    "best-eval-policy-00000007-at-update-00000001.pt",
    "best-eval-policy-00000008-at-update-00000002.pt",
    "fastest-eval-policy-00000007-at-update-00000001.pt",
    "fastest-eval-policy-00000008-at-update-00000002.pt",
    "notes.pt",
)


def _write_retention_checkpoints(directory: Path) -> None:
    for name in _RETENTION_CHECKPOINTS:
        (directory / name).write_bytes(b"checkpoint")


def _retention_results(directory: Path) -> tuple[CheckpointRetentionResult, ...]:
    return (
        prune_checkpoint_family(directory / _RETENTION_CHECKPOINTS[2], directory, keep=2),
        prune_checkpoint_family(directory / _RETENTION_CHECKPOINTS[5], directory, keep=1),
        prune_checkpoint_family(directory / _RETENTION_CHECKPOINTS[7], directory, keep=1),
    )


def _assert_retention_results(results: tuple[CheckpointRetentionResult, ...]) -> None:
    regular, best, fastest = results
    assert regular.family == "local"
    assert [path.name for path in regular.removed] == ["update-00000001.pt"]
    assert best.family == "best-eval"
    assert [path.name for path in best.removed] == [_RETENTION_CHECKPOINTS[4]]
    assert fastest.family == "fastest-eval"
    assert [path.name for path in fastest.removed] == [_RETENTION_CHECKPOINTS[6]]


def test_checkpoint_retention_keeps_filename_families_separate(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    _write_retention_checkpoints(directory)
    _assert_retention_results(_retention_results(directory))
    assert (directory / "distributed-update-00000001.pt").is_file()
    assert (directory / "notes.pt").is_file()


def test_checkpoint_retention_rejects_a_path_outside_the_exact_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoints"
    outside = tmp_path / "outside" / "checkpoints"
    directory.mkdir()
    outside.mkdir(parents=True)
    saved = outside / "update-00000001.pt"
    saved.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="outside"):
        prune_checkpoint_family(saved, directory, keep=1)


def test_trainer_evaluation_artifact_is_bound_to_the_current_checkpoint(
    tmp_path: Path,
) -> None:
    training = {
        "total_transitions": 8,
        "max_episode_steps": 2,
        "batch_size": 4,
        "warmup_transitions": 4,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": 100,
        "evaluate_every_episodes": 1,
    }
    evaluation = _evaluation_run(tmp_path, training)
    result = _train_and_close(evaluation.run)
    assert evaluation.evaluator.checkpoints[-1] == result.checkpoints[-1]
    assert all(path.is_file() for path in evaluation.evaluator.checkpoints)
