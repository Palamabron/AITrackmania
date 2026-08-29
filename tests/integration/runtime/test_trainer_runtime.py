"""Trainer lifecycle tests for the isolated runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.integration.runtime.core_runtime_support import FakeEnvironment, runtime_spec
from trackmaniarl.core.builtins import SmokeLearner
from trackmaniarl.core.runtime import ResolvedRun, resolve_run
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.core.training import Trainer
from trackmaniarl.core.training_support import TrainingResult

_FAKE_ENVIRONMENT = "tests.integration.runtime.core_runtime_support:FakeEnvironmentFactory"
_FAILING_ENVIRONMENT = "tests.integration.runtime.core_runtime_support:FailingEnvironmentFactory"
_PPO_COMPONENTS = {
    "learner": {
        "class_path": (
            "trackmaniarl.algorithms.proximal_policy_optimization:ProximalPolicyOptimization"
        ),
        "kwargs": {"update_epochs": 1, "minibatch_size": 2},
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
    "max_episode_steps": 2,
    "batch_size": 2,
    "sequence_length": 2,
    "checkpoint_interval_updates": None,
}
_CHECKPOINT_TRAINING = {
    "total_transitions": 8,
    "max_episode_steps": 2,
    "batch_size": 4,
    "warmup_transitions": 4,
    "updates_per_transition": 1.0,
    "checkpoint_interval_updates": 2,
}


def _ppo_spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "ppo-train",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": _PPO_COMPONENTS,
            "training": _PPO_TRAINING,
        }
    )


def _trainer_spec(
    tmp_path: Path, training: Mapping[str, object], environment_class: str
) -> RunSpec:
    payload = runtime_spec(tmp_path).model_dump(mode="json")
    payload["components"]["environment"] = {"class_path": environment_class}
    payload["training"] = dict(training)
    return RunSpec.model_validate(payload)


def _train_and_close(run: ResolvedRun, resume_checkpoint: Path | None = None) -> TrainingResult:
    try:
        return Trainer(run, resume_checkpoint=resume_checkpoint).train()
    finally:
        run.logger.close()


def _event_payloads(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _event_names(run_dir: Path) -> list[str]:
    return [str(event["event"]) for event in _event_payloads(run_dir)]


class _InterruptingFactory:
    def __init__(self) -> None:
        self.created = 0

    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        self.created += 1
        if self.created > 2:
            raise KeyboardInterrupt
        return FakeEnvironment()


class _ManifestAwareSmokeLearner(SmokeLearner):
    def __init__(self) -> None:
        super().__init__()
        self.manifest_present_at_setup = False

    def setup(self, context: Mapping[str, Any]) -> None:
        self.manifest_present_at_setup = (Path(context["run_dir"]) / "manifest.json").is_file()
        super().setup(context)


def _assert_interrupted_state(state: Mapping[str, Any]) -> None:
    assert state["counters"]["transitions"] == 4
    assert state["counters"]["episodes"] == 2


def test_local_trainer_updates_ppo_once_per_fresh_episode(tmp_path: Path) -> None:
    result = _train_and_close(resolve_run(_ppo_spec(tmp_path)))
    assert result.episodes == 2
    assert result.updates == 2


def test_trainer_collects_updates_and_checkpoints(tmp_path: Path) -> None:
    spec = _trainer_spec(tmp_path, _CHECKPOINT_TRAINING, _FAKE_ENVIRONMENT)
    learner = _ManifestAwareSmokeLearner()
    run = replace(resolve_run(spec), learner=learner)
    result = _train_and_close(run)
    assert learner.manifest_present_at_setup
    assert (result.transitions, result.updates) == (8, 4)
    assert len(result.checkpoints) == len(set(result.checkpoints))
    names = _event_names(run.run_dir)
    assert names.count("train/checkpoint_completed") >= len(result.checkpoints)
    assert "train/checkpoint_failed" not in names
    resumed_learner = _ManifestAwareSmokeLearner()
    resumed_run = replace(resolve_run(spec), learner=resumed_learner)
    resumed = _train_and_close(resumed_run, result.checkpoints[-1])
    assert resumed_learner.manifest_present_at_setup
    assert (resumed.transitions, resumed.updates) == (result.transitions, result.updates)


def test_local_checkpoint_retention_runs_only_after_successful_saves(tmp_path: Path) -> None:
    training = {
        **_CHECKPOINT_TRAINING,
        "checkpoint_interval_updates": 1,
        "checkpoint_keep_last": 1,
    }
    run = resolve_run(_trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT))

    result = _train_and_close(run)

    files = sorted((run.run_dir / "checkpoints").glob("update-*.pt"))
    assert files == [run.run_dir / "checkpoints" / "update-00000004.pt"]
    assert result.checkpoints == tuple(files)
    retention = _retention_events(run.run_dir)
    assert retention
    assert retention[-1]["payload"]["removed_count"] == 1


def _retention_events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        event
        for event in _event_payloads(run_dir)
        if event["event"] == "train/checkpoint_retention"
    ]


def test_trainer_checkpoints_latest_completed_episode_on_interrupt(tmp_path: Path) -> None:
    training = {
        "total_transitions": 10,
        "max_episode_steps": 2,
        "batch_size": 4,
        "warmup_transitions": 4,
        "updates_per_transition": 0.25,
        "checkpoint_interval_updates": 100,
    }
    resolved = resolve_run(_trainer_spec(tmp_path, training, _FAKE_ENVIRONMENT))
    run = replace(resolved, environment_factory=_InterruptingFactory())
    checkpoint = run.run_dir / "checkpoints" / "update-00000000.pt"
    try:
        with pytest.raises(KeyboardInterrupt):
            Trainer(run).train()
        state = run.checkpoint_codec.load(checkpoint)
    finally:
        run.logger.close()
    _assert_interrupted_state(state)


def test_trainer_emits_run_failure_event(tmp_path: Path) -> None:
    training = {
        "total_transitions": 2,
        "max_episode_steps": 2,
        "batch_size": 1,
        "warmup_transitions": 1,
    }
    run = resolve_run(_trainer_spec(tmp_path, training, _FAILING_ENVIRONMENT))
    try:
        with pytest.raises(RuntimeError, match="simulated environment failure"):
            Trainer(run).train()
    finally:
        run.logger.close()
    events = _event_payloads(run.run_dir)
    failure = next(item for item in events if item["event"] == "run/failure")
    assert sum(item["event"] == "run/failure" for item in events) == 1
    assert failure["payload"]["exception_type"] == "RuntimeError"
    assert failure["payload"]["message"] == "simulated environment failure"
