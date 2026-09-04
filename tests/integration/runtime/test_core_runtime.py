"""Configuration and contract tests for the isolated runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.runtime.core_runtime_support import runtime_spec
from trackmaniarl.core.runtime import (
    _instantiate,
    _redact_config,
    _RunResolver,
    _TrainingComponents,
    _validate_training_contract,
    resolve_run,
    validate_resolved_run,
)
from trackmaniarl.core.spec import ComponentSpec, RunSpec


@dataclass(frozen=True, slots=True)
class _TrainingContractCase:
    training: dict[str, int]
    components: _TrainingComponents
    message: str


_TRAINING_CONTRACT_CASES = (
    _TrainingContractCase(
        {"sequence_length": 4, "n_step": 4},
        _TrainingComponents(
            SimpleNamespace(burn_in=0),
            SimpleNamespace(supports_sequence_sampling=True),
            SimpleNamespace(history_length=1),
        ),
        "n_step",
    ),
    _TrainingContractCase(
        {"sequence_length": 4, "n_step": 1},
        _TrainingComponents(
            SimpleNamespace(burn_in=4),
            SimpleNamespace(supports_sequence_sampling=True),
            SimpleNamespace(history_length=1),
        ),
        "burn_in",
    ),
    _TrainingContractCase(
        {"sequence_length": 4, "n_step": 1},
        _TrainingComponents(
            SimpleNamespace(burn_in=1),
            SimpleNamespace(supports_sequence_sampling=True),
            SimpleNamespace(history_length=2),
        ),
        "history_length",
    ),
    _TrainingContractCase(
        {"sequence_length": 4, "n_step": 1},
        _TrainingComponents(
            SimpleNamespace(burn_in=1),
            SimpleNamespace(supports_sequence_sampling=False),
            SimpleNamespace(history_length=1),
        ),
        "sequence_length=1",
    ),
)

_PPO_COMPONENTS = {
    "learner": {
        "class_path": (
            "trackmaniarl.algorithms.proximal_policy_optimization:ProximalPolicyOptimization"
        ),
        "kwargs": {"update_epochs": 1, "minibatch_size": 4},
    },
    "model_factory": {
        "class_path": "trackmaniarl.trackmania.baseline:TelemetryPpoModelFactory",
        "kwargs": {"input_dim": 33, "hidden_dim": 8},
    },
    "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
    "sampler": {"class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler"},
    "feature_pipeline": {"class_path": "trackmaniarl.trackmania.features:TelemetryFeaturePipeline"},
}
_TQC_COMPONENTS = {
    "learner": {
        "class_path": ("trackmaniarl.algorithms.truncated_quantile_critic:TruncatedQuantileCritic")
    },
    "model_factory": {
        "class_path": "trackmaniarl.trackmania.baseline:TelemetryTqcModelFactory",
        "kwargs": {"config": {"input_dim": 33, "hidden_dim": 8, "quantiles": 5, "critics": 2}},
    },
    "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
    "sampler": {
        "class_path": "trackmaniarl.core.replay:SequenceSampler",
        "kwargs": {"sequence_length": 2},
    },
    "feature_pipeline": {"class_path": "trackmaniarl.trackmania.features:TelemetryFeaturePipeline"},
}


def _algorithm_spec(tmp_path: Path, run_id: str, components: object) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": run_id,
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": components,
            "training": {"batch_size": 2, "sequence_length": 2},
        }
    )


def test_resolved_run_writes_manifest_and_smoke_checkpoint(tmp_path: Path) -> None:
    run = resolve_run(runtime_spec(tmp_path))
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()
    manifest = run.run_dir / "manifest.json"
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
    component = ComponentSpec(
        class_path="trackmaniarl.core.builtins:IdentityFeaturePipeline",
        kwargs={"api_token": "private"},
    )
    with pytest.raises(TypeError) as error:
        _instantiate(component)
    assert "private" not in str(error.value)
    assert "<redacted>" in str(error.value)


def test_additional_logger_receives_descriptive_run_identity(tmp_path: Path) -> None:
    resolver = object.__new__(_RunResolver)
    resolver.spec = runtime_spec(tmp_path)
    resolver.run_dir = tmp_path / "run"
    component = ComponentSpec(
        class_path="tests.integration.runtime.core_runtime_support:CapturingLogger"
    )

    logger = resolver._instantiate_additional_logger(component)

    assert logger.kwargs["run_id"] == "smoke"
    assert logger.kwargs["run_dir"] == tmp_path / "run"


def test_runtime_rejects_mismatched_trackmania_reward_discount(tmp_path: Path) -> None:
    config = runtime_spec(tmp_path).model_dump(mode="json")
    config["components"]["environment"] = {
        "class_path": "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory",
        "kwargs": {"config": {"geometry_path": "geometry.npz", "reward_gamma": 0.5}},
    }
    config["training"]["gamma"] = 0.99
    with pytest.raises(
        ValueError,
        match=r"training.gamma.*environment.config.reward_gamma.*0.99.*0.5",
    ):
        resolve_run(RunSpec.model_validate(config), base_dir=tmp_path)


def test_runtime_rejects_elite_sampler_without_episode_pace_store(tmp_path: Path) -> None:
    config = runtime_spec(tmp_path).model_dump(mode="json")
    config["components"]["sampler"] = {
        "class_path": "trackmaniarl.core.replay:PrioritizedSampler",
        "kwargs": {"elite_time_s": 37.0},
    }
    supported = resolve_run(RunSpec.model_validate(config), base_dir=tmp_path)
    supported.logger.close()
    config["components"]["replay_store"] = {
        "class_path": "tests.integration.runtime.core_runtime_support:BasicReplayStore"
    }

    with pytest.raises(TypeError, match="EpisodePaceReplayStore"):
        resolve_run(RunSpec.model_validate(config), base_dir=tmp_path)


def test_training_contract_rejects_invalid_sequence_configuration(tmp_path: Path) -> None:
    original = runtime_spec(tmp_path)
    for case in _TRAINING_CONTRACT_CASES:
        training = original.training.model_copy(update=case.training)
        spec = original.model_copy(update={"training": training})
        with pytest.raises(ValueError, match=case.message):
            _validate_training_contract(spec, case.components)


def test_ppo_stack_validates_with_trackmania_control_bounds(tmp_path: Path) -> None:
    run = resolve_run(_algorithm_spec(tmp_path, "ppo-smoke", _PPO_COMPONENTS))
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()
    assert "loss/policy" in metrics


def test_runtime_rejects_sequence_training_for_nonrecurrent_tqc(tmp_path: Path) -> None:
    spec = _algorithm_spec(tmp_path, "tqc-sequence", _TQC_COMPONENTS)
    with pytest.raises(ValueError, match=r"TruncatedQuantileCritic.*sequence_length=1"):
        resolve_run(spec)
