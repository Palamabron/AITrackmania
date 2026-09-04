import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.cli import (
    _restore_smoke_checkpoint,
    _resumed_attempt_spec,
    _smoke_training,
    _train,
    entrypoint,
)
from trackmaniarl.commands.common import _matches_attempt, _next_versioned_run_id
from trackmaniarl.commands.smoke import _smoke_spec
from trackmaniarl.commands.training import _offline_pretrain
from trackmaniarl.core.spec import RunSpec, TrainingSpec
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import run_fingerprint


def _components(learner: str) -> dict[str, dict[str, str]]:
    return {
        "learner": {"class_path": learner},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"},
    }


def _run_spec(run_id: str, learner: str, artifacts_dir: str = "artifacts") -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": run_id,
            "artifacts_dir": artifacts_dir,
            "components": _components(learner),
        }
    )


class _Closable:
    def close(self) -> None:
        return None


class _SetupLearner:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def setup(self, context: object) -> None:
        self.captured["setup"] = context


class _FakeCoordinator:
    def __init__(self, captured: dict[str, object], config: CoordinatorConfig) -> None:
        self.captured = captured
        captured["fingerprint"] = config.fingerprint
        self.counters = SimpleNamespace(updates=1)
        self._checkpoint_writer = _Closable()
        self.journal = _Closable()

    def restore_checkpoint(self, path: Path) -> None:
        self.captured["checkpoint"] = path


def _smoke_run(tmp_path: Path, captured: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        learner=_SetupLearner(captured),
        run_dir=tmp_path,
        model_factory=None,
        logger=_Closable(),
    )


def _checkpoint_path(config: Path, spec: RunSpec) -> Path:
    return (
        config.parent
        / spec.artifacts_dir
        / spec.run_id
        / "checkpoints"
        / "distributed-update-00000001.pt"
    )


def test_smoke_training_reserves_transitions_for_a_learner_update() -> None:
    training = _smoke_training(TrainingSpec(batch_size=256, n_step=3), 100)

    assert training.batch_size == 49
    assert training.warmup_transitions == 51
    assert training.total_transitions > training.warmup_transitions
    assert training.checkpoint_interval_updates == 100


def _smoke_evaluation_suite() -> dict[str, object]:
    return {
        "name": "scheduled-smoke",
        "version": "1",
        "maps": [
            {
                "id": "test-map",
                "map_path": "test.Map.Gbx",
                "geometry_path": "test.geometry.npz",
                "expected_map_uid": "test-map",
            }
        ],
    }


def _scheduled_smoke_spec() -> RunSpec:
    source = _run_spec("scheduled-smoke", "trackmaniarl.core.builtins:SmokeLearner")
    data = source.model_dump(mode="python")
    data["components"]["evaluator"] = {"class_path": "trackmaniarl.core.builtins:SmokeEvaluator"}
    data["evaluation"] = _smoke_evaluation_suite()
    data["training"].update(
        {
            "evaluate_every_episodes": 1,
            "evaluation_stop_min_finish_rate": 1.0,
            "evaluation_stop_median_s": 37.0,
            "evaluation_stop_consecutive_batches": 2,
        }
    )
    return RunSpec.model_validate(data)


def test_smoke_spec_disables_scheduled_evaluation() -> None:
    source = _scheduled_smoke_spec()

    smoke = RunSpec.model_validate(_smoke_spec(source, 100).model_dump(mode="python"))

    assert smoke.components.evaluator is None
    assert smoke.training.evaluate_every_episodes is None
    assert smoke.training.evaluation_stop_min_finish_rate is None
    assert smoke.training.evaluation_stop_median_s is None
    assert smoke.training.evaluation_stop_consecutive_batches is None


def test_smoke_checkpoint_restore_uses_the_run_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _run_spec("async-smoke", "trackmaniarl.core.builtins:SmokeLearner")
    config = tmp_path / "run.yaml"
    checkpoint = _checkpoint_path(config, spec)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}
    run = _smoke_run(tmp_path, captured)
    monkeypatch.setattr("trackmaniarl.commands.smoke.resolve_run", lambda *_, **__: run)
    monkeypatch.setattr(
        "trackmaniarl.distributed.coordinator.Coordinator",
        lambda resolved, coordinator_config: _FakeCoordinator(captured, coordinator_config),
    )
    _restore_smoke_checkpoint(config, spec)
    assert captured["fingerprint"] == run_fingerprint(spec, config.parent)
    assert captured["checkpoint"] == checkpoint


def test_train_accepts_a_model_initialization_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.commands.parser_training._train", capture)

    entrypoint(["train", "run.yaml", "--model-initialization-checkpoint", "bc-best.pt"])

    assert captured["model_initialization_checkpoint"] == Path("bc-best.pt")


class _LocalTrainer:
    def __init__(
        self, captured: dict[str, object], resolved: object, checkpoint: Path | None
    ) -> None:
        self.captured = captured
        captured["run"] = resolved
        captured["checkpoint"] = checkpoint

    def train(self) -> object:
        self.captured["trained"] = True
        return SimpleNamespace(transitions=128, updates=1)


def _training_args(config: Path, checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        checkpoint=checkpoint,
        resume=None,
        reset_replay=False,
        model_initialization_checkpoint=None,
        demo=[],
    )


def _ppo_spec() -> RunSpec:
    components = _components("trackmaniarl.algorithms:ProximalPolicyOptimization")
    components["sampler"] = {"class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler"}
    return RunSpec.model_validate(
        {"api_version": "2.0", "run_id": "ppo-local", "components": components}
    )


def _patch_local_training(
    monkeypatch: pytest.MonkeyPatch, run: object, captured: dict[str, object]
) -> None:
    monkeypatch.setattr("trackmaniarl.commands.training.resolve_run", lambda *_, **__: run)
    monkeypatch.setattr(
        "trackmaniarl.commands.training.Trainer",
        lambda resolved, resume_checkpoint=None: _LocalTrainer(
            captured, resolved, resume_checkpoint
        ),
    )
    monkeypatch.setattr(
        "trackmaniarl.commands.training._spawn_context",
        lambda: pytest.fail("on-policy training must not start the distributed coordinator"),
    )


def test_train_routes_on_policy_learner_through_local_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "run.yaml"
    spec = _ppo_spec()
    config.write_text(spec.to_yaml(), encoding="utf-8")
    captured: dict[str, object] = {}
    logger = SimpleNamespace(close=lambda: captured.update(closed=True))
    run = SimpleNamespace(spec=spec, logger=logger, run_dir=tmp_path / "artifacts" / spec.run_id)
    _patch_local_training(monkeypatch, run, captured)
    checkpoint = tmp_path / "resume.pt"
    _train(_training_args(config, checkpoint))
    assert captured == {
        "run": run,
        "checkpoint": checkpoint,
        "trained": True,
        "closed": True,
    }


def test_resume_recovers_numbered_sibling_for_a_descriptive_run_id(tmp_path: Path) -> None:
    config = tmp_path / "project" / "run.yaml"
    spec = _run_spec("trackmania-v102d-completion", "trackmaniarl.core.builtins:SmokeLearner")
    run_dir = config.parent / "artifacts" / "trackmania-v102d-completion-1"
    checkpoint = _write_checkpoint(run_dir)
    _write_manifest(run_dir)

    resumed = _resumed_attempt_spec(
        config,
        spec,
        SimpleNamespace(checkpoint=checkpoint, resume=None, reset_replay=False),
    )

    assert resumed.run_id == "trackmania-v102d-completion-1"
    assert resumed.components.learner.kwargs["model_initialization_checkpoint"] == "historic.pt"


@pytest.mark.parametrize("run_id", ["experiment", "experiment-v1"])
def test_new_attempt_uses_a_numeric_suffix(run_id: str, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / f"{run_id}-1").mkdir(parents=True)

    assert _next_versioned_run_id(run_id, artifacts) == f"{run_id}-2"


@pytest.mark.parametrize("resumed_id", ["experiment", "experiment-1", "experiment-20"])
def test_resume_adopts_generated_numeric_attempts(resumed_id: str) -> None:
    assert _matches_attempt("experiment", resumed_id)


@pytest.mark.parametrize("resumed_id", ["experiment-01", "experiment-fast", "experiment-copy-1"])
def test_resume_rejects_unrelated_attempt_names(resumed_id: str) -> None:
    assert not _matches_attempt("experiment", resumed_id)


def _write_checkpoint(run_dir: Path) -> Path:
    checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    return checkpoint


def _write_manifest(run_dir: Path) -> None:
    manifest = {
        "config": {
            "components": {
                "learner": {"kwargs": {"model_initialization_checkpoint": "historic.pt"}}
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class _SpawnProcess:
    exitcode = 0
    pid = 1

    def __init__(self, captured: dict[str, object], args: tuple[Any, ...], name: str) -> None:
        self.captured = captured
        self.args = args
        self.name = name

    def start(self) -> None:
        spawned = RunSpec.from_yaml(Path(self.args[0].config_path))
        self.captured[self.name] = spawned.components.learner.kwargs[
            "model_initialization_checkpoint"
        ]

    def is_alive(self) -> bool:
        return False

    def join(self, *, timeout: float) -> None:
        del timeout

    def terminate(self) -> None:
        return None


def _spawn_context(captured: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        Event=lambda: SimpleNamespace(set=lambda: None),
        Process=lambda **values: _SpawnProcess(captured, values["args"], values["name"]),
    )


def _spawn_training_files(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "run.yaml"
    spec = _run_spec("trackmania-v102d-completion", "trackmaniarl.core.builtins:SmokeLearner")
    config.write_text(spec.to_yaml(), encoding="utf-8")
    run_dir = project / "artifacts" / spec.run_id
    checkpoint = _write_checkpoint(run_dir)
    _write_manifest(run_dir)
    return config, checkpoint


def test_train_serializes_manifest_restored_warm_start_for_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, checkpoint = _spawn_training_files(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "trackmaniarl.commands.training._spawn_context", lambda: _spawn_context(captured)
    )
    _train(_training_args(config, checkpoint))
    assert captured == {
        "trackmaniarl-learner": "historic.pt",
        "trackmaniarl-local-actor": "historic.pt",
    }


def test_offline_pretrain_accepts_repeatable_demos_and_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.commands.parser_training._offline_pretrain", capture)

    entrypoint(_offline_pretrain_args())

    assert captured["demo"] == [Path("elite"), Path("recovery.npz")]
    assert captured["model_initialization_checkpoint"] == Path("bc-best.pt")


def test_offline_pretrain_reports_when_final_checkpoint_is_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trackmaniarl.commands.training._offline_training_spec",
        lambda args: (Path("run.yaml"), object(), ()),
    )
    monkeypatch.setattr(
        "trackmaniarl.commands.training._run_offline_pretraining",
        lambda *args: SimpleNamespace(updates=7, checkpoints=()),
    )

    _offline_pretrain(SimpleNamespace())

    assert capsys.readouterr().out == (
        "Offline pretraining complete: updates=7, final checkpoint disabled.\n"
    )


def _offline_pretrain_args() -> list[str]:
    return [
        "offline-pretrain",
        "run.yaml",
        "--demo",
        "elite",
        "--demo",
        "recovery.npz",
        "--model-initialization-checkpoint",
        "bc-best.pt",
    ]
