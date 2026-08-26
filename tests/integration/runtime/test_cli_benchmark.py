import importlib.util
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trackmaniarl.cli import (
    _bc_benchmark,
    _demo_benchmark,
    _recovery_contract,
    _restore_smoke_checkpoint,
    _resumed_attempt_spec,
    _smoke_training,
    _train,
    _trajectory_optimize,
    _validate,
    entrypoint,
)
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.spec import RunSpec, TrainingSpec
from trackmaniarl.distributed.protocol import run_fingerprint
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.session import (
    OpenPlanetSessionNotReadyError,
    OpenPlanetSessionProtocolError,
)


def test_recovery_contract_uses_parsed_environment_defaults() -> None:
    config = TrackmaniaEnvironmentConfig(trajectory_path=Path("trajectory.csv"))
    geometry = SimpleNamespace(map_uid="map", sha256="a" * 64)

    contract = _recovery_contract(config, geometry)

    assert contract.action_repeat_frames == 4
    assert contract.decision_interval_ms is None


def test_validate_disables_configured_remote_trackers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(
        """api_version: "2.0"
run_id: validation
components:
  learner: {class_path: trackmaniarl.core.builtins:SmokeLearner}
  replay_store: {class_path: trackmaniarl.core.replay:InMemoryReplayStore}
  sampler: {class_path: trackmaniarl.core.replay:UniformSampler}
  feature_pipeline: {class_path: trackmaniarl.core.builtins:IdentityFeaturePipeline}
  additional_loggers:
    - class_path: trackmaniarl.observability.trackers:WandbTracker
      kwargs: {project: private}
""",
        encoding="utf-8",
    )
    captured: dict[str, RunSpec] = {}

    class FakeLogger:
        def close(self) -> None:
            return None

    fake_run = type("FakeRun", (), {"logger": FakeLogger(), "run_dir": tmp_path})()

    def resolve(spec: RunSpec, *, base_dir: Path) -> object:
        del base_dir
        captured["spec"] = spec
        return fake_run

    monkeypatch.setattr("trackmaniarl.cli.resolve_run", resolve)
    monkeypatch.setattr("trackmaniarl.cli.validate_resolved_run", lambda run: {"loss": 1.0})

    _validate(type("Args", (), {"config": config})())

    assert captured["spec"].components.additional_loggers == ()


def test_track_check_reports_a_disconnected_openplanet_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class DisconnectedClient:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def close(self) -> None:
            return None

        def read(self) -> object:
            raise ConnectionError("telemetry unavailable")

    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetClient", DisconnectedClient)

    with pytest.raises(SystemExit, match="1"):
        entrypoint(["track", "check"])

    error = capsys.readouterr().err
    assert "Trackmania/Openplanet check failed: telemetry unavailable" in error
    assert "Traceback" not in error


def test_track_check_validates_three_exact_frames_and_session_readiness(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {"reads": 0}

    class Client:
        def __init__(
            self,
            host: str,
            port: int,
            *,
            field_count: int,
            timeout_s: float,
        ) -> None:
            captured["telemetry"] = (host, port, field_count, timeout_s)

        def close(self) -> None:
            captured["closed"] = True

        def read(self) -> object:
            captured["reads"] = int(captured["reads"]) + 1
            values = np.zeros(33, dtype=np.float32)
            values[3] = float(captured["reads"])
            return SimpleNamespace(values=values)

    class Session:
        def __init__(self, host: str, port: int, *, timeout_s: float) -> None:
            captured["session"] = (host, port, timeout_s)

        def inspect_loaded_map(self) -> object:
            return SimpleNamespace(map_uid="active-map", protocol_version="2")

        def confirm_ready(self, map_uid: str) -> None:
            captured["ready_uid"] = map_uid

    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetClient", Client)
    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetSessionClient", Session)

    entrypoint(["track", "check"])

    output = capsys.readouterr().out
    assert captured["telemetry"] == ("127.0.0.1", 9000, 33, 5.0)
    assert captured["session"] == ("127.0.0.1", 9001, 5.0)
    assert captured["reads"] == 3
    assert captured["ready_uid"] == "active-map"
    assert captured["closed"] is True
    assert "telemetry_schema=33" in output
    assert "session_protocol=2" in output
    assert "map_uid='active-map'" in output


def test_track_check_config_mismatch_is_a_readable_exit_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Client:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def close(self) -> None:
            return None

        def read(self) -> object:
            return SimpleNamespace(values=np.zeros(33, dtype=np.float32))

    class Session:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def inspect_loaded_map(self) -> object:
            return SimpleNamespace(map_uid="active-map", protocol_version="2")

    factory = SimpleNamespace(config=SimpleNamespace(expected_map_uid="configured-map"))
    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetClient", Client)
    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetSessionClient", Session)
    monkeypatch.setattr("trackmaniarl.cli._trackmania_factory", lambda _: factory)

    with pytest.raises(SystemExit, match="1"):
        entrypoint(["track", "check", "--config", "run.yaml"])

    error = capsys.readouterr().err
    assert "expected 'configured-map', got 'active-map'" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("stage", ["protocol", "readiness"])
def test_track_check_known_session_failures_are_readable_exit_ones(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Client:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def close(self) -> None:
            return None

        def read(self) -> object:
            return SimpleNamespace(values=np.zeros(33, dtype=np.float32))

    class Session:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def inspect_loaded_map(self) -> object:
            if stage == "protocol":
                raise OpenPlanetSessionProtocolError("protocol version mismatch")
            return SimpleNamespace(map_uid="active-map", protocol_version="2")

        def confirm_ready(self, map_uid: str) -> None:
            del map_uid
            raise OpenPlanetSessionNotReadyError("local player is not ready")

    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetClient", Client)
    monkeypatch.setattr("trackmaniarl.cli.OpenPlanetSessionClient", Session)

    with pytest.raises(SystemExit, match="1"):
        entrypoint(["track", "check"])

    error = capsys.readouterr().err
    expected = "protocol version mismatch" if stage == "protocol" else "local player is not ready"
    assert expected in error
    assert "Traceback" not in error


def test_init_prints_commands_from_inside_the_created_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "new agent"

    entrypoint(["init", str(target), "--package", "agent"])

    output = capsys.readouterr().out
    assert f'cd "{target}"' in output
    assert "uv sync" in output
    assert "uv run trackmaniarl validate run.yaml" in output
    assert str(target / "run.yaml") not in output


def test_starter_scaffold_has_no_game_controller_dependencies(tmp_path: Path) -> None:
    target = tmp_path / "agent"
    entrypoint(["init", str(target), "--package", "agent"])
    generated = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    repository = tomllib.loads(
        (Path(__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependencies = generated["project"]["dependencies"]
    trackmaniarl = next(item for item in dependencies if item.startswith("trackmaniarl"))
    distributed = repository["project"]["optional-dependencies"]["distributed"]
    sources = generated["tool"]["uv"]["sources"]

    assert trackmaniarl.partition("[")[2].partition("]")[0] == "distributed"
    assert not any("vgamepad" in item or "libevdev" in item for item in dependencies)
    assert not any("vgamepad" in item or "libevdev" in item for item in distributed)
    assert "vgamepad" not in sources


def test_starter_scaffold_initializes_from_the_declared_seed(tmp_path: Path) -> None:
    target = tmp_path / "agent"
    entrypoint(["init", str(target), "--package", "agent"])
    component_path = target / "src" / "agent" / "components.py"
    module_spec = importlib.util.spec_from_file_location(
        "generated_agent_components", component_path
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    learner_class = module.StarterMlpLearner
    left = learner_class()
    right = learner_class()
    left.setup({"seed": 17})
    right.setup({"seed": 17})
    batch = TrainingBatch(
        data={"observations": [{"speed": 1.0}, {"speed": 2.0}], "rewards": [1.0, -1.0]},
        observations=(),
        actions=(),
        rewards=(),
        next_observations=(),
        terminated=(),
        truncated=(),
        bootstrap_discounts=(),
        transition_ids=(),
    )

    assert left.update(batch) == right.update(batch)
    for left_parameter, right_parameter in zip(
        left.network.parameters(), right.network.parameters(), strict=True
    ):
        assert torch.equal(left_parameter, right_parameter)


def test_smoke_training_reserves_transitions_for_a_learner_update() -> None:
    training = _smoke_training(TrainingSpec(batch_size=256, n_step=3), 100)

    assert training.batch_size == 49
    assert training.warmup_transitions == 51
    assert training.total_transitions > training.warmup_transitions
    assert training.checkpoint_interval_updates == 100


def test_smoke_checkpoint_restore_uses_the_run_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "async-smoke",
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
    config = tmp_path / "run.yaml"
    checkpoint = (
        tmp_path
        / spec.artifacts_dir
        / spec.run_id
        / "checkpoints"
        / ("distributed-update-00000001.pt")
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}

    class Closable:
        def close(self) -> None:
            return None

    class Learner:
        def setup(self, context: object) -> None:
            captured["setup"] = context

    run = SimpleNamespace(
        learner=Learner(),
        run_dir=tmp_path,
        model_factory=None,
        logger=Closable(),
    )

    class FakeCoordinator:
        def __init__(
            self,
            resolved_run: object,
            *,
            bind: str,
            token: str,
            fingerprint: str,
        ) -> None:
            del resolved_run, bind, token
            captured["fingerprint"] = fingerprint
            self.counters = SimpleNamespace(updates=1)
            self._checkpoint_writer = Closable()
            self.journal = Closable()

        def restore_checkpoint(self, path: Path) -> None:
            captured["checkpoint"] = path

    monkeypatch.setattr("trackmaniarl.cli.resolve_run", lambda *_, **__: run)
    monkeypatch.setattr("trackmaniarl.distributed.coordinator.Coordinator", FakeCoordinator)

    _restore_smoke_checkpoint(config, spec)

    assert captured["fingerprint"] == run_fingerprint(spec, config.parent)
    assert captured["checkpoint"] == checkpoint


def test_train_accepts_a_model_initialization_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._train", capture)

    entrypoint(
        [
            "train",
            "run.yaml",
            "--model-initialization-checkpoint",
            "bc-best.pt",
        ]
    )

    assert captured["model_initialization_checkpoint"] == Path("bc-best.pt")


def test_train_routes_on_policy_learner_through_local_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "run.yaml"
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "ppo-local",
            "components": {
                "learner": {"class_path": "trackmaniarl.algorithms:ProximalPolicyOptimization"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:OnPolicySequenceSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
        }
    )
    config.write_text(spec.to_yaml(), encoding="utf-8")
    captured: dict[str, object] = {}
    logger = SimpleNamespace(close=lambda: captured.update(closed=True))
    run = SimpleNamespace(spec=spec, logger=logger, run_dir=tmp_path / "artifacts" / spec.run_id)

    class LocalTrainer:
        def __init__(self, resolved: object, *, resume_checkpoint: Path | None = None) -> None:
            captured["run"] = resolved
            captured["checkpoint"] = resume_checkpoint

        def train(self) -> object:
            captured["trained"] = True
            return SimpleNamespace(transitions=128, updates=1)

    monkeypatch.setattr("trackmaniarl.cli.resolve_run", lambda *_, **__: run)
    monkeypatch.setattr("trackmaniarl.cli.Trainer", LocalTrainer)
    monkeypatch.setattr(
        "trackmaniarl.cli._spawn_context",
        lambda: pytest.fail("on-policy training must not start the distributed coordinator"),
    )
    checkpoint = tmp_path / "resume.pt"

    _train(
        SimpleNamespace(
            config=config,
            checkpoint=checkpoint,
            resume=None,
            reset_replay=False,
            model_initialization_checkpoint=None,
            demo=[],
        )
    )

    assert captured == {
        "run": run,
        "checkpoint": checkpoint,
        "trained": True,
        "closed": True,
    }


def test_resume_recovers_numbered_sibling_for_a_descriptive_run_id(tmp_path: Path) -> None:
    config = tmp_path / "project" / "run.yaml"
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "trackmania-v102d-completion",
            "artifacts_dir": "artifacts",
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
    checkpoint = (
        config.parent
        / "artifacts"
        / "trackmania-v102d-completion-1"
        / "checkpoints"
        / "checkpoint.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parent.parent / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "components": {
                        "learner": {"kwargs": {"model_initialization_checkpoint": "historic.pt"}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resumed = _resumed_attempt_spec(
        config,
        spec,
        SimpleNamespace(checkpoint=checkpoint, resume=None, reset_replay=False),
    )

    assert resumed.run_id == "trackmania-v102d-completion-1"
    assert resumed.components.learner.kwargs["model_initialization_checkpoint"] == "historic.pt"


def test_train_serializes_manifest_restored_warm_start_for_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "run.yaml"
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "trackmania-v102d-completion",
            "artifacts_dir": "artifacts",
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
    config.write_text(spec.to_yaml(), encoding="utf-8")
    run_dir = project / "artifacts" / spec.run_id
    checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "components": {
                        "learner": {"kwargs": {"model_initialization_checkpoint": "historic.pt"}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class Process:
        exitcode = 0
        pid = 1

        def __init__(self, *, args: tuple[object, ...], name: str, **_: object) -> None:
            self.args = args
            self.name = name

        def start(self) -> None:
            spawned = RunSpec.from_yaml(Path(str(self.args[0])))
            captured[self.name] = spawned.components.learner.kwargs.get(
                "model_initialization_checkpoint"
            )

        def is_alive(self) -> bool:
            return False

        def join(self, *, timeout: float) -> None:
            del timeout

        def terminate(self) -> None:
            return None

    context = SimpleNamespace(
        Event=lambda: SimpleNamespace(set=lambda: None),
        Process=lambda **kwargs: Process(**kwargs),
    )
    monkeypatch.setattr("trackmaniarl.cli._spawn_context", lambda: context)
    _train(
        SimpleNamespace(
            config=config,
            checkpoint=checkpoint,
            resume=None,
            reset_replay=False,
            model_initialization_checkpoint=None,
            demo=[],
        )
    )

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

    monkeypatch.setattr("trackmaniarl.cli._offline_pretrain", capture)

    entrypoint(
        [
            "offline-pretrain",
            "run.yaml",
            "--demo",
            "elite",
            "--demo",
            "recovery.npz",
            "--model-initialization-checkpoint",
            "bc-best.pt",
        ]
    )

    assert captured["demo"] == [Path("elite"), Path("recovery.npz")]
    assert captured["model_initialization_checkpoint"] == Path("bc-best.pt")


def test_bc_benchmark_accepts_minimum_action_hold_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._bc_benchmark", capture)

    entrypoint(
        [
            "bc-benchmark",
            "run.yaml",
            "bc-best.pt",
            "--minimum-action-hold-steps",
            "3",
        ]
    )

    assert captured["minimum_action_hold_steps"] == 3


def test_bc_benchmark_accepts_report_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._bc_benchmark", capture)

    entrypoint(["bc-benchmark", "run.yaml", "bc-best.pt", "--report-only"])

    assert captured["report_only"] is True


def test_bc_benchmark_rejects_non_positive_minimum_action_hold() -> None:
    with pytest.raises(
        ValueError,
        match="bc-benchmark --minimum-action-hold-steps must be positive",
    ):
        entrypoint(
            [
                "bc-benchmark",
                "missing.yaml",
                "bc-best.pt",
                "--minimum-action-hold-steps",
                "0",
            ]
        )


def test_bc_benchmark_applies_hold_override_and_logs_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "bc",
            "components": {
                "learner": {"class_path": "package:Learner"},
                "model_factory": {
                    "class_path": "package:Factory",
                    "kwargs": {"minimum_action_hold_steps": 1},
                },
                "replay_store": {"class_path": "package:Replay"},
                "sampler": {"class_path": "package:Sampler"},
                "feature_pipeline": {"class_path": "package:Pipeline"},
            },
            "evaluation": {
                "name": "suite",
                "version": "1",
                "maps": [
                    {
                        "id": "map",
                        "map_path": "map.Gbx",
                        "geometry_path": "geometry.npz",
                        "expected_map_uid": "uid",
                    }
                ],
                "target_median_s": 37.0,
                "min_finish_rate": 1.0,
            },
        }
    )
    metrics = {
        "eval/finish_time_s": 36.0,
        "eval/median_finish_time_s": 36.0,
    }
    trials = [
        {
            "trial_index": 0,
            "map_id": "map",
            "finished": True,
            "finish_time_s": 36.0,
            "progress_pct": 100.0,
            "telemetry_error": None,
            "controller_error": None,
        }
    ]
    captured: dict[str, object] = {}

    class Learner:
        def setup(self, context: object) -> None:
            del context

        def load_state_dict(self, state: object) -> None:
            del state

        def policy(self) -> object:
            return object()

    class Evaluator:
        def set_checkpoint(self, checkpoint: Path) -> None:
            del checkpoint

        def evaluate(self, policy: object) -> dict[str, float]:
            del policy
            (tmp_path / "evaluation.json").write_text(
                json.dumps({"trials": trials}), encoding="utf-8"
            )
            return metrics

    class Logger:
        def log(self, group: str, values: object, *, step: int) -> None:
            captured["log"] = (group, values, step)

        def close(self) -> None:
            return None

    def resolve(spec: RunSpec, *, base_dir: Path) -> object:
        del base_dir
        captured["spec"] = spec
        return SimpleNamespace(
            spec=spec,
            run_dir=tmp_path,
            model_factory=object(),
            learner=Learner(),
            evaluator=Evaluator(),
            logger=Logger(),
            checkpoint_codec=SimpleNamespace(load=lambda _: {"learner": {}}),
        )

    monkeypatch.setattr("trackmaniarl.cli.RunSpec.from_yaml", lambda _: source)
    monkeypatch.setattr("trackmaniarl.cli.resolve_run", resolve)

    _bc_benchmark(
        SimpleNamespace(
            config=tmp_path / "run.yaml",
            checkpoint=tmp_path / "bc-best.pt",
            trials=1,
            minimum_action_hold_steps=3,
        )
    )

    resolved = captured["spec"]
    assert isinstance(resolved, RunSpec)
    assert resolved.components.model_factory is not None
    assert resolved.components.model_factory.kwargs["minimum_action_hold_steps"] == 3
    assert captured["log"] == ("eval/summary", metrics, 0)


@pytest.mark.parametrize(
    ("extra", "expected_phase", "expected_tracking"),
    [
        ([], False, False),
        (["--open-loop"], False, False),
        (["--phase-locked"], True, False),
        (["--trajectory-tracking"], False, True),
    ],
)
def test_demo_benchmark_defaults_to_faithful_open_loop_replay(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    expected_phase: bool,
    expected_tracking: bool,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._demo_benchmark", capture)

    entrypoint(["demo-benchmark", "run.yaml", "demo.npz", *extra])

    assert captured["phase_locked"] is expected_phase
    assert captured["trajectory_tracking"] is expected_tracking


@pytest.mark.parametrize("aggregate_controls", [False, True])
@pytest.mark.parametrize(
    ("cli_action_lead_ms", "expected_action_lead_ms"),
    [(None, 20.0), (35.0, 35.0)],
)
def test_demo_benchmark_preserves_the_configured_open_loop_control_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_controls: bool,
    cli_action_lead_ms: float | None,
    expected_action_lead_ms: float,
) -> None:
    class Evaluation:
        maps = ("map",)
        trials_per_map = 1
        min_finish_rate = 1.0
        target_median_s = 37.0

    class EvaluationStoppedError(RuntimeError):
        pass

    class Evaluator:
        def set_checkpoint(self, _: Path) -> None:
            return None

        def evaluate(self, _: object) -> dict[str, float]:
            raise EvaluationStoppedError

    class Logger:
        def close(self) -> None:
            return None

    raw_environment = {
        "trajectory_path": "track.npy",
        "action_repeat_frames": 1,
        "decision_interval_ms": 50.0,
        "demonstration_action_lead_ms": 20.0,
        "demonstration_control_aggregation": aggregate_controls,
        "compact_action_ids": list(range(78)),
        "control_backend": "gamepad",
    }
    environment_config = TrackmaniaEnvironmentConfig.model_validate(raw_environment)
    spec = SimpleNamespace(
        evaluation=Evaluation(),
        components=SimpleNamespace(environment=SimpleNamespace(kwargs={"config": raw_environment})),
    )
    run = SimpleNamespace(
        evaluator=Evaluator(),
        environment_factory=SimpleNamespace(config=environment_config),
        feature_pipeline=object(),
        logger=Logger(),
        run_dir=tmp_path,
    )
    demonstration = SimpleNamespace(
        decision_interval_ms=None,
        action_repeat_frames=1,
        control_alignment="frame_start",
    )
    captured: dict[str, object] = {}

    def replay_policy(_: Path, action_ids: tuple[int, ...], **kwargs: object) -> object:
        captured["action_ids"] = action_ids
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("trackmaniarl.cli.RunSpec.from_yaml", lambda _: spec)
    monkeypatch.setattr("trackmaniarl.cli.load_demonstration", lambda _: demonstration)
    monkeypatch.setattr("trackmaniarl.cli.validate_recording_quality", lambda _: None)
    monkeypatch.setattr("trackmaniarl.cli.resolve_run", lambda *_, **__: run)
    monkeypatch.setattr("trackmaniarl.cli.TrackmaniaEvaluator", Evaluator)
    monkeypatch.setattr(
        "trackmaniarl.cli.DemonstrationReplayPolicy",
        SimpleNamespace(from_path=replay_policy),
    )
    monkeypatch.setattr(
        "trackmaniarl.cli._with_environment_decision_interval",
        lambda *_: pytest.fail("configured open-loop interval must not be replaced"),
    )
    args = SimpleNamespace(
        config=tmp_path / "run.yaml",
        demo=tmp_path / "demo.npz",
        trajectory_schedule=None,
        trajectory_tracking=False,
        phase_locked=False,
        action_offset_ms=0.0,
        action_lead_ms=cli_action_lead_ms,
        trials=None,
        target_median=None,
        min_finish_rate=None,
    )

    with pytest.raises(EvaluationStoppedError):
        _demo_benchmark(args)

    assert captured["decision_interval_ms"] == 50.0
    assert captured["action_lead_ms"] == expected_action_lead_ms
    assert captured["aggregate_controls"] is aggregate_controls


def test_trajectory_optimize_handler_saves_best_schedule_without_a_live_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controls = np.asarray([[1.0, 0.0, 0.0]] * 8, dtype=np.float32)
    demonstration = SimpleNamespace(controls=controls, decision_interval_ms=10.0)
    spec = SimpleNamespace(training=SimpleNamespace(max_episode_steps=10))

    class Environment:
        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            del seed
            return np.zeros(33, dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            del action
            return (
                np.zeros(33, dtype=np.float32),
                0.0,
                True,
                False,
                {
                    "progress_pct": 100.0,
                    "termination_reason": "finished",
                    "race_time_ms": 35_900.0,
                },
            )

        def close(self) -> None:
            return None

    class Policy:
        def reset_episode(self) -> None:
            return None

        def act(self, observation: np.ndarray, *, deterministic: bool) -> np.ndarray:
            del observation, deterministic
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("trackmaniarl.cli.RunSpec.from_yaml", lambda _: spec)
    monkeypatch.setattr("trackmaniarl.cli.load_demonstration", lambda _: demonstration)
    monkeypatch.setattr("trackmaniarl.cli.validate_recording_quality", lambda _: None)
    monkeypatch.setattr(
        "trackmaniarl.cli._with_environment_decision_interval", lambda value, _: value
    )
    monkeypatch.setattr(
        "trackmaniarl.cli._trajectory_search_environment", lambda *args: Environment()
    )
    monkeypatch.setattr("trackmaniarl.cli.build_scheduled_policy", lambda *args: Policy())
    output = tmp_path / "best-schedule"
    args = SimpleNamespace(
        config=tmp_path / "run.yaml",
        demo=tmp_path / "demo.npz",
        output=output,
        seed=5,
        action_lead_ms=10.0,
        shortening_ms=(40.0, 20.0, 10.0),
        minimum_window_ms=30.0,
        baseline_trials=3,
        confirmation_trials=2,
        minimum_improvement_ms=15.0,
        target_time=36.0,
        max_trials=16,
    )

    _trajectory_optimize(args)

    assert output.with_suffix(".npz").is_file()
    assert "median=35.900s" in capsys.readouterr().out
