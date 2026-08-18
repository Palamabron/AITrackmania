from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.cli import (
    _behavior_cloning_checkpoint_improved,
    _behavior_cloning_control_score,
    _BehaviorCloningSelection,
    _dagger_sample_weight,
    _expert_environment_config,
    _new_attempt_spec,
    _next_versioned_run_id,
    _print_benchmark_report,
    _resumed_attempt_spec,
    _signal_shutdown,
    _trajectory_optimize,
    _validate,
    _with_environment_decision_interval,
    _with_model_initialization_checkpoint,
    entrypoint,
)
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig


def test_validate_disables_configured_remote_trackers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(
        """run_id: validation
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

    assert "OpenPlanet telemetry check failed: telemetry unavailable" in capsys.readouterr().err


def test_benchmark_report_prints_trials_and_summary(capsys: pytest.CaptureFixture[str]) -> None:
    trials = [
        {
            "trial_index": 0,
            "map_id": "trackmaniarl-test",
            "finished": True,
            "finish_time_s": 37.25,
            "progress_pct": 100.0,
            "telemetry_error": None,
            "controller_error": None,
        },
        {
            "trial_index": 1,
            "map_id": "trackmaniarl-test",
            "finished": False,
            "finish_time_s": None,
            "progress_pct": 12.5,
            "telemetry_error": None,
            "controller_error": "controller disconnected",
        },
    ]
    metrics = {"eval/finish_time_s": 37.25, "eval/median_finish_time_s": 37.25}

    _print_benchmark_report(trials, metrics)

    output = capsys.readouterr().out
    assert "trial=0 map=trackmaniarl-test finished=True time=37.250s" in output
    assert "trial=1 map=trackmaniarl-test finished=False time=-" in output
    assert "finishes=1/2" in output
    assert "mean_completed=37.250s" in output
    assert "median_completed=37.250s" in output


def test_benchmark_accepts_explicit_release_gate_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._benchmark", capture)

    entrypoint(
        [
            "benchmark",
            "run.yaml",
            "checkpoint.pt",
            "--trials",
            "30",
            "--target-median",
            "38.3",
            "--min-finish-rate",
            "0.9",
        ]
    )

    assert captured["trials"] == 30
    assert captured["target_median"] == 38.3
    assert captured["min_finish_rate"] == 0.9


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


def test_model_initialization_override_updates_only_the_learner() -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "transfer",
            "components": {
                "learner": {
                    "class_path": (
                        "trackmaniarl.algorithms.implicit_quantile_q_learning:"
                        "ImplicitQuantileQLearning"
                    ),
                    "kwargs": {"learning_rate": 1e-5},
                },
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
        }
    )

    updated = _with_model_initialization_checkpoint(spec, Path("C:/models/bc.pt"))

    assert updated.components.learner.kwargs["model_initialization_checkpoint"] == (
        "C:\\models\\bc.pt"
    )
    assert "model_initialization_checkpoint" not in spec.components.learner.kwargs


def test_trajectory_optimize_parses_physical_timing_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._trajectory_optimize", capture)

    entrypoint(
        [
            "trajectory-optimize",
            "run.yaml",
            "demo.npz",
            "best.npz",
            "--max-trials",
            "24",
            "--seed",
            "7",
            "--action-lead-ms",
            "15",
            "--shortening-ms",
            "30",
            "10",
        ]
    )

    assert captured["max_trials"] == 24
    assert captured["seed"] == 7
    assert captured["action_lead_ms"] == 15.0
    assert captured["shortening_ms"] == [30.0, 10.0]


def test_trajectory_stitch_accepts_multiple_demo_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("trackmaniarl.cli._trajectory_stitch", capture)

    entrypoint(
        [
            "trajectory-stitch",
            "run.yaml",
            "stitched.npz",
            "--demo",
            "elite",
            "--demo",
            "native",
        ]
    )

    assert captured["demo"] == [Path("elite"), Path("native")]


def test_dagger_weights_prioritize_disagreement_and_intervention() -> None:
    nominal = _dagger_sample_weight(False, False, 0.0, 0.5)
    disagreement = _dagger_sample_weight(True, False, 0.0, 0.5)
    intervention = _dagger_sample_weight(True, True, 1.0, 0.5)

    assert nominal == 0.25
    assert disagreement == 2.0
    assert intervention == 6.0


def test_behavior_cloning_control_score_prefers_better_closed_loop_proxies() -> None:
    early = _behavior_cloning_control_score(
        {
            "loss": 0.55360,
            "accuracy": 0.8155,
            "balanced_accuracy": 0.8840,
            "transition_accuracy": 0.4432,
            "steering_accuracy": 0.91,
            "steering_transition_accuracy": 0.45,
            "intervention_count": 0.0,
        }
    )
    later = _behavior_cloning_control_score(
        {
            "loss": 0.58202,
            "accuracy": 0.8193,
            "balanced_accuracy": 0.8971,
            "transition_accuracy": 0.5530,
            "steering_accuracy": 0.93,
            "steering_transition_accuracy": 0.60,
            "intervention_count": 0.0,
        }
    )

    assert later > early


def test_behavior_cloning_checkpoint_rejects_excess_validation_loss() -> None:
    selection = _BehaviorCloningSelection(
        minimum_loss=0.5,
        checkpoint_score=0.6,
        checkpoint_loss=0.52,
    )

    assert _behavior_cloning_checkpoint_improved(selection, 0.54, 0.7)
    assert not _behavior_cloning_checkpoint_improved(selection, 0.56, 0.9)


def test_behavior_cloning_control_score_includes_recovery_accuracy() -> None:
    baseline = {
        "loss": 0.5,
        "accuracy": 0.8,
        "balanced_accuracy": 0.8,
        "transition_accuracy": 0.8,
        "steering_accuracy": 0.8,
        "steering_transition_accuracy": 0.8,
        "weighted_accuracy": 0.8,
        "intervention_accuracy": 0.4,
        "intervention_count": 2.0,
    }
    recovered = {**baseline, "intervention_accuracy": 0.9}

    assert _behavior_cloning_control_score(recovered) > _behavior_cloning_control_score(baseline)


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


def test_open_loop_replay_overrides_environment_with_demonstration_cadence() -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "replay",
            "components": {
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "environment": {
                    "class_path": (
                        "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
                    ),
                    "kwargs": {"config": {"action_repeat_frames": 2, "decision_interval_ms": 20.0}},
                },
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
        }
    )

    updated = _with_environment_decision_interval(spec, 10.0)
    config = updated.components.environment.kwargs["config"]

    assert config["action_repeat_frames"] == 1
    assert config["decision_interval_ms"] == 10.0


def test_expert_diagnostics_rejects_compact_action_heads(tmp_path: Path) -> None:
    config = TrackmaniaEnvironmentConfig(
        trajectory_path=tmp_path / "trajectory.npy",
        compact_action_ids=(1, 2),
    )
    run = SimpleNamespace(
        environment_factory=SimpleNamespace(config=config),
        learner=SimpleNamespace(model=SimpleNamespace(action_count=2)),
    )

    with pytest.raises(ValueError, match="canonical 78-action"):
        _expert_environment_config(run)


def test_shutdown_signal_skips_processes_that_already_exited() -> None:
    calls: list[None] = []
    shutdown = SimpleNamespace(set=lambda: calls.append(None))

    _signal_shutdown(shutdown, SimpleNamespace(is_alive=lambda: False))
    _signal_shutdown(shutdown, SimpleNamespace(is_alive=lambda: True))

    assert calls == [None]


def test_next_versioned_run_id_advances_letter_suffix(tmp_path: Path) -> None:
    (tmp_path / "trackmania-v37a").mkdir()
    (tmp_path / "trackmania-v37b").mkdir()

    assert _next_versioned_run_id("trackmania-v37a", tmp_path) == "trackmania-v37c"


def test_next_versioned_run_id_uses_numeric_suffix_for_non_versioned_name(tmp_path: Path) -> None:
    (tmp_path / "experiment-1").mkdir()

    assert _next_versioned_run_id("experiment", tmp_path) == "experiment-2"


def test_new_attempt_spec_assigns_free_run_id_for_reset_replay(tmp_path: Path) -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "trackmania-v37a",
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
    (tmp_path / "artifacts" / "trackmania-v37a").mkdir(parents=True)

    result = _new_attempt_spec(
        tmp_path / "run.yaml",
        spec,
        SimpleNamespace(checkpoint=tmp_path / "checkpoint.pt", reset_replay=True),
    )

    assert result.run_id == "trackmania-v37b"


def test_resumed_attempt_spec_recovers_auto_assigned_sibling_run_id(tmp_path: Path) -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "trackmania-v37a",
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
        tmp_path
        / "artifacts"
        / "trackmania-v37b"
        / "checkpoints"
        / "distributed-update-00005000.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    result = _resumed_attempt_spec(
        tmp_path / "run.yaml",
        spec,
        SimpleNamespace(checkpoint=checkpoint, reset_replay=False),
    )

    assert result.run_id == "trackmania-v37b"


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
