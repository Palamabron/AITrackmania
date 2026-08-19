from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.cli import (
    _smoke_training,
    _trajectory_optimize,
    _validate,
    entrypoint,
)
from trackmaniarl.core.spec import RunSpec, TrainingSpec


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


def test_smoke_training_reserves_transitions_for_a_learner_update() -> None:
    training = _smoke_training(TrainingSpec(batch_size=256, n_step=3), 100)

    assert training.batch_size == 49
    assert training.warmup_transitions == 51
    assert training.total_transitions > training.warmup_transitions


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
