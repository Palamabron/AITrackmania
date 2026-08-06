from pathlib import Path
from types import SimpleNamespace

import pytest

from tmrl.cli import (
    _expert_environment_config,
    _new_attempt_spec,
    _next_versioned_run_id,
    _print_benchmark_report,
    _resumed_attempt_spec,
    _signal_shutdown,
    entrypoint,
)
from tmrl.core.spec import RunSpec
from tmrl.trackmania.environment import TrackmaniaEnvironmentConfig


def test_benchmark_report_prints_trials_and_summary(capsys: pytest.CaptureFixture[str]) -> None:
    trials = [
        {
            "trial_index": 0,
            "map_id": "tmrl-test",
            "finished": True,
            "finish_time_s": 37.25,
            "progress_pct": 100.0,
            "telemetry_error": None,
            "controller_error": None,
        },
        {
            "trial_index": 1,
            "map_id": "tmrl-test",
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
    assert "trial=0 map=tmrl-test finished=True time=37.250s" in output
    assert "trial=1 map=tmrl-test finished=False time=-" in output
    assert "finishes=1/2" in output
    assert "mean_completed=37.250s" in output
    assert "median_completed=37.250s" in output


def test_benchmark_accepts_explicit_release_gate_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(args: object) -> None:
        captured.update(vars(args))

    monkeypatch.setattr("tmrl.cli._benchmark", capture)

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
                "learner": {"class_path": "tmrl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
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
                "learner": {"class_path": "tmrl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
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
