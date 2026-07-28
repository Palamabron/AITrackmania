from pathlib import Path
from types import SimpleNamespace

import pytest

from tmrl.cli import (
    _new_attempt_spec,
    _next_versioned_run_id,
    _print_benchmark_report,
    _resumed_attempt_spec,
)
from tmrl.core.spec import RunSpec


def test_benchmark_report_prints_trials_and_summary(capsys: pytest.CaptureFixture[str]) -> None:
    trials = [
        {
            "trial_index": 0,
            "map_id": "test-3",
            "finished": True,
            "finish_time_s": 37.25,
            "progress_pct": 100.0,
            "telemetry_error": None,
            "controller_error": None,
        },
        {
            "trial_index": 1,
            "map_id": "test-3",
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
    assert "trial=0 map=test-3 finished=True time=37.250s" in output
    assert "trial=1 map=test-3 finished=False time=-" in output
    assert "finishes=1/2" in output
    assert "mean_completed=37.250s" in output
    assert "median_completed=37.250s" in output


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
