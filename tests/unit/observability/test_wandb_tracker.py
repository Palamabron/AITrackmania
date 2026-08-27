from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from trackmaniarl.experiments.evaluation import EvaluationResult, aggregate_results
from trackmaniarl.observability.trackers import WandbTracker


class _FakeRun:
    def __init__(self) -> None:
        self.url = ""
        self.definitions: list[tuple[str, dict[str, object]]] = []
        self.logged: list[dict[str, object]] = []
        self.finished: list[int] = []

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.definitions.append((name, kwargs))

    def log(self, values: dict[str, object]) -> None:
        self.logged.append(values)

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class _FailingRun(_FakeRun):
    def log(self, values: dict[str, object]) -> None:
        del values
        raise RuntimeError("remote unavailable")


class _FakeWandb:
    class Settings:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def __init__(self, run: _FakeRun) -> None:
        self.run = run
        self.init_kwargs: dict[str, object] = {}

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run


def _install_wandb(monkeypatch: pytest.MonkeyPatch, run: _FakeRun) -> _FakeWandb:
    wandb = _FakeWandb(run)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return wandb


def _metric_event(run: _FakeRun, metric: str) -> dict[str, object]:
    return next(item for item in run.logged if metric in item)


@dataclass(frozen=True, slots=True)
class _Scenario:
    run: _FakeRun
    wandb: _FakeWandb


def _tracker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: _FakeRun
) -> tuple[WandbTracker, _FakeWandb]:
    wandb = _install_wandb(monkeypatch, run)
    tracker = WandbTracker(
        "project",
        run_dir=str(tmp_path),
        run_id="run-17",
        attempt_id="attempt-2",
        resumed_from="checkpoint-1",
    )
    return tracker, wandb


def _log_distributed_events(tracker: WandbTracker) -> None:
    tracker.log(
        "distributed/ingest",
        {
            "actor_id": "actor-a",
            "transitions": 512,
            "policy_lag_updates": 4,
            "queue_delay_s": 0.25,
            "rollout_queue_depth": 3,
            "utd": 0.75,
        },
        step=7,
    )
    tracker.log("actor/heartbeat", {"actor_id": "actor-a", "spool_bytes": 128}, step=7)


def _log_update(tracker: WandbTracker) -> None:
    tracker.log(
        "train/update",
        {
            "loss/total": 1.25,
            "debug/action_batch_entropy": 0.8,
            "debug/bootstrap_discount_mean": 0.95,
            "debug/n_step_return_mean": 1.5,
            "debug/q_selected_mean": 2.5,
            "debug/q_target_mean": 2.25,
            "debug/initialized_exact_tensors": 99,
            "replay_size": 4_096,
            "updates_per_s": 10.0,
            "health/wal_pending_rows": 3,
            "health/wal_pending_payload_bytes": 1_024,
        },
        step=7,
    )


def _log_episodes(tracker: WandbTracker) -> None:
    first = {
        "index": 1,
        "return": 3.0,
        "progress_pct": 80.0,
        "reward/time": -1.5,
        "control/gas_fraction": 0.75,
        "termination/time_limit": 1.0,
        "timing/step_race_ms_p99": 51.0,
        "telemetry_skipped_frames_total": 4.0,
        "controller_apply_ms_mean": 1.5,
    }
    tracker.log("train/episode", first, step=7)
    tracker.log("train/episode", {"index": 2, "return": 4.0, "progress_pct": 90.0}, step=7)


def _finished_evaluation() -> EvaluationResult:
    return EvaluationResult(
        True,
        35.0,
        False,
        4.0,
        1.0,
        50.0,
        steps=2,
        controller_apply_ms=2.25,
        telemetry_wait_ms=49.0,
        telemetry_skipped_frames_total=3,
    )


def _log_evaluations(tracker: WandbTracker) -> None:
    summary = {
        "finish_rate": 0.75,
        "trials": 4,
        "control_gas_fraction_mean": 0.8,
        "telemetry_skipped_frames_total": 6.0,
    }
    unfinished = EvaluationResult(False, None, True, 2.0, 3.0, 25.0, steps=1)
    tracker.log("eval/summary", summary, step=7)
    tracker.log("eval/suite", aggregate_results([_finished_evaluation(), unfinished]), step=8)


def _scenario(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Scenario:
    run = _FakeRun()
    tracker, wandb = _tracker(monkeypatch, tmp_path, run)
    _log_distributed_events(tracker)
    _log_update(tracker)
    tracker.log("train/progress_bin", {"00_05/action_count": 100}, step=7)
    _log_episodes(tracker)
    _log_evaluations(tracker)
    tracker.close()
    return _Scenario(run, wandb)


def _assert_update_axes(run: _FakeRun) -> None:
    update = _metric_event(run, "learner/loss_total")
    assert update["trainer/update"] == 7
    assert update["env/transitions"] == 512
    assert update["pipeline/policy_lag_updates"] == 4.0
    assert update["health/active_actors"] == 1
    assert update["health/spool_bytes"] == 128
    assert update["health/wal_pending_rows"] == 3
    assert update["health/wal_pending_payload_bytes"] == 1_024


def _assert_bounded_metric_catalog(run: _FakeRun) -> None:
    update = _metric_event(run, "learner/loss_total")
    assert update["learner/action_batch_entropy"] == 0.8
    assert update["learner/bootstrap_discount_mean"] == 0.95
    assert update["learner/n_step_return_mean"] == 1.5
    assert update["learner/q_target_mean"] == 2.25
    assert "learner/initialized_exact_tensors" not in update
    assert all(key.count("/") <= 1 for item in run.logged for key in item)
    assert not any("progress_bin" in key for item in run.logged for key in item)


def _assert_episode_metrics(run: _FakeRun) -> None:
    episodes = [item for item in run.logged if "episode/return" in item]
    assert [item["env/episode"] for item in episodes] == [1, 2]
    assert episodes[0]["episode/reward_time"] == -1.5
    assert episodes[0]["episode/control_gas_fraction"] == 0.75
    assert episodes[0]["episode/termination_time_limit"] == 1.0
    assert episodes[0]["episode/timing_step_race_ms_p99"] == 51.0
    assert episodes[0]["episode/telemetry_skipped_frames_total"] == 4.0
    assert episodes[0]["episode/controller_apply_ms_mean"] == 1.5


def test_wandb_tracker_maps_training_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _scenario(monkeypatch, tmp_path).run
    _assert_update_axes(run)
    _assert_bounded_metric_catalog(run)
    _assert_episode_metrics(run)


def _assert_evaluation_summary(run: _FakeRun) -> None:
    evaluation = _metric_event(run, "evaluation/finish_rate")
    assert evaluation["eval/batch"] == 1
    assert evaluation["evaluation/control_gas_fraction_mean"] == 0.8
    assert evaluation["evaluation/telemetry_skipped_frames_total"] == 6.0


def _assert_evaluation_suite(run: _FakeRun) -> None:
    suite = next(item for item in run.logged if item.get("eval/batch") == 2)
    assert suite["evaluation/finish_rate"] == 0.5
    assert suite["evaluation/finish_time_mean_s"] == 35.0
    assert suite["evaluation/finish_time_median_s"] == 35.0
    assert suite["evaluation/finish_time_best_s"] == 35.0
    assert suite["evaluation/reward"] == 3.0
    assert suite["evaluation/crash_rate"] == 0.5
    assert suite["evaluation/sub_36_rate"] == 0.5
    assert suite["evaluation/sub_38_rate"] == 0.5
    assert suite["evaluation/sub_40_rate"] == 0.5
    assert suite["evaluation/controller_apply_ms"] == 1.5
    assert suite["evaluation/telemetry_wait_ms"] == pytest.approx(98 / 3)
    assert suite["evaluation/telemetry_skipped_frames_total"] == 3.0


def test_wandb_tracker_maps_evaluation_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _scenario(monkeypatch, tmp_path).run
    _assert_evaluation_summary(run)
    _assert_evaluation_suite(run)


def _assert_configuration(scenario: _Scenario) -> None:
    definitions = dict(scenario.run.definitions)
    assert definitions["learner/*"] == {"step_metric": "trainer/update"}
    assert definitions["episode/*"] == {"step_metric": "env/episode"}
    assert definitions["evaluation/*"] == {"step_metric": "eval/batch"}
    assert scenario.wandb.init_kwargs["group"] == "run-17"
    settings = scenario.wandb.init_kwargs["settings"]
    assert isinstance(settings, _FakeWandb.Settings)
    assert settings.kwargs["console"] == "wrap"
    config = scenario.wandb.init_kwargs["config"]
    assert isinstance(config, dict)
    assert config["observability/attempt_id"] == "attempt-2"
    assert config["observability/resumed_from"] == "checkpoint-1"
    assert scenario.run.finished == [0]


def test_wandb_tracker_configuration_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _assert_configuration(_scenario(monkeypatch, tmp_path))
    with pytest.raises(ValueError, match="queue_size must be positive"):
        WandbTracker("project", queue_size=0)


def test_wandb_tracker_marks_remote_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _FailingRun()
    _install_wandb(monkeypatch, run)
    tracker = WandbTracker("project", run_dir=str(tmp_path))

    tracker.log("train/update", {"loss/total": 1.0}, step=1)
    tracker.close()

    assert run.finished == [1]
