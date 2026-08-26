from __future__ import annotations

import sys
from pathlib import Path

import pytest

from trackmaniarl.observability.trackers import WandbTracker


class _FakeRun:
    def __init__(self, *, fail_logging: bool = False) -> None:
        self.url = ""
        self.fail_logging = fail_logging
        self.definitions: list[tuple[str, dict[str, object]]] = []
        self.logged: list[dict[str, object]] = []
        self.finished: list[int] = []

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.definitions.append((name, kwargs))

    def log(self, values: dict[str, object]) -> None:
        if self.fail_logging:
            raise RuntimeError("remote unavailable")
        self.logged.append(values)

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


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


def test_wandb_tracker_uses_domain_axes_and_bounded_metric_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _FakeRun()
    wandb = _install_wandb(monkeypatch, run)
    tracker = WandbTracker(
        "project",
        run_dir=str(tmp_path),
        run_id="run-17",
        attempt_id="attempt-2",
        resumed_from="checkpoint-1",
    )

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
        },
        step=7,
    )
    tracker.log("train/progress_bin", {"00_05/action_count": 100}, step=7)
    tracker.log(
        "train/episode",
        {
            "index": 1,
            "return": 3.0,
            "progress_pct": 80.0,
            "reward/time": -1.5,
            "control/gas_fraction": 0.75,
            "termination/time_limit": 1.0,
            "timing/step_race_ms_p99": 51.0,
        },
        step=7,
    )
    tracker.log("train/episode", {"index": 2, "return": 4.0, "progress_pct": 90.0}, step=7)
    tracker.log(
        "eval/summary",
        {"finish_rate": 0.75, "trials": 4, "control_gas_fraction_mean": 0.8},
        step=7,
    )
    tracker.close()

    update = _metric_event(run, "learner/loss_total")
    assert update["trainer/update"] == 7
    assert update["env/transitions"] == 512
    assert update["pipeline/policy_lag_updates"] == 4.0
    assert update["health/active_actors"] == 1
    assert update["health/spool_bytes"] == 128
    assert update["learner/action_batch_entropy"] == 0.8
    assert update["learner/bootstrap_discount_mean"] == 0.95
    assert update["learner/n_step_return_mean"] == 1.5
    assert update["learner/q_target_mean"] == 2.25
    assert "learner/initialized_exact_tensors" not in update
    assert all(key.count("/") <= 1 for item in run.logged for key in item)

    episodes = [item for item in run.logged if "episode/return" in item]
    assert [item["env/episode"] for item in episodes] == [1, 2]
    assert episodes[0]["episode/reward_time"] == -1.5
    assert episodes[0]["episode/control_gas_fraction"] == 0.75
    assert episodes[0]["episode/termination_time_limit"] == 1.0
    assert episodes[0]["episode/timing_step_race_ms_p99"] == 51.0
    evaluation = _metric_event(run, "evaluation/finish_rate")
    assert evaluation["eval/batch"] == 1
    assert evaluation["evaluation/control_gas_fraction_mean"] == 0.8
    assert not any("progress_bin" in key for item in run.logged for key in item)

    definitions = dict(run.definitions)
    assert definitions["learner/*"] == {"step_metric": "trainer/update"}
    assert definitions["episode/*"] == {"step_metric": "env/episode"}
    assert definitions["evaluation/*"] == {"step_metric": "eval/batch"}
    assert wandb.init_kwargs["group"] == "run-17"
    settings = wandb.init_kwargs["settings"]
    assert isinstance(settings, _FakeWandb.Settings)
    assert settings.kwargs["console"] == "wrap"
    config = wandb.init_kwargs["config"]
    assert isinstance(config, dict)
    assert config["observability/attempt_id"] == "attempt-2"
    assert config["observability/resumed_from"] == "checkpoint-1"
    assert run.finished == [0]


def test_wandb_tracker_marks_remote_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _FakeRun(fail_logging=True)
    _install_wandb(monkeypatch, run)
    tracker = WandbTracker("project", run_dir=str(tmp_path))

    tracker.log("train/update", {"loss/total": 1.0}, step=1)
    tracker.close()

    assert run.finished == [1]


def test_wandb_tracker_rejects_non_positive_queue_size() -> None:
    with pytest.raises(ValueError, match="queue_size must be positive"):
        WandbTracker("project", queue_size=0)
