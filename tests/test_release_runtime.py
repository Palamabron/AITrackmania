"""Release contracts for observability, resume, and the optional game adapter."""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tmrl.core.builtins import JsonlRunLogger, TorchCheckpointCodec
from tmrl.core.spec import RunSpec
from tmrl.distributed.actor import ActorRuntime
from tmrl.experiments.orchestration import GridStrategy, StudyLedger, StudyRunner, StudySpec
from tmrl.observability.trackers import WandbTracker, _wandb_metric_name
from tmrl.project.scaffold import create_project
from tmrl.trackmania.assets import record_boundary, record_trajectory
from tmrl.trackmania.environment import OpenPlanetEnvironment, OpenPlanetEnvironmentFactory
from tmrl.trackmania.evaluation import TrackmaniaEvaluator
from tmrl.trackmania.reward import TrajectoryReward
from tmrl.trackmania.telemetry import (
    DEFAULT_POSITION_INDICES,
    DEFAULT_TELEMETRY_FIELD_COUNT,
    OpenPlanetClient,
    TelemetryFrame,
)


def test_jsonl_events_have_release_envelope(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path, run_id="release")
    logger.log("train/update", {"loss": 1.0}, step=3)
    logger.close()
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["schema_version"] == "1.0"
    assert event["run_id"] == "release"
    assert event["timestamp_utc"]
    assert event["elapsed_s"] >= 0
    assert event["segment_id"]


def test_spawn_context_uses_the_active_virtual_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from tmrl import cli

    executable: list[str] = []
    expected_context = object()
    monkeypatch.setattr(cli.sys, "executable", "C:/venv/Scripts/python.exe")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(cli.multiprocessing, "set_executable", executable.append)
    monkeypatch.setattr(cli.multiprocessing, "get_context", lambda method: expected_context)

    assert cli._spawn_context() is expected_context
    assert executable == ["C:/venv/Scripts/python.exe"]


def test_spawn_executable_prefers_the_active_virtual_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tmrl import cli

    directory = "Scripts" if cli.os.name == "nt" else "bin"
    filename = "python.exe" if cli.os.name == "nt" else "python"
    executable = tmp_path / directory / filename
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))

    assert cli._spawn_executable() == str(executable)


def test_torch_checkpoints_are_zstd_streamed_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    codec = TorchCheckpointCodec()
    state = {"tensor": torch.zeros(1024, dtype=torch.float32), "counter": 3}

    codec.save(state, path)
    restored = codec.load(path)

    assert path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    assert path.stat().st_size < state["tensor"].numel() * state["tensor"].element_size()
    assert torch.equal(restored["tensor"], state["tensor"])
    assert restored["counter"] == 3


def test_torch_checkpoint_round_trips_numpy_replay_state_with_weights_only(
    tmp_path: Path,
) -> None:
    import random

    path = tmp_path / "checkpoint.pt"
    codec = TorchCheckpointCodec()
    state = {
        "array": np.arange(4, dtype=np.float32),
        "flags": np.zeros(2, dtype=np.bool_),
        "rng": random.Random(0).getstate(),
        "info": {3: {"progress_pct": 1.0}},
    }

    codec.save(state, path)
    restored = codec.load(path)

    np.testing.assert_array_equal(restored["array"], state["array"])
    np.testing.assert_array_equal(restored["flags"], state["flags"])
    assert restored["rng"] == state["rng"]
    assert restored["info"] == state["info"]


def test_wandb_metrics_use_readable_sections() -> None:
    assert _wandb_metric_name("train/episode", "finish_time_s") == "episode/finish_time_s"
    assert _wandb_metric_name("train/update", "loss/iqn") == "learner/loss/iqn"
    assert (
        _wandb_metric_name("train/update", "timing/replay_sample_s")
        == "performance/replay_sample_s"
    )
    assert _wandb_metric_name("train/update", "replay_size") == "replay/size"
    assert (
        _wandb_metric_name("train/update", "debug/gradient_norm_max") == "learner/gradient_norm_max"
    )
    assert (
        _wandb_metric_name("train/update", "debug/gradient_clipped_fraction")
        == "learner/clipped_fraction"
    )
    assert _wandb_metric_name("train/update", "debug/q_selected_mean") == "learner/q_mean"
    assert _wandb_metric_name("train/update", "debug/q_selected_max") == "learner/q_max"
    assert _wandb_metric_name("train/update", "debug/q_selected_abs_max") == "learner/q_abs_max"


def test_wandb_tracker_queues_remote_logging_without_reusing_global_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logged: list[dict[str, object]] = []

    class FakeWandb:
        class Settings:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

        @staticmethod
        def init(**kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(url="")

        @staticmethod
        def log(values: dict[str, object]) -> None:
            logged.append(values)

        @staticmethod
        def finish(*, exit_code: int) -> None:
            assert exit_code == 0

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    tracker = WandbTracker("project", run_dir=str(tmp_path))
    tracker.log("train/episode", {"index": 1}, step=10)
    tracker.log("train/episode", {"index": 2}, step=10)
    tracker.close()

    assert logged == [{"episode/index": 1}, {"episode/index": 2}]


def test_actor_episode_summary_zeroes_non_finish_time() -> None:
    summary = ActorRuntime._summary(
        12.0,
        {
            "termination_reason": "no_progress",
            "race_time_ms": 45_000.0,
            "reward_time": -4.5,
            "reward_pbrs": 0.2,
            "reward_terminal": -1.0,
        },
        6,
    )

    assert summary["return"] == 12.0
    assert summary["reward_per_transition"] == 2.0
    assert summary["reward/time"] == -4.5
    assert summary["reward/pbrs"] == 0.2
    assert summary["finish_time_s"] == 0.0
    assert summary["finished"] == 0.0


def test_actor_episode_summary_reports_finish_time() -> None:
    summary = ActorRuntime._summary(
        20.0,
        {"termination_reason": "finished", "race_time_ms": 12_345.0},
        10,
    )

    assert summary["finish_time_s"] == pytest.approx(12.345)
    assert summary["finished"] == 1.0


def test_environment_step_reports_applied_control_and_race_time_delta() -> None:
    from tmrl.trackmania.actions import build_brake_tap_action_table
    from tmrl.trackmania.control import RecordingController

    class Client:
        def read(self) -> TelemetryFrame:
            values = np.zeros(33, dtype=np.float32)
            values[3] = 166.0
            return TelemetryFrame(values)

    def reward_step(position: np.ndarray, **kwargs: object) -> SimpleNamespace:
        del position, kwargs
        return SimpleNamespace(
            reward=1.0,
            terminated=False,
            reason=None,
            time_reward=0.0,
            pbrs_reward=0.0,
            progress_reward=0.0,
            projected_velocity_reward=0.0,
            projected_speed_reward=0.0,
            steering_delta_reward=0.0,
            collision_reward=0.0,
            collided=False,
            collision_detected=False,
            terminal_reward=0.0,
            potential_progress=0.0,
            projected_velocity_mps=0.0,
            projected_velocity_ratio=0.0,
        )

    environment = object.__new__(OpenPlanetEnvironment)
    environment.config = SimpleNamespace(
        action_repeat_frames=2,
        position_indices=(4, 5, 6),
        velocity_indices=(7, 8, 9),
    )
    environment.client = Client()
    environment.controller = RecordingController()
    environment.reward = SimpleNamespace(step=reward_step, progress_m=12.0, progress_pct=0.5)
    environment._episode_started_at = 0.0
    environment._last_race_time_ms = 100.0
    environment._action_count, environment._action_table = build_brake_tap_action_table()

    _, _, _, _, info = environment.step(3)

    assert info["control_gas"] == 1.0
    assert info["control_brake"] == 0.0
    assert info["control_steer"] == -1.0
    assert info["step_race_time_ms"] == pytest.approx(66.0)
    assert info["race_time_ms"] == pytest.approx(166.0)
    assert environment._last_race_time_ms == pytest.approx(166.0)


def test_environment_waits_for_race_timer_restart() -> None:
    class Client:
        def __init__(self) -> None:
            self._race_times_ms = iter((1_000.0, 0.0, 50.0))

        def read(self) -> TelemetryFrame:
            return TelemetryFrame(
                np.asarray([0.0, 0.0, 0.0, next(self._race_times_ms)], dtype=np.float32)
            )

    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = Client()
    environment.config = SimpleNamespace(start_timeout_s=1.0, start_poll_s=0.0)

    frame = environment._wait_for_active_run(500.0)

    assert float(frame.values[3]) == 50.0


def test_environment_retries_an_unconfirmed_restart() -> None:
    class Controller:
        def __init__(self) -> None:
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

    class Client:
        def __init__(self) -> None:
            self._race_times_ms = iter((1_000.0, 2_000.0))

        def read(self) -> TelemetryFrame:
            return TelemetryFrame(
                np.asarray([0.0, 0.0, 0.0, next(self._race_times_ms)], dtype=np.float32)
            )

    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = Client()
    environment.controller = Controller()
    environment.evaluation_map = None
    attempts = 0
    frame = TelemetryFrame(np.asarray([0.0, 0.0, 0.0, 50.0], dtype=np.float32))

    def wait_for_active_run(previous_race_time_ms: float) -> TelemetryFrame:
        nonlocal attempts
        attempts += 1
        assert previous_race_time_ms in {1_000.0, 2_000.0}
        if attempts == 1:
            raise TimeoutError
        return frame

    environment._wait_for_active_run = wait_for_active_run

    assert environment._restart_race() is frame
    assert environment.controller.resets == 2


def test_openplanet_client_validates_a_complete_packet() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            connection.sendall(struct.pack("<fff", 1.0, 2.0, 3.0))
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = OpenPlanetClient(host, port, field_count=3, timeout_s=1)
    try:
        assert np.array_equal(client.read().values, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    finally:
        client.close()
        thread.join(timeout=1)


def test_openplanet_client_discards_queued_stale_frames() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            connection.sendall(
                struct.pack("<fff", 1.0, 2.0, 3.0)
                + struct.pack("<fff", 4.0, 5.0, 6.0)
                + struct.pack("<fff", 7.0, 8.0, 9.0)
            )
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = OpenPlanetClient(host, port, field_count=3, timeout_s=1)
    try:
        assert np.array_equal(client.read().values, np.array([7.0, 8.0, 9.0], dtype=np.float32))
    finally:
        client.close()
        thread.join(timeout=1)


def test_openplanet_client_reconnects_after_the_producer_closes() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    host, port = server.getsockname()

    def serve() -> None:
        first, _ = server.accept()
        first.close()
        second, _ = server.accept()
        with second:
            second.sendall(struct.pack("<fff", 4.0, 5.0, 6.0))
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = OpenPlanetClient(host, port, field_count=3, timeout_s=1)
    try:
        assert np.array_equal(client.read().values, np.array([4.0, 5.0, 6.0], dtype=np.float32))
    finally:
        client.close()
        thread.join(timeout=1)


def test_openplanet_client_keeps_a_valid_origin_position_after_a_previous_frame() -> None:
    client = OpenPlanetClient(field_count=DEFAULT_TELEMETRY_FIELD_COUNT)
    first = np.zeros(DEFAULT_TELEMETRY_FIELD_COUNT, dtype=np.float32)
    first[4:7] = [100.0, 0.0, 100.0]
    first[16] = 10.0
    origin = np.zeros(DEFAULT_TELEMETRY_FIELD_COUNT, dtype=np.float32)
    origin[16] = 10.0

    client._validate_grab_data(first)
    client._validate_grab_data(origin)
    assert np.array_equal(origin[4:7], np.zeros(3, dtype=np.float32))


def test_trajectory_reward_reports_progress_finish_and_off_track() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    progress = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )
    assert progress.reward > 0
    assert progress.reward == (
        progress.time_reward
        + progress.pbrs_reward
        + progress.progress_reward
        + progress.projected_velocity_reward
        + progress.steering_delta_reward
    )
    assert (
        reward.step(
            np.array([2, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=200.0,
        ).reason
        is None
    )
    finish = reward.step(
        np.array([2, 0, 0]),
        finish_ui_active=True,
        velocity=np.zeros(3),
        race_time_ms=300.0,
    )
    assert finish.reason == "finished"
    assert finish.reward == (
        finish.time_reward
        + finish.pbrs_reward
        + finish.progress_reward
        + finish.projected_velocity_reward
        + finish.steering_delta_reward
        + finish.terminal_reward
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    assert (
        reward.step(
            np.array([100, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=100.0,
        ).reason
        == "off_track"
    )


def _finish_at(race_time_ms: float, **kwargs: float) -> float:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
        finish_reward=25.0,
        **kwargs,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    reward.step(
        np.array([1, 0, 0]), finish_ui_active=False, velocity=np.zeros(3), race_time_ms=10.0
    )
    finish = reward.step(
        np.array([2, 0, 0]),
        finish_ui_active=True,
        velocity=np.zeros(3),
        race_time_ms=race_time_ms,
    )
    assert finish.reason == "finished"
    return finish.terminal_reward


def test_finish_bonus_pays_a_faster_lap_strictly_more() -> None:
    scale = {"finish_time_bonus_per_second": 5.0, "finish_reference_time_s": 45.0}

    assert _finish_at(36_000.0) == pytest.approx(25.0)
    assert _finish_at(36_000.0, **scale) == pytest.approx(25.0 + 5.0 * 9.0)
    assert _finish_at(38_000.0, **scale) == pytest.approx(25.0 + 5.0 * 7.0)
    assert _finish_at(36_000.0, **scale) - _finish_at(38_000.0, **scale) == pytest.approx(10.0)
    assert _finish_at(50_000.0, **scale) == pytest.approx(25.0)


def test_finish_bonus_rejects_negative_settings() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TrajectoryReward(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            finish_time_bonus_per_second=-1.0,
        )


def test_trajectory_reward_has_dense_progress_signal_and_stall_termination() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        no_progress_steps=3,
        slow_progress_window_steps=10,
        minimum_finish_steps=1,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    assert (
        reward.step(
            np.array([1, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=100.0,
        ).reward
        > 0.0
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    assert (
        reward.step(
            np.array([0, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=100.0,
        ).reason
        is None
    )
    assert (
        reward.step(
            np.array([0, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=200.0,
        ).reason
        is None
    )
    assert (
        reward.step(
            np.array([0, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=300.0,
        ).reason
        == "no_progress"
    )


def test_trajectory_reward_applies_a_light_nonterminal_collision_penalty() -> None:
    trajectory = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    without_collision = TrajectoryReward(
        trajectory,
        collision_penalty=0.05,
        time_penalty_per_second=0.0,
    )
    with_collision = TrajectoryReward(
        trajectory,
        collision_penalty=0.05,
        time_penalty_per_second=0.0,
    )
    for reward in (without_collision, with_collision):
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    baseline = without_collision.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )
    collision = with_collision.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
        collision=True,
    )

    assert not collision.terminated
    assert collision.reason is None
    assert collision.collided
    assert collision.collision_reward == pytest.approx(-0.05)
    assert collision.reward == pytest.approx(baseline.reward - 0.05)


def test_trajectory_reward_debounces_collision_penalties_by_race_time() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        collision_penalty=0.5,
        collision_cooldown_s=2.0,
        time_penalty_per_second=0.0,
        potential_progress_weight=0.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    first = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
        collision=True,
    )
    repeated = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=1_100.0,
        collision=True,
    )
    later = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=2_100.0,
        collision=True,
    )

    assert first.collision_detected
    assert first.collided
    assert first.collision_reward == pytest.approx(-0.5)
    assert repeated.collision_detected
    assert not repeated.collided
    assert repeated.collision_reward == 0.0
    assert later.collision_detected
    assert later.collided
    assert later.collision_reward == pytest.approx(-0.5)


def test_trajectory_reward_does_not_skip_ahead_at_a_track_crossover() -> None:
    # Point 4 passes close to point 1, but comes after a distant part of the lap.
    # A global nearest-point lookup would award four samples of progress at once.
    reward = TrajectoryReward(
        np.array(
            [[0, 0, 0], [10, 0, 0], [20, 0, 0], [20, 0, 10], [10, 0, 1], [0, 0, 1]],
            dtype=np.float32,
        ),
        nearest_forward_points=2,
        minimum_finish_steps=1,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        np.array([10, 0, 0.9]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )

    assert result.pbrs_reward > 0.0
    assert reward._index == 1


def test_trajectory_reward_does_not_reward_perpendicular_velocity() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
        max_projected_speed_mps=10.0,
        velocity_to_mps_scale=1.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.array([0, 10, 0]),
        race_time_ms=100.0,
    )

    assert result.projected_velocity_ratio == 0.0
    assert result.projected_velocity_reward == 0.0


def test_trajectory_reward_projects_velocity_without_an_index_change() -> None:
    trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    reward = TrajectoryReward(
        trajectory,
        max_projected_speed_mps=10.0,
        velocity_to_mps_scale=1.0,
        projected_velocity_scale=0.1,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    forward = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.array([10, 0, 0]),
        race_time_ms=100.0,
    )
    stationary = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.array([10, 0, 0]),
        race_time_ms=200.0,
    )

    assert forward.projected_velocity_mps == pytest.approx(10.0)
    assert forward.projected_velocity_reward == pytest.approx(0.1)
    assert stationary.projected_velocity_reward == pytest.approx(0.1)


def test_trajectory_reward_penalizes_reverse_projected_velocity() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        projected_velocity_scale=0.1,
        velocity_to_mps_scale=1.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.array([-10, 0, 0]),
        race_time_ms=100.0,
    )

    assert result.projected_velocity_mps == pytest.approx(-10.0)
    assert result.projected_velocity_reward == pytest.approx(-0.1)


def test_trajectory_reward_bonus_separates_high_projected_speed() -> None:
    trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)

    def reward_at(speed: float) -> float:
        reward = TrajectoryReward(
            trajectory,
            max_projected_speed_mps=10.0,
            velocity_to_mps_scale=1.0,
            projected_speed_bonus_scale=0.5,
        )
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
        result = reward.step(
            np.array([0, 0, 0]),
            finish_ui_active=False,
            velocity=np.array([speed, 0, 0]),
            race_time_ms=100.0,
        )
        return result.projected_speed_reward

    assert reward_at(10.0) == pytest.approx(0.05)
    assert reward_at(5.0) == pytest.approx(0.0125)
    assert reward_at(-10.0) == 0.0


def test_trajectory_reward_pace_bonus_prefers_a_faster_fixed_distance() -> None:
    trajectory = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)

    def lap_reward(speed: float, elapsed_s: float) -> float:
        reward = TrajectoryReward(
            trajectory,
            minimum_finish_steps=1,
            progress_reward_full_lap=100.0,
            finish_reward=20.0,
            max_projected_speed_mps=10.0,
            velocity_to_mps_scale=1.0,
            projected_velocity_scale=0.1,
            projected_speed_bonus_scale=8.0,
            time_penalty_per_second=0.0,
            potential_progress_weight=0.0,
        )
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
        result = reward.step(
            np.array([10, 0, 0]),
            finish_ui_active=True,
            velocity=np.array([speed, 0, 0]),
            race_time_ms=elapsed_s * 1_000.0,
        )
        assert result.terminated
        return result.reward

    assert lap_reward(10.0, 1.0) > lap_reward(5.0, 2.0)


def test_trajectory_reward_penalizes_steering_delta() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        steering_delta_penalty=0.1,
        time_penalty_per_second=0.0,
        potential_progress_weight=0.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    first = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
        steering=0.5,
    )
    unchanged = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=200.0,
        steering=0.5,
    )
    reversal = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=300.0,
        steering=-0.5,
    )

    assert first.steering_delta_reward == pytest.approx(-0.05)
    assert unchanged.steering_delta_reward == 0.0
    assert reversal.steering_delta_reward == pytest.approx(-0.1)


def test_trajectory_reward_pbrs_telescopes_at_the_terminal_state() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    progress = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )
    finish = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=True,
        velocity=np.zeros(3),
        race_time_ms=200.0,
    )

    assert progress.pbrs_reward + reward.reward_gamma * finish.pbrs_reward == pytest.approx(0.0)


def test_trajectory_reward_prefers_farther_failed_run() -> None:
    trajectory = np.asarray([[index, 0, 0] for index in range(11)], dtype=np.float32)

    def failed_return(progress_index: int) -> float:
        reward = TrajectoryReward(
            trajectory,
            no_progress_steps=2,
            slow_progress_window_steps=10,
            terminal_failure_penalty=2.0,
            time_penalty_per_second=0.0,
            potential_progress_weight=0.0,
        )
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
        total = 0.0
        for race_time_ms in (100.0, 200.0, 300.0):
            result = reward.step(
                np.array([progress_index, 0, 0]),
                finish_ui_active=False,
                velocity=np.zeros(3),
                race_time_ms=race_time_ms,
            )
            total += result.reward
            if result.terminated:
                break
        return total

    assert failed_return(4) > failed_return(0)


def test_trajectory_reward_high_gamma_limits_stationary_pbrs_penalty() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        reward_gamma=0.9995,
    )
    reward.reset(np.array([1, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )

    assert result.pbrs_reward == pytest.approx(-0.0005)


def test_trajectory_reward_prefers_a_shorter_completed_run() -> None:
    trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)

    def completed_return(step_time_ms: float) -> float:
        reward = TrajectoryReward(trajectory, minimum_finish_steps=1)
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
        first = reward.step(
            np.array([1, 0, 0]),
            finish_ui_active=False,
            velocity=np.zeros(3),
            race_time_ms=step_time_ms,
        )
        finish = reward.step(
            np.array([2, 0, 0]),
            finish_ui_active=True,
            velocity=np.zeros(3),
            race_time_ms=2.0 * step_time_ms,
        )
        return first.reward + reward.reward_gamma * finish.reward

    assert completed_return(100.0) > completed_return(200.0)


def test_trajectory_reward_reset_restarts_the_race_time_delta() -> None:
    reward = TrajectoryReward(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32))
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    first = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
    second = reward.step(
        np.array([0, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )

    assert second.time_reward == first.time_reward


def test_trajectory_reward_does_not_terminate_an_unconfirmed_finish() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    first = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=100.0,
    )
    second = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=200.0,
    )

    assert first.reason is None
    assert second.reason is None
    assert not second.terminated


def test_trajectory_recorder_writes_portable_csv(tmp_path: Path) -> None:
    class Client:
        def read(self) -> TelemetryFrame:
            return TelemetryFrame(np.arange(33, dtype=np.float32))

    path = record_trajectory(tmp_path / "trajectory.csv", Client(), samples=2)
    points = np.loadtxt(path, delimiter=",")
    assert points.shape == (2, 3)
    assert np.array_equal(points[0], np.asarray(DEFAULT_POSITION_INDICES, dtype=np.float32))


def test_boundary_recorder_stops_when_the_game_reports_a_finish(tmp_path: Path) -> None:
    class Client:
        def __init__(self) -> None:
            self.count = 0

        def read(self) -> TelemetryFrame:
            self.count += 1
            values = np.zeros(33, dtype=np.float32)
            values[list(DEFAULT_POSITION_INDICES)] = self.count
            values[2] = float(self.count in {1, 4})
            values[3] = 0.0
            return TelemetryFrame(values)

    messages: list[str] = []
    path = record_boundary(
        tmp_path / "boundary.npy", Client(), minimum_spacing_m=0.1, status=messages.append
    )

    assert np.load(path).shape == (3, 3)
    assert messages == [
        "Waiting for an active run; restart the map if it is on the finish screen.",
        "Recording started at race time 0 ms after moving 1.7 m.",
        "Finish detected at race time 0 ms after 3 samples.",
    ]


def test_openplanet_defaults_match_the_bundled_plugin_and_resolve_assets(tmp_path: Path) -> None:
    environment = OpenPlanetEnvironmentFactory(
        {"trajectory_path": "assets/trajectory.csv"}, base_dir=tmp_path
    )
    assert environment.config.field_count == DEFAULT_TELEMETRY_FIELD_COUNT
    assert environment.config.position_indices == DEFAULT_POSITION_INDICES
    assert environment.config.trajectory_path == tmp_path / "assets" / "trajectory.csv"


def test_openplanet_configuration_rejects_positions_outside_the_packet() -> None:
    with pytest.raises(ValueError, match="position_indices"):
        OpenPlanetEnvironmentFactory(
            {"trajectory_path": "trajectory.csv", "field_count": 3}, base_dir="."
        )


def test_gamepad_reset_uses_the_trackmania_respawn_button(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGamepad:
        def __init__(self) -> None:
            self.events: list[object] = []
            self.callback: object | None = None

        def register_notification(self, *, callback_function: object) -> None:
            self.callback = callback_function

        def reset(self) -> None:
            self.events.append("reset")

        def press_button(self, *, button: int) -> None:
            self.events.append(("press", button))

        def release_button(self, *, button: int) -> None:
            self.events.append(("release", button))

        def update(self) -> None:
            self.events.append("update")

    from tmrl.trackmania import control

    monkeypatch.setitem(sys.modules, "vgamepad", SimpleNamespace(VX360Gamepad=FakeGamepad))
    monkeypatch.setattr(control, "sleep", lambda _: None)
    gamepad = control.GamepadController()
    gamepad.reset()
    assert gamepad._gamepad.events == [
        "reset",
        ("press", 0x2000),
        "update",
        ("release", 0x2000),
        "update",
    ]


def test_gamepad_ignores_a_brake_tap_callback_cancelled_by_a_newer_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGamepad:
        def __init__(self) -> None:
            self.events: list[object] = []
            self.callback: object | None = None

        def register_notification(self, *, callback_function: object) -> None:
            self.callback = callback_function

        def right_trigger_float(self, value: float) -> None:
            self.events.append(("gas", value))

        def left_trigger_float(self, value: float) -> None:
            self.events.append(("brake", value))

        def left_joystick_float(self, value: float, _: float) -> None:
            self.events.append(("steer", value))

        def update(self) -> None:
            self.events.append("update")

    from tmrl.trackmania import control

    monkeypatch.setitem(sys.modules, "vgamepad", SimpleNamespace(VX360Gamepad=FakeGamepad))
    gamepad = control.GamepadController()
    gamepad._tap_generation = 2
    gamepad._release_tap(1, 1.0, 0.0)

    assert gamepad._gamepad.events == []


def test_gamepad_consumes_haptic_collision_events(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGamepad:
        def __init__(self) -> None:
            self.callback: object | None = None

        def register_notification(self, *, callback_function: object) -> None:
            self.callback = callback_function

    from tmrl.trackmania import control

    monkeypatch.setitem(sys.modules, "vgamepad", SimpleNamespace(VX360Gamepad=FakeGamepad))
    gamepad = control.GamepadController()
    assert callable(gamepad._gamepad.callback)
    from inspect import signature

    def expected_callback(
        client: object,
        target: object,
        large_motor: int,
        small_motor: int,
        led_number: int,
        user_data: object,
    ) -> None:
        return None

    expected_callback.__annotations__.clear()
    assert signature(gamepad._on_vibration) == signature(expected_callback)
    gamepad._gamepad.callback(None, None, 101, 0, 0, None)

    assert gamepad.consume_collision()
    assert not gamepad.consume_collision()


def test_gamepad_allows_platforms_without_haptic_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGamepad:
        pass

    from tmrl.trackmania import control

    monkeypatch.setitem(sys.modules, "vgamepad", SimpleNamespace(VX360Gamepad=FakeGamepad))
    gamepad = control.GamepadController()

    assert not gamepad.consume_collision()


def test_trackmania_template_contains_first_party_components(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent", template="trackmania")
    config = (target / "run.yaml").read_text(encoding="utf-8")
    assert "OpenPlanetEnvironmentFactory" in config
    assert "TrackmaniaEvaluator" in config
    assert (target / "assets" / "trajectory.csv").is_file()
    assert (target / "maps").is_dir()
    plugin = target / "openplanet" / "TMRL_GrabData_IQN.as"
    assert plugin.is_file()
    assert 'const string PROTOCOL_VERSION = "2"' in plugin.read_text(encoding="utf-8")
    assert RunSpec.from_yaml(target / "run.yaml").evaluation is not None


def test_generated_project_uses_the_current_checkout_before_first_publish(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent")
    pyproject = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = pyproject["project"]["dependencies"][0]
    assert requirement == "tmrl[distributed]"
    assert requirement.count("tmrl") == 1
    assert pyproject["tool"]["uv"]["sources"]["tmrl"]["editable"] is True
    assert "pytest>=7.0" in pyproject["dependency-groups"]["dev"]


def test_trackmania_evaluator_runs_every_declared_seed_and_episode() -> None:
    class Environment:
        def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
            del seed
            return 0.0, {}

        def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, str | float]]:
            del action
            return (
                1.0,
                2.0,
                True,
                False,
                {
                    "termination_reason": "finished",
                    "race_time_ms": 12_345.0,
                },
            )

        def close(self) -> None:
            return None

    class EnvironmentFactory:
        def create(self, *, seed: int) -> Environment:
            del seed
            return Environment()

    class Pipeline:
        def transform_observation(self, observation: object) -> object:
            return observation

    class Policy:
        def act(self, observation: object, *, deterministic: bool = False) -> float:
            del observation
            assert deterministic
            return 0.0

    suite = SimpleNamespace(seeds=(1, 2), episodes_per_seed=2)
    metrics = TrackmaniaEvaluator(suite, EnvironmentFactory(), Pipeline()).evaluate(Policy())
    assert metrics["eval/finish_rate"] == 1.0
    assert metrics["eval/reward"] == 2.0
    assert metrics["eval/finish_time_s"] == pytest.approx(12.345)


def test_trackmania_evaluator_records_telemetry_errors_and_continues() -> None:
    class FailingEnvironment:
        def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
            del seed
            raise TimeoutError("telemetry stalled")

        def close(self) -> None:
            return None

    class FinishingEnvironment:
        def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
            del seed
            return 0.0, {}

        def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, str | float]]:
            del action
            return (
                1.0,
                2.0,
                True,
                False,
                {"termination_reason": "finished", "race_time_ms": 12_345.0},
            )

        def close(self) -> None:
            return None

    class EnvironmentFactory:
        def __init__(self) -> None:
            self.created = 0

        def create(self, *, seed: int) -> object:
            del seed
            self.created += 1
            return FailingEnvironment() if self.created == 1 else FinishingEnvironment()

    class Pipeline:
        def transform_observation(self, observation: object) -> object:
            return observation

    class Policy:
        def act(self, observation: object, *, deterministic: bool = False) -> float:
            del observation, deterministic
            return 0.0

    suite = SimpleNamespace(seeds=(1,), episodes_per_seed=2)
    metrics = TrackmaniaEvaluator(suite, EnvironmentFactory(), Pipeline()).evaluate(Policy())

    assert metrics["eval/finish_rate"] == 0.5


def test_trackmania_evaluator_reuses_environment_for_map_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Geometry:
        def __init__(self, path: str, *, expected_map_uid: str) -> None:
            del path, expected_map_uid

        def validate_map(self, path: str) -> None:
            del path

    class Environment:
        def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
            del seed
            return 0.0, {}

        def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, str | float]]:
            del action
            return (
                1.0,
                1.0,
                True,
                False,
                {
                    "termination_reason": "finished",
                    "race_time_ms": 12_345.0,
                },
            )

        def close(self) -> None:
            return None

    class EnvironmentFactory:
        def __init__(self) -> None:
            self.created = 0

        def create(self, *, seed: int, evaluation_map: object) -> Environment:
            del seed, evaluation_map
            self.created += 1
            return Environment()

    class Pipeline:
        def set_evaluation_map(self, map_spec: object) -> None:
            del map_spec

        def transform_observation(self, observation: object) -> object:
            return observation

    class Policy:
        def act(self, observation: object, *, deterministic: bool = False) -> float:
            del observation
            assert deterministic
            return 0.0

    monkeypatch.setattr("tmrl.trackmania.evaluation.BoundaryGeometry", Geometry)
    factory = EnvironmentFactory()
    map_spec = SimpleNamespace(
        id="test", map_path="map", geometry_path="geometry", expected_map_uid="uid"
    )
    suite = SimpleNamespace(maps=(map_spec,), trials_per_map=3)

    metrics = TrackmaniaEvaluator(suite, factory, Pipeline()).evaluate(Policy())

    assert metrics["eval/finish_rate"] == 1.0
    assert factory.created == 1


def test_trackmania_evaluator_uses_elapsed_time_when_plugin_reports_zero() -> None:
    class Environment:
        def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
            del seed
            return 0.0, {}

        def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, str | float]]:
            del action
            return 1.0, 1.0, True, False, {"termination_reason": "finished", "race_time_ms": 0.0}

        def close(self) -> None:
            return None

    class EnvironmentFactory:
        def create(self, *, seed: int) -> Environment:
            del seed
            return Environment()

    class Pipeline:
        def transform_observation(self, observation: object) -> object:
            return observation

    class Policy:
        def act(self, observation: object, *, deterministic: bool = False) -> float:
            del observation
            assert deterministic
            return 0.0

    suite = SimpleNamespace(seeds=(1,), episodes_per_seed=1)
    metrics = TrackmaniaEvaluator(suite, EnvironmentFactory(), Pipeline()).evaluate(Policy())

    assert metrics["eval/finish_time_s"] > 0.0


def test_study_runner_records_success_and_failure(tmp_path: Path) -> None:
    study = StudySpec(
        name="release", max_trials=2, evaluation_suite="suite", search_space={"lr": [1, 2]}
    )
    runner = StudyRunner(GridStrategy(), StudyLedger(tmp_path))
    scores = runner.run(study, lambda patch: float(patch["lr"]))
    events = (tmp_path / "study.jsonl").read_text(encoding="utf-8").splitlines()
    assert scores == [1.0, 2.0]
    assert all("timestamp_utc" in json.loads(line) for line in events)
