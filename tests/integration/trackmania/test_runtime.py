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

from trackmaniarl.core.builtins import JsonlRunLogger, TorchCheckpointCodec
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.observability.trackers import WandbTracker
from trackmaniarl.project.scaffold import create_project
from trackmaniarl.trackmania.environment import OpenPlanetEnvironment
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.telemetry import (
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


def test_distributed_token_requires_at_least_32_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl import cli

    config = tmp_path / "run.yaml"
    config.write_text(
        """api_version: \"2.0\"
run_id: token-test
components:
  learner: {class_path: trackmaniarl.core.builtins:NullLearner}
  replay_store: {class_path: trackmaniarl.core.replay:InMemoryReplayStore}
  sampler: {class_path: trackmaniarl.core.replay:UniformSampler}
  feature_pipeline: {class_path: trackmaniarl.core.builtins:IdentityFeaturePipeline}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRACKMANIARL_DISTRIBUTED_TOKEN", "short")

    with pytest.raises(ValueError, match="at least 32 characters"):
        cli._required_token(config)

    token = "a" * 32
    monkeypatch.setenv("TRACKMANIARL_DISTRIBUTED_TOKEN", token)
    assert cli._required_token(config) == token


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


def test_torch_checkpoint_concurrent_loads_use_distinct_temporary_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import trackmaniarl.core.builtins as builtins

    path = tmp_path / "checkpoint.pt"
    codec = TorchCheckpointCodec()
    state = {"tensor": torch.ones(1_000_000, dtype=torch.float32)}
    codec.save(state, path)
    original = builtins._load_torch_checkpoint
    barrier = threading.Barrier(2)
    temporary_paths: list[Path] = []
    failures: list[Exception] = []

    def delayed_load(temporary: Path) -> dict[str, object]:
        temporary_paths.append(temporary)
        barrier.wait(timeout=10)
        return dict(original(temporary))

    def load() -> None:
        try:
            restored = codec.load(path)
            assert torch.equal(restored["tensor"], state["tensor"])
        except Exception as exc:
            failures.append(exc)

    monkeypatch.setattr(builtins, "_load_torch_checkpoint", delayed_load)
    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert not any(thread.is_alive() for thread in threads)
    assert len(set(temporary_paths)) == 2


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


def test_environment_step_reports_applied_control_and_race_time_delta() -> None:
    from trackmaniarl.trackmania.actions import build_brake_tap_action_table
    from trackmaniarl.trackmania.control import RecordingController

    class Client:
        def __init__(self) -> None:
            self.race_times_ms = iter((105.0, 112.0, 121.0))

        def read(self) -> TelemetryFrame:
            values = np.zeros(33, dtype=np.float32)
            values[3] = next(self.race_times_ms)
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
            pace_reward=0.0,
            reference_time_s=0.0,
            time_debt_s=0.0,
        )

    environment = object.__new__(OpenPlanetEnvironment)
    environment.config = SimpleNamespace(
        action_repeat_frames=2,
        decision_interval_ms=20.0,
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
    assert info["step_race_time_ms"] == pytest.approx(21.0)
    assert info["decision_interval_error_ms"] == pytest.approx(1.0)
    assert info["race_time_ms"] == pytest.approx(121.0)
    assert environment._last_race_time_ms == pytest.approx(121.0)


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


def test_trackmania_template_contains_first_party_components(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent", template="trackmania")
    config = (target / "run.yaml").read_text(encoding="utf-8")
    pyproject = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert "OpenPlanetEnvironmentFactory" in config
    assert "control_backend: gamepad" in config
    assert "TrackmaniaEvaluator" in config
    assert "WandbTracker" not in config
    assert (target / "assets" / "trajectory.csv").is_file()
    assert (target / "maps").is_dir()
    plugin = target / "openplanet" / "TrackmaniaRL_GrabData_IQN.as"
    assert plugin.is_file()
    assert 'const string PROTOCOL_VERSION = "2"' in plugin.read_text(encoding="utf-8")
    assert RunSpec.from_yaml(target / "run.yaml").evaluation is not None
    assert pyproject["tool"]["poe"]["tasks"]["record-left"]
    assert pyproject["tool"]["uv"]["sources"]["torch"]
    assert "TRACKMANIARL_DISTRIBUTED_TOKEN=" in (target / ".env-example").read_text()
    assert ".env" in (target / ".gitignore").read_text()


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
