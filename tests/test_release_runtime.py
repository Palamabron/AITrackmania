"""Release contracts for observability, resume, and the optional game adapter."""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tmrl.core.builtins import JsonlRunLogger
from tmrl.experiments.orchestration import GridStrategy, StudyLedger, StudyRunner, StudySpec
from tmrl.project.scaffold import create_project
from tmrl.trackmania.assets import record_trajectory
from tmrl.trackmania.environment import OpenPlanetEnvironmentFactory
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
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32), minimum_finish_steps=1
    )
    assert reward.step(np.array([1, 0, 0]), finish_ui_active=False).reward > 0
    assert reward.step(np.array([2, 0, 0]), finish_ui_active=False).reason is None
    assert reward.step(np.array([2, 0, 0]), finish_ui_active=True).reason == "finished"
    reward.reset()
    assert reward.step(np.array([100, 0, 0]), finish_ui_active=False).reason == "off_track"


def test_trajectory_reward_has_dense_progress_signal_and_stall_termination() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        no_progress_steps=3,
        slow_progress_window_steps=10,
        minimum_finish_steps=1,
    )
    assert reward.step(np.array([1, 0, 0]), finish_ui_active=False, speed_mps=50).reward > 100.0
    reward.reset()
    assert reward.step(np.array([0, 0, 0]), finish_ui_active=False).reason is None
    assert reward.step(np.array([0, 0, 0]), finish_ui_active=False).reason is None
    assert reward.step(np.array([0, 0, 0]), finish_ui_active=False).reason == "no_progress"


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

    result = reward.step(np.array([10, 0, 0.9]), finish_ui_active=False)

    assert result.reward == pytest.approx(40.0)
    assert reward._index == 1


def test_trajectory_recorder_writes_portable_csv(tmp_path: Path) -> None:
    class Client:
        def read(self) -> TelemetryFrame:
            return TelemetryFrame(np.arange(33, dtype=np.float32))

    path = record_trajectory(tmp_path / "trajectory.csv", Client(), samples=2)  # type: ignore[arg-type]
    points = np.loadtxt(path, delimiter=",")
    assert points.shape == (2, 3)
    assert np.array_equal(points[0], np.asarray(DEFAULT_POSITION_INDICES, dtype=np.float32))


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


def test_trackmania_template_contains_first_party_components(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent", template="trackmania")
    config = (target / "run.yaml").read_text(encoding="utf-8")
    assert "OpenPlanetEnvironmentFactory" in config
    assert "TrackmaniaEvaluator" in config
    assert (target / "assets" / "trajectory.csv").is_file()
    plugin = target / "openplanet" / "TMRL_GrabData_IQN.as"
    assert plugin.is_file()
    assert 'const string PROTOCOL_VERSION = "2"' in plugin.read_text(encoding="utf-8")


def test_generated_project_uses_the_current_checkout_before_first_publish(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent")
    pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "tmrl @ file:///" in pyproject


def test_trackmania_evaluator_runs_every_declared_seed_and_episode() -> None:
    class Environment:
        def reset(self, *, seed: int | None = None):
            del seed
            return 0.0, {}

        def step(self, action: object):
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


def test_study_runner_records_success_and_failure(tmp_path: Path) -> None:
    study = StudySpec(
        name="release", max_trials=2, evaluation_suite="suite", search_space={"lr": [1, 2]}
    )
    runner = StudyRunner(GridStrategy(), StudyLedger(tmp_path))
    scores = runner.run(study, lambda patch: float(patch["lr"]))
    events = (tmp_path / "study.jsonl").read_text(encoding="utf-8").splitlines()
    assert scores == [1.0, 2.0]
    assert all("timestamp_utc" in json.loads(line) for line in events)
