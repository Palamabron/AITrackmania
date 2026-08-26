"""Release contracts for observability, resume, and the optional game adapter."""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import tomllib
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trackmaniarl.core.builtins import JsonlRunLogger, TorchCheckpointCodec
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.observability.trackers import WandbTracker
from trackmaniarl.project.scaffold import create_project
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironment,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.reward import RewardResult, TrajectoryReward
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
  learner: {class_path: trackmaniarl.core.builtins:SmokeLearner}
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
    definitions: list[tuple[str, dict[str, object]]] = []
    finished: list[int] = []

    class FakeRun:
        url = ""

        @staticmethod
        def define_metric(name: str, **kwargs: object) -> None:
            definitions.append((name, kwargs))

        @staticmethod
        def log(values: dict[str, object]) -> None:
            logged.append(values)

        @staticmethod
        def finish(*, exit_code: int) -> None:
            finished.append(exit_code)

    class FakeWandb:
        class Settings:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

        @staticmethod
        def init(**kwargs: object) -> FakeRun:
            del kwargs
            return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    tracker = WandbTracker("project", run_dir=str(tmp_path))
    tracker.log("train/episode", {"index": 1}, step=10)
    tracker.log("train/episode", {"index": 2}, step=10)
    tracker.close()

    assert [event["episode/index"] for event in logged] == [1, 2]
    assert [event["env/episode"] for event in logged] == [1, 2]
    assert all("trainer/update" not in event for event in logged)
    assert ("episode/*", {"step_metric": "env/episode"}) in definitions
    assert finished == [0]


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
            nearest_distance_m=0.0,
            accepted_progress_delta_m=0.0,
            window_progress_m=0.0,
            steps_since_progress=0,
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


def test_environment_recovers_reset_timeout_with_finish_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Client:
        def close(self) -> None:
            calls.append("close")

    class Controller:
        def confirm_finish(self) -> None:
            calls.append("enter")

        def reset(self) -> None:
            calls.append("delete")

    environment = object.__new__(OpenPlanetEnvironment)
    environment.client = Client()
    environment.controller = Controller()
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", lambda _: None)

    environment._recover_reset_timeout()

    assert calls == ["close", "enter", "delete"]


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


def test_off_track_transition_cannot_collect_progress() -> None:
    trajectory = np.stack(
        (
            np.arange(101, dtype=np.float32),
            np.zeros(101, dtype=np.float32),
            np.zeros(101, dtype=np.float32),
        ),
        axis=1,
    )
    reward = TrajectoryReward(
        trajectory,
        crash_distance=10.0,
        terminal_failure_penalty=0.0,
        time_penalty_per_second=0.0,
        potential_progress_weight=0.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        np.array([50, 20, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=50.0,
    )

    assert result.reason == "off_track"
    assert result.progress_reward == 0.0
    assert result.potential_progress == 0.0
    assert reward.progress_m == 0.0


def _task_aligned_reward(trajectory: np.ndarray) -> TrajectoryReward:
    return TrajectoryReward(
        trajectory,
        crash_distance=15.0,
        no_progress_steps=2_000,
        slow_progress_window_steps=2_000,
        minimum_finish_steps=1,
        terminal_failure_penalty=0.0,
        time_penalty_per_second=0.25,
        progress_reward_full_lap=40.0,
        finish_reward=60.0,
        potential_progress_weight=0.0,
        projected_velocity_scale=0.0,
        projected_speed_bonus_scale=0.0,
        time_attack_target_s=55.0,
        time_attack_linear_scale=2.0,
        reward_gamma=0.9994,
    )


def test_task_aligned_reward_strictly_ranks_finish_times() -> None:
    trajectory = np.stack(
        (
            np.arange(1_101, dtype=np.float32),
            np.zeros(1_101, dtype=np.float32),
            np.zeros(1_101, dtype=np.float32),
        ),
        axis=1,
    )

    def finish_return(finish_time_s: float) -> tuple[float, float]:
        reward = _task_aligned_reward(trajectory)
        reward.reset(trajectory[0], velocity=np.zeros(3), race_time_ms=0.0)
        steps = round(finish_time_s * 20.0)
        total = 0.0
        discounted = 0.0
        discount = 1.0
        for step in range(1, steps + 1):
            index = round(step * (len(trajectory) - 1) / steps)
            result = reward.step(
                trajectory[index],
                finish_ui_active=step == steps,
                velocity=np.zeros(3),
                race_time_ms=step * 50.0,
            )
            assert result.terminated == (step == steps)
            total += result.reward
            discounted += discount * result.reward
            discount *= 0.9994
        return total, discounted

    finish_times = (35.0, 36.0, 37.0, 40.0, 50.0, 55.0)
    returns = [finish_return(finish_time) for finish_time in finish_times]

    assert [value[0] for value in returns] == pytest.approx(
        [131.25, 129.0, 126.75, 120.0, 97.5, 86.25]
    )
    assert all(left[0] > right[0] for left, right in pairwise(returns))
    assert all(left[1] > right[1] for left, right in pairwise(returns))


def test_finished_lap_receives_the_complete_progress_reward() -> None:
    trajectory = np.stack(
        (
            np.arange(201, dtype=np.float32),
            np.zeros(201, dtype=np.float32),
            np.zeros(201, dtype=np.float32),
        ),
        axis=1,
    )
    reward = _task_aligned_reward(trajectory)
    reward.max_projected_speed_mps = 1_000.0
    reward.reset(trajectory[0], velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        trajectory[-2],
        finish_ui_active=True,
        velocity=np.zeros(3),
        race_time_ms=36_000.0,
    )

    assert result.reason == "finished"
    assert result.progress_reward == pytest.approx(40.0)
    assert reward.progress_pct == 100.0


def test_progress_jump_requires_elapsed_time_and_observed_motion() -> None:
    trajectory = np.stack(
        (
            np.arange(201, dtype=np.float32),
            np.zeros(201, dtype=np.float32),
            np.zeros(201, dtype=np.float32),
        ),
        axis=1,
    )
    reward = TrajectoryReward(
        trajectory,
        crash_distance=200.0,
        nearest_forward_points=200,
        no_progress_steps=2_000,
        slow_progress_window_steps=2_000,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=40.0,
        potential_progress_weight=0.0,
        max_projected_speed_mps=100.0,
        max_time_delta_s=1.0,
    )
    reward.reset(trajectory[0], race_time_ms=0.0)

    first = reward.step(
        trajectory[100],
        finish_ui_active=False,
        race_time_ms=50.0,
    )

    assert reward.progress_m == pytest.approx(5.0)
    assert first.accepted_progress_delta_m == pytest.approx(5.0)

    reward.step(
        trajectory[100],
        finish_ui_active=False,
        race_time_ms=1_050.0,
    )

    assert reward.progress_m == pytest.approx(5.0)


def test_slowest_finish_is_better_than_a_near_complete_failure() -> None:
    trajectory = np.stack(
        (
            np.arange(1_001, dtype=np.float32),
            np.zeros(1_001, dtype=np.float32),
            np.zeros(1_001, dtype=np.float32),
        ),
        axis=1,
    )

    def rollout(*, progress: float, duration_s: float, finish: bool) -> float:
        reward = _task_aligned_reward(trajectory)
        reward.reset(trajectory[0], race_time_ms=0.0)
        steps = round(duration_s * 20.0)
        total = 0.0
        for step in range(1, steps + 1):
            index = round(step * progress * (len(trajectory) - 1) / steps)
            position = trajectory[index]
            if step == steps and not finish:
                position = position + np.asarray([0.0, 20.0, 0.0])
            result = reward.step(
                position,
                finish_ui_active=finish and step == steps,
                race_time_ms=step * 50.0,
            )
            total += result.reward
        return total

    slowest_finish = rollout(progress=1.0, duration_s=55.0, finish=True)
    near_complete_failure = rollout(progress=0.999, duration_s=35.0, finish=False)

    assert slowest_finish == pytest.approx(86.25)
    assert near_complete_failure < 40.0
    assert slowest_finish > near_complete_failure


def test_task_aligned_discount_does_not_reward_delaying_the_same_failure() -> None:
    trajectory = np.stack(
        (
            np.arange(1_001, dtype=np.float32),
            np.zeros(1_001, dtype=np.float32),
            np.zeros(1_001, dtype=np.float32),
        ),
        axis=1,
    )

    def failure_return(failure_time_s: float) -> float:
        reward = _task_aligned_reward(trajectory)
        reward.reset(trajectory[0], velocity=np.zeros(3), race_time_ms=0.0)
        steps = round(failure_time_s * 20.0)
        discounted = 0.0
        discount = 1.0
        for step in range(1, steps + 1):
            if step == steps:
                position = np.array([100.0, 20.0, 0.0], dtype=np.float32)
            else:
                position = trajectory[round(step * 100 / steps)]
            result = reward.step(
                position,
                finish_ui_active=False,
                velocity=np.zeros(3),
                race_time_ms=step * 50.0,
            )
            discounted += discount * result.reward
            discount *= 0.9994
        assert result.reason == "off_track"
        return discounted

    assert failure_return(2.7) > failure_return(10.8)


def test_time_attack_reward_is_bounded_and_ranks_finish_times() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        finish_reward=1.0,
        potential_progress_weight=0.0,
        time_attack_target_s=65.0,
        time_attack_bonus_scale=0.0,
        time_attack_linear_scale=1.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    def finish_at(race_time_ms: float) -> RewardResult:
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
        return reward.step(
            np.array([1, 0, 0]),
            finish_ui_active=True,
            velocity=np.zeros(3),
            race_time_ms=race_time_ms,
        )

    slow, target, fast = finish_at(59_400.0), finish_at(36_000.0), finish_at(35_000.0)

    assert (
        fast.time_attack_terminal_reward
        > target.time_attack_terminal_reward
        > (slow.time_attack_terminal_reward)
    )
    assert slow.reward == pytest.approx(6.6)
    assert target.reward == 30.0
    assert fast.reward == 31.0
    assert slow.terminal_reward == target.terminal_reward == fast.terminal_reward == 1.0


def test_time_attack_linear_reward_keeps_ranking_slower_than_target_finishes() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        finish_reward=30.0,
        potential_progress_weight=0.0,
        time_attack_target_s=36.0,
        time_attack_bonus_scale=0.0,
        time_attack_linear_scale=1.0,
    )

    def finish_at(race_time_ms: float) -> RewardResult:
        reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)
        return reward.step(
            np.array([1, 0, 0]),
            finish_ui_active=True,
            velocity=np.zeros(3),
            race_time_ms=race_time_ms,
        )

    fast, target, slow = finish_at(35_000.0), finish_at(36_000.0), finish_at(59_000.0)

    assert fast.reward == 31.0
    assert target.reward == 30.0
    assert slow.reward == 7.0
    assert fast.terminal_reward == target.terminal_reward == slow.terminal_reward == 30.0


def test_time_attack_penalty_cannot_make_a_finish_terminal_reward_negative() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        minimum_finish_steps=1,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        finish_reward=60.0,
        potential_progress_weight=0.0,
        time_attack_target_s=55.0,
        time_attack_linear_scale=2.0,
    )
    reward.reset(np.array([0, 0, 0]), race_time_ms=0.0)

    result = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=True,
        race_time_ms=165_000.0,
    )

    assert result.reason == "finished"
    assert result.time_attack_terminal_reward == -60.0
    assert result.terminal_reward == 60.0
    assert result.reward == 0.0


def test_projected_velocity_reward_clips_telemetry_outliers() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        no_progress_steps=100,
        slow_progress_window_steps=100,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        potential_progress_weight=0.0,
        max_projected_speed_mps=100.0,
        velocity_to_mps_scale=1.0,
        projected_velocity_scale=1.0,
    )

    def reward_for(velocity_x: float) -> RewardResult:
        reward.reset(np.array([0, 0, 0]), race_time_ms=0.0)
        return reward.step(
            np.array([0, 0, 0]),
            finish_ui_active=False,
            velocity=np.array([velocity_x, 0, 0]),
            race_time_ms=1_000.0,
        )

    forward, reverse = reward_for(1_000.0), reward_for(-1_000.0)

    assert forward.projected_velocity_mps == 100.0
    assert forward.projected_velocity_reward == 100.0
    assert reverse.projected_velocity_mps == -100.0
    assert reverse.projected_velocity_reward == -100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crash_distance": float("inf")},
        {"maximum_race_time_s": float("inf")},
        {"time_attack_target_s": float("inf")},
        {"terminal_failure_penalty": float("nan")},
    ],
)
def test_trajectory_reward_rejects_non_finite_limits(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        TrajectoryReward(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crash_distance": float("inf")},
        {"maximum_race_time_s": float("inf")},
        {"time_attack_target_s": float("inf")},
        {"terminal_failure_penalty": float("inf")},
    ],
)
def test_environment_config_rejects_non_finite_reward_limits(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        TrackmaniaEnvironmentConfig(trajectory_path=Path("trajectory.npy"), **kwargs)


def test_slow_progress_threshold_tolerates_geometry_rounding() -> None:
    def second_step(distance_m: float) -> RewardResult:
        reward = TrajectoryReward(
            np.array([[0, 0, 0], [distance_m, 0, 0], [10, 0, 0]], dtype=np.float32),
            no_progress_steps=100,
            slow_progress_window_steps=2,
            minimum_progress_per_window_m=5.0,
            time_penalty_per_second=0.0,
            progress_reward_full_lap=0.0,
            potential_progress_weight=0.0,
        )
        reward.reset(np.array([0, 0, 0]), race_time_ms=0.0)
        reward.step(
            np.array([0, 0, 0]),
            finish_ui_active=False,
            race_time_ms=500.0,
        )
        return reward.step(
            np.array([distance_m, 0, 0]),
            finish_ui_active=False,
            race_time_ms=1_000.0,
        )

    rounded = second_step(4.996)
    genuinely_slow = second_step(4.9)

    assert rounded.reason is None
    assert rounded.window_progress_m == pytest.approx(4.996)
    assert genuinely_slow.reason == "slow_progress"


def test_race_time_limit_is_a_penalized_terminal_transition() -> None:
    reward = TrajectoryReward(
        np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32),
        no_progress_steps=100,
        slow_progress_window_steps=100,
        terminal_failure_penalty=7.0,
        time_penalty_per_second=0.0,
        progress_reward_full_lap=0.0,
        finish_reward=0.0,
        potential_progress_weight=0.0,
        maximum_race_time_s=1.0,
    )
    reward.reset(np.array([0, 0, 0]), velocity=np.zeros(3), race_time_ms=0.0)

    result = reward.step(
        np.array([1, 0, 0]),
        finish_ui_active=False,
        velocity=np.zeros(3),
        race_time_ms=1_000.0,
    )

    assert result.terminated
    assert result.reason == "time_limit"
    assert result.terminal_reward == -7.0


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


def test_starter_template_partitions_torch_sources_by_platform(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent")
    pyproject = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))

    torch_sources = pyproject["tool"]["uv"]["sources"]["torch"]
    assert torch_sources == [
        {
            "index": "pytorch-cuda",
            "marker": "sys_platform == 'win32' or sys_platform == 'linux'",
        },
        {
            "index": "pytorch-cpu",
            "marker": "sys_platform != 'win32' and sys_platform != 'linux'",
        },
    ]
    assert pyproject["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        },
        {
            "name": "pytorch-cuda",
            "url": "https://download.pytorch.org/whl/cu128",
            "explicit": True,
        },
    ]


def test_trackmania_template_contains_first_party_components(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent", template="trackmania")
    config = (target / "run.yaml").read_text(encoding="utf-8")
    pyproject = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert "OpenPlanetEnvironmentFactory" in config
    assert "control_backend: gamepad" in config
    assert "action_repeat_frames: 1" in config
    assert "decision_interval_ms: 50.0" in config
    assert "demonstration_control_aggregation: true" in config
    assert "TrackmaniaEvaluator" in config
    assert "WandbTracker" not in config
    assert (target / "assets" / "trajectory.csv").is_file()
    assert (target / "maps").is_dir()
    plugin = target / "openplanet" / "SAC_GetData-2.4.0-reference.as"
    assert plugin.is_file()
    plugin_source = plugin.read_text(encoding="utf-8")
    assert 'const string PROTOCOL_VERSION = "2"' in plugin_source
    assert "if (api is null || vis is null) { yield(); continue; }" in plugin_source
    assert "vis is null ?" not in plugin_source
    plugin_info = tomllib.loads(
        (target / "openplanet" / "info.reference.toml").read_text(encoding="utf-8")
    )
    assert plugin_info["meta"]["name"] == "TrackmaniaRL Connect"
    assert plugin_info["meta"]["version"] == "2.4.0"
    assert plugin_info["meta"]["siteid"] == 421
    assert RunSpec.from_yaml(target / "run.yaml").evaluation is not None
    assert pyproject["tool"]["poe"]["tasks"]["record-left"]
    assert pyproject["tool"]["uv"]["sources"]["torch"]
    assert pyproject["tool"]["uv"]["sources"]["vgamepad"]
    assert "TRACKMANIARL_DISTRIBUTED_TOKEN=" in (target / ".env-example").read_text()
    assert "WANDB_API_KEY" not in (target / ".env-example").read_text()
    assert ".env" in (target / ".gitignore").read_text()
    assert not (target / "run.py").exists()
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "Plugin Manager" in readme
    assert "SAC_GetData" in readme
    assert 'uv add "trackmaniarl[trackmania,algorithms,distributed,wandb]"' in readme
    dependency = pyproject["project"]["dependencies"][0]
    assert "wandb" not in dependency


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


def test_trackmania_evaluator_passes_raw_observation_to_opt_in_policy() -> None:
    raw = np.arange(33, dtype=np.float32)

    class Environment:
        def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
            del seed
            return raw.copy(), {}

        def step(
            self, action: object
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, str | float]]:
            del action
            return raw.copy(), 0.0, True, False, {"termination_reason": "no_progress"}

        def close(self) -> None:
            return None

    class EnvironmentFactory:
        def create(self, *, seed: int) -> Environment:
            del seed
            return Environment()

    class Pipeline:
        def transform_observation(self, observation: object) -> str:
            del observation
            return "prepared"

    class Policy:
        requires_raw_observation = True

        def act(self, observation: object, *, deterministic: bool = False) -> float:
            assert deterministic
            assert np.array_equal(observation, raw)
            return 0.0

    suite = SimpleNamespace(seeds=(1,), episodes_per_seed=1)
    metrics = TrackmaniaEvaluator(suite, EnvironmentFactory(), Pipeline()).evaluate(Policy())

    assert metrics["eval/finish_rate"] == 0.0
