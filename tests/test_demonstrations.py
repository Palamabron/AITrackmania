"""Demonstration files must round-trip and seed replay without update credit."""

from __future__ import annotations

import socket
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tmrl.core.builtins import TorchCheckpointCodec
from tmrl.core.data import Transition
from tmrl.core.replay import InMemoryReplayStore, UniformSampler
from tmrl.core.runtime import ResolvedRun
from tmrl.core.spec import RunSpec
from tmrl.distributed.codec import WireCodec
from tmrl.distributed.coordinator import Coordinator
from tmrl.distributed.demos import (
    load_demonstration_transitions,
    resolve_demo_files,
    save_demonstration,
)
from tmrl.trackmania.actions import build_brake_tap_action_table
from tmrl.trackmania.demos import human_action


class _Pipeline:
    def transform_observation(self, observation: Any) -> Any:
        return observation

    def collate(self, transitions: list[Transition]) -> Mapping[str, Any]:
        return {"reward": np.asarray([item.reward for item in transitions])}


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        del step
        self.events.append((event, dict(payload)))

    def close(self) -> None:
        return


class _IdleLearner:
    def setup(self, context: Mapping[str, Any]) -> None:
        del context

    def update(self, batch: Any) -> Mapping[str, float]:
        raise AssertionError("demo loading must not trigger learner updates")

    def policy(self) -> Any:
        return _Policy()

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        del state


class _Policy:
    def act(self, observation: Any, *, deterministic: bool = False) -> int:
        del observation, deterministic
        return 0

    def export_state(self) -> Mapping[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        del state


def _demo_transition(step: int, *, terminal: bool = False) -> Transition:
    return Transition(
        observation=np.full(3, float(step), dtype=np.float32),
        action=step,
        reward=1.0,
        next_observation=np.full(3, float(step + 1), dtype=np.float32),
        terminated=terminal,
        truncated=False,
        info={"race_time_ms": 1_000.0},
        episode_id="demo/file/0000",
        step=step,
    )


def test_demo_file_round_trips_and_forces_the_demo_marker(tmp_path: Path) -> None:
    transitions = [_demo_transition(0), _demo_transition(1, terminal=True)]
    path = save_demonstration(
        tmp_path / "lap.tmdemo", transitions, {"finish_time_s": 39.5, "steps": 2}
    )

    loaded = load_demonstration_transitions(path)

    assert [item.step for item in loaded] == [0, 1]
    assert all(item.info == {"is_demo": True} for item in loaded)
    assert loaded[1].terminated
    np.testing.assert_array_equal(loaded[0].observation, transitions[0].observation)


def test_load_rejects_files_without_the_demo_format(tmp_path: Path) -> None:
    path = tmp_path / "broken.tmdemo"
    path.write_bytes(WireCodec(1 << 20).encode({"format": "other"}))

    with pytest.raises(ValueError, match="tmrl-demo-v1"):
        load_demonstration_transitions(path)


def test_resolve_demo_files_expands_directories_and_rejects_missing_paths(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "demos"
    directory.mkdir()
    second = directory / "b.tmdemo"
    first = directory / "a.tmdemo"
    second.write_bytes(b"")
    first.write_bytes(b"")
    (directory / "ignored.txt").write_bytes(b"")

    assert resolve_demo_files([directory]) == [first, second]
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_demo_files([tmp_path / "missing"])
    with pytest.raises(FileNotFoundError, match=r"no \.tmdemo"):
        resolve_demo_files([])


def test_human_action_maps_live_inputs_to_the_discrete_table() -> None:
    _, table = build_brake_tap_action_table()
    values = np.zeros(33, dtype=np.float32)
    values[30] = -1.0
    values[31] = 1.0
    values[32] = 0.0

    action = human_action(values, table)

    np.testing.assert_array_equal(table[action], np.asarray([1.0, 0.0, -1.0], dtype=np.float32))


def test_coordinator_seeds_replay_from_demos_without_update_credit(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    demo_path = save_demonstration(
        tmp_path / "lap.tmdemo",
        [_demo_transition(0), _demo_transition(1, terminal=True)],
        {"finish_time_s": 39.5, "steps": 2},
    )
    spec = RunSpec.model_validate(
        {
            "api_version": "1.2",
            "run_id": "demo-seed",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "training": {"total_transitions": 1, "warmup_transitions": 100},
        }
    )
    pipeline = _Pipeline()
    logger = _RecordingLogger()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "demo-seed",
        learner=_IdleLearner(),
        environment_factory=None,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline),
        feature_pipeline=pipeline,
        logger=logger,
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
    )
    stop = threading.Event()
    stop.set()
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{port}",
        token="secret",
        fingerprint="fingerprint",
        external_stop=stop,
        demo_paths=(demo_path,),
    )

    coordinator.run_forever()

    assert len(run.replay_store) == 2
    assert run.replay_store.get([0])[0].info == {"is_demo": True}
    assert coordinator.counters.transitions == 0
    assert coordinator.counters.update_credit == 0.0
    loaded_events = [payload for event, payload in logger.events if event == "demo/loaded"]
    assert loaded_events == [{"files": 1, "transitions": 2, "replay_size": 2}]
