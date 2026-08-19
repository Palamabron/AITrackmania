from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tmrl.core.builtins import IdentityFeaturePipeline
from tmrl.core.data import BatchRequest, Trajectory, Transition
from tmrl.core.offline import OfflineBufferLoader, save_trajectory
from tmrl.core.replay import InMemoryReplayStore, UniformSampler
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from tmrl.trackmania.environment import TrackmaniaEnvironmentConfig
from tmrl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from tmrl.trackmania.ghost import (
    GHOST_DT_MS,
    GbxExtractRequest,
    GhostFrame,
    decode_ghost_packet,
    encode_ghost_packet,
    extract_gbx_demo,
    inspect_gbx,
)
from tmrl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


class _IdentityPipeline:
    def reset_episode(self) -> None:
        return

    def transform_observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).copy()

    def collate(self, transitions: list[Transition]) -> list[Transition]:
        return transitions


class _GhostClient:
    def __init__(self, frames: list[GhostFrame]) -> None:
        self.frames = list(frames)
        self.cursor = 0

    def read(self) -> GhostFrame:
        frame = self.frames[min(self.cursor, len(self.frames) - 1)]
        self.cursor += 1
        return frame

    def close(self) -> None:
        return


def _geometry(tmp_path: Path) -> BoundaryGeometry:
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(101)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    map_path = tmp_path / "map.Map.Gbx"
    map_path.write_bytes(b"GBX")
    asset = build_geometry_asset(
        tmp_path / "geometry.npz",
        tmp_path / "left.npy",
        tmp_path / "right.npy",
        map_uid="test-map",
        map_path=map_path,
        spacing_m=1.0,
        lookahead_points=0,
    )
    return BoundaryGeometry(asset, expected_map_uid="test-map")


def _config(geometry: BoundaryGeometry) -> TrackmaniaEnvironmentConfig:
    return TrackmaniaEnvironmentConfig(
        geometry_path=geometry.path,
        expected_map_uid=geometry.map_uid,
        action_repeat_frames=2,
        start_timeout_s=15.0,
        start_poll_s=0.0,
        velocity_to_mps_scale=1.0,
        minimum_finish_steps=50,
        no_progress_steps=100,
        slow_progress_window_steps=80,
        minimum_progress_per_window_m=1.0,
    )


def _values(step: int, *, count: int, finished: bool, steer: float) -> np.ndarray:
    values = np.zeros(DEFAULT_TELEMETRY_FIELD_COUNT, dtype=np.float32)
    values[2] = float(finished)
    values[3] = float(step) * GHOST_DT_MS
    values[4] = 100.0 * float(step) / float(count)
    values[7] = 40.0
    values[12] = 1.0
    values[30] = steer
    values[31] = 1.0
    return values


def _ghost_lap(count: int = 61) -> list[GhostFrame]:
    idle = _values(0, count=count, finished=False, steer=0.0)
    idle[3] = 0.0
    frames = [GhostFrame(game_time_ms=0.0, values=idle)]
    steers = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    for step, steer in enumerate(steers, start=1):
        frames.append(
            GhostFrame(
                game_time_ms=float(step) * GHOST_DT_MS,
                values=_values(step, count=count, finished=step == count, steer=float(steer)),
            )
        )
    return frames


def _fake_gbx(path: Path, uid: str | None = None) -> Path:
    payload = b"GBX\x06" + b"\x00" * 32
    if uid is not None:
        payload += uid.encode("ascii")
    path.write_bytes(payload)
    return path


def test_ghost_datagram_round_trip_over_udp() -> None:
    values = _values(3, count=3, finished=False, steer=-0.5)
    payload = encode_ghost_packet(150.0, values)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(1.0)
        sock.sendto(payload, sock.getsockname())
        data, _ = sock.recvfrom(2048)
    frame = decode_ghost_packet(data)
    assert frame.game_time_ms == 150.0
    assert frame.values[30] == pytest.approx(-0.5)
    assert frame.race_time_ms == pytest.approx(150.0)


def test_inspect_gbx_reads_header_and_optional_map_uid(tmp_path: Path) -> None:
    path = _fake_gbx(tmp_path / "lap.Replay.Gbx", "oqIJ5rQDRrNwLPTh9H2p_W4tLof")
    meta = inspect_gbx(path)
    assert meta.map_uid == "oqIJ5rQDRrNwLPTh9H2p_W4tLof"
    assert len(meta.sha256) == 64
    (tmp_path / "notes.txt").write_text("not a ghost", encoding="utf-8")
    with pytest.raises(ValueError, match="not a GBX"):
        inspect_gbx(tmp_path / "notes.txt")


def test_extract_gbx_demo_pairs_action_with_same_game_time_sample(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    gbx = _fake_gbx(tmp_path / "ghost.Replay.Gbx")
    output = tmp_path / "lap.pkl"
    path = extract_gbx_demo(
        GbxExtractRequest(
            gbx_path=gbx,
            output=output,
            pipeline=_IdentityPipeline(),
            config=_config(geometry),
            geometry=geometry,
            max_duration_s=1.0,
        ),
        _GhostClient(_ghost_lap()),
        status=lambda _message: None,
    )
    store = InMemoryReplayStore(capacity=128)
    imported = store.load_demonstrations(path)
    batch = UniformSampler(IdentityFeaturePipeline(), seed=0).sample(
        store, BatchRequest(batch_size=8)
    )
    _, table = build_brake_tap_action_table()
    first_steer = float(_ghost_lap()[1].values[30])
    expected = continuous_control_to_discrete_index(
        np.asarray([1.0, 0.0, first_steer], dtype=np.float32), table
    )
    loaded = store.get(store.available_ids())
    assert path == output
    assert imported == 60
    assert loaded[0].action == expected
    assert loaded[0].info["is_demo"] is True
    assert loaded[-1].terminated
    assert len(batch.transition_ids) == 8


def test_extract_gbx_demo_rejects_map_uid_mismatch(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    gbx = _fake_gbx(tmp_path / "ghost.Replay.Gbx", "aaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(ValueError, match="map UID"):
        extract_gbx_demo(
            GbxExtractRequest(
                gbx_path=gbx,
                output=tmp_path / "lap.pkl",
                pipeline=_IdentityPipeline(),
                config=_config(geometry),
                geometry=geometry,
            ),
            _GhostClient(_ghost_lap()),
        )


def test_load_demonstrations_protects_expert_data_from_online_overwrite(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    path = extract_gbx_demo(
        GbxExtractRequest(
            gbx_path=_fake_gbx(tmp_path / "ghost.Replay.Gbx"),
            output=tmp_path / "lap.pkl",
            pipeline=_IdentityPipeline(),
            config=_config(geometry),
            geometry=geometry,
            max_duration_s=1.0,
        ),
        _GhostClient(_ghost_lap()),
        status=lambda _message: None,
    )
    store = InMemoryReplayStore(capacity=128)
    OfflineBufferLoader(store).load_demonstrations(path)
    example = store.get([store.available_ids()[0]])[0]
    for step in range(80):
        store.append(
            Transition(
                observation=np.asarray(example.observation, dtype=np.float32).copy(),
                action=int(example.action),
                reward=0.0,
                next_observation=np.asarray(example.next_observation, dtype=np.float32).copy(),
                terminated=step == 79,
                truncated=False,
                episode_id="online",
                step=step,
            )
        )
    flags = store.demo_flags(store.available_ids())
    assert sum(flags) == 60


def test_bundled_openplanet_plugin_declares_ghost_replay_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    plugin = (root / "tmrl/project/openplanet/TMRL_GrabData_IQN.as").read_text(encoding="utf-8")
    agent = (root / "my-trackmania-agent/openplanet/TMRL_GrabData_IQN.as").read_text(
        encoding="utf-8"
    )
    assert plugin == agent
    assert "const uint GHOST_PORT = 9002" in plugin
    assert "const float GHOST_DT_MS = 50.0f" in plugin
    assert "AppendGhostTelemetry" in plugin


def test_save_trajectory_rejects_hdf5(tmp_path: Path) -> None:
    trajectory = Trajectory(
        episode_id="demo",
        transitions=[
            Transition(
                observation=0.0,
                action=0,
                reward=1.0,
                next_observation=1.0,
                terminated=True,
                truncated=False,
            )
        ],
    )
    with pytest.raises(ValueError, match="HDF5"):
        save_trajectory(tmp_path / "lap.h5", trajectory)
