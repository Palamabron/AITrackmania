"""Ghost-replay datagrams aligned to gameTime for lag-free demonstration capture."""

from __future__ import annotations

import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from typing import Protocol

import numpy as np

from tmrl.core.contracts import FeaturePipeline
from tmrl.core.data import Trajectory
from tmrl.core.offline import save_trajectory
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from tmrl.trackmania.demonstrations import (
    CONTROL_INDICES,
    Demonstration,
    demonstration_transitions,
    save_demonstration,
    validate_demonstration,
)
from tmrl.trackmania.environment import TrackmaniaEnvironmentConfig
from tmrl.trackmania.geometry import BoundaryGeometry, file_sha256
from tmrl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT, TelemetryFrame

GHOST_MAGIC = 9002.0
GHOST_PROTOCOL = 3.0
GHOST_DT_MS = 50.0
GHOST_HEADER_FLOATS = 3
GHOST_FIELD_COUNT = DEFAULT_TELEMETRY_FIELD_COUNT
GHOST_PACKET_FLOATS = GHOST_HEADER_FLOATS + GHOST_FIELD_COUNT
GHOST_PACKET_BYTES = GHOST_PACKET_FLOATS * 4
DEFAULT_GHOST_PORT = 9002
_GBX_UID = re.compile(rb"[A-Za-z0-9_\-]{26,28}")


class GhostFrameReader(Protocol):
    def read(self) -> GhostFrame: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GhostFrame:
    game_time_ms: float
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (GHOST_FIELD_COUNT,) or not np.isfinite(self.values).all():
            raise ValueError("ghost frame must contain 33 finite telemetry fields")
        if not np.isfinite(self.game_time_ms):
            raise ValueError("ghost gameTime must be finite")

    @property
    def race_time_ms(self) -> float:
        return float(self.values[3])

    @property
    def finished(self) -> bool:
        return bool(self.values[2])

    def telemetry(self) -> TelemetryFrame:
        return TelemetryFrame(self.values.copy())


@dataclass(frozen=True, slots=True)
class GbxMeta:
    path: Path
    sha256: str
    map_uid: str | None


@dataclass(frozen=True, slots=True)
class GbxExtractRequest:
    gbx_path: Path
    output: Path
    pipeline: FeaturePipeline
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry
    max_duration_s: float = 180.0


def encode_ghost_packet(game_time_ms: float, values: np.ndarray) -> bytes:
    fields = np.asarray(values, dtype=np.float32).reshape(-1)
    if fields.shape != (GHOST_FIELD_COUNT,):
        raise ValueError(f"ghost telemetry must have {GHOST_FIELD_COUNT} fields")
    packet = np.empty(GHOST_PACKET_FLOATS, dtype=np.float32)
    packet[0] = GHOST_MAGIC
    packet[1] = GHOST_PROTOCOL
    packet[2] = game_time_ms
    packet[3:] = fields
    return packet.tobytes()


def decode_ghost_packet(data: bytes) -> GhostFrame:
    if len(data) != GHOST_PACKET_BYTES:
        raise ValueError(f"ghost datagram must be {GHOST_PACKET_BYTES} bytes, got {len(data)}")
    packet = np.frombuffer(data, dtype="<f4")
    if float(packet[0]) != GHOST_MAGIC or float(packet[1]) != GHOST_PROTOCOL:
        raise ValueError("invalid ghost datagram magic or protocol")
    return GhostFrame(
        game_time_ms=float(packet[2]),
        values=np.array(packet[3:], dtype=np.float32, copy=True),
    )


def inspect_gbx(path: str | Path) -> GbxMeta:
    source = Path(path)
    payload = source.read_bytes()
    if not payload.startswith(b"GBX"):
        raise ValueError(f"not a GBX file: {source}")
    match = _GBX_UID.search(payload)
    uid = match.group().decode("ascii") if match is not None else None
    return GbxMeta(path=source.resolve(), sha256=sha256(payload).hexdigest(), map_uid=uid)


class GhostReplayClient:
    """Reads every physics-aligned ghost datagram; queued frames are never dropped."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_GHOST_PORT,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        if port < 1 or timeout_s <= 0:
            raise ValueError("port and timeout_s must be positive")
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self._socket: socket.socket | None = None
        self._buffer = bytearray()

    def connect(self) -> None:
        if self._socket is None:
            connection = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            connection.settimeout(self.timeout_s)
            self._socket = connection

    def read(self) -> GhostFrame:
        reconnect_deadline = monotonic() + self.timeout_s
        while True:
            try:
                return decode_ghost_packet(self._read_connected())
            except ConnectionError as error:
                self.close()
                remaining_s = reconnect_deadline - monotonic()
                if remaining_s <= 0.0:
                    raise ConnectionError(
                        "OpenPlanet ghost replay disconnected and did not accept a new "
                        f"connection within {self.timeout_s:g}s"
                    ) from error
                sleep(min(0.1, remaining_s))

    def _read_connected(self) -> bytes:
        self.connect()
        assert self._socket is not None
        while len(self._buffer) < GHOST_PACKET_BYTES:
            try:
                chunk = self._socket.recv(GHOST_PACKET_BYTES - len(self._buffer))
            except TimeoutError as error:
                self.close()
                raise TimeoutError(
                    "OpenPlanet accepted the ghost connection but sent no complete "
                    f"{GHOST_PACKET_BYTES}-byte datagram within {self.timeout_s:g}s. "
                    "Enable Ghost Replay Mode, load the .Replay.Gbx on the map, and play it."
                ) from error
            if not chunk:
                self.close()
                raise ConnectionError("OpenPlanet closed the ghost replay connection")
            self._buffer.extend(chunk)
        packet = bytes(self._buffer[:GHOST_PACKET_BYTES])
        del self._buffer[:GHOST_PACKET_BYTES]
        return packet

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer.clear()


def _control(values: np.ndarray) -> np.ndarray:
    selected = values[list(CONTROL_INDICES)]
    return np.asarray(
        [
            np.clip(selected[0], 0.0, 1.0),
            np.clip(selected[1], 0.0, 1.0),
            np.clip(selected[2], -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def _wait_for_ghost_start(client: GhostFrameReader, *, timeout_s: float) -> GhostFrame:
    previous_time = float("inf")
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        frame = client.read()
        race_time = frame.race_time_ms
        started = race_time > 0.0 and (race_time < previous_time or race_time <= GHOST_DT_MS * 2)
        if started and not frame.finished:
            return frame
        previous_time = race_time
    raise TimeoutError("no ghost replay start was observed; load the .Gbx and play it from 0:00")


def record_ghost_demonstration(
    client: GhostFrameReader,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
    *,
    max_duration_s: float,
    status: Callable[[str], None] = print,
) -> Demonstration:
    if max_duration_s <= 0.0:
        raise ValueError("max_duration_s must be positive")
    status("Waiting for Ghost Replay Mode. Load the replay and play it from the start.")
    current = _wait_for_ghost_start(client, timeout_s=config.start_timeout_s)
    _, table = build_brake_tap_action_table()
    frames, game_times, actions, controls = _ghost_buffers(current)
    deadline = monotonic() + max_duration_s
    while monotonic() < deadline:
        control = _control(current.values)
        actions.append(continuous_control_to_discrete_index(control, table))
        controls.append(control)
        current = client.read()
        if current.race_time_ms + 1e-3 < float(frames[-1][3]):
            current, deadline = _restart_ghost_lap(client, config, max_duration_s, status)
            frames, game_times, actions, controls = _ghost_buffers(current)
            continue
        _assert_aligned_sample(game_times[-1], current)
        frames.append(current.values.copy())
        game_times.append(current.game_time_ms)
        if current.finished:
            finish_time_s = current.race_time_ms / 1_000.0
            status(f"Finished ghost demonstration in {finish_time_s:.3f}s.")
            return Demonstration(
                map_uid=geometry.map_uid,
                geometry_sha256=geometry.sha256,
                action_repeat_frames=config.action_repeat_frames,
                frames=np.asarray(frames, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                controls=np.asarray(controls, dtype=np.float32),
                finish_time_s=finish_time_s,
            )
    raise TimeoutError("ghost demonstration did not reach the finish before max_duration_s")


def _ghost_buffers(
    current: GhostFrame,
) -> tuple[list[np.ndarray], list[float], list[int], list[np.ndarray]]:
    return [current.values.copy()], [current.game_time_ms], [], []


def _restart_ghost_lap(
    client: GhostFrameReader,
    config: TrackmaniaEnvironmentConfig,
    max_duration_s: float,
    status: Callable[[str], None],
) -> tuple[GhostFrame, float]:
    status("Ghost restart detected; discarding the partial lap.")
    deadline = monotonic() + max_duration_s
    remaining = deadline - monotonic()
    if remaining <= 0.0:
        raise TimeoutError("ghost demonstration did not reach the finish before max_duration_s")
    current = _wait_for_ghost_start(client, timeout_s=min(config.start_timeout_s, remaining))
    return current, deadline


def _assert_aligned_sample(previous_game_time_ms: float, frame: GhostFrame) -> None:
    delta_ms = frame.game_time_ms - previous_game_time_ms
    if delta_ms <= 0.0:
        raise ValueError("ghost gameTime must increase with each physics sample")
    if delta_ms + 1.0 < GHOST_DT_MS:
        raise ValueError(
            f"ghost sample interval {delta_ms:.1f} ms is faster than the {GHOST_DT_MS:.0f} ms grid"
        )


def extract_gbx_demo(
    request: GbxExtractRequest,
    client: GhostFrameReader,
    status: Callable[[str], None] = print,
) -> Path:
    meta = inspect_gbx(request.gbx_path)
    if meta.map_uid is not None and meta.map_uid != request.geometry.map_uid:
        raise ValueError(
            f"GBX map UID {meta.map_uid!r} does not match geometry {request.geometry.map_uid!r}"
        )
    demonstration = record_ghost_demonstration(
        client,
        request.config,
        request.geometry,
        max_duration_s=request.max_duration_s,
        status=status,
    )
    return _write_extracted_trajectory(request, meta, demonstration)


def _write_extracted_trajectory(
    request: GbxExtractRequest, meta: GbxMeta, demonstration: Demonstration
) -> Path:
    validate_demonstration(demonstration, request.config, request.geometry)
    npz_path = save_demonstration(request.output.with_suffix(".npz"), demonstration)
    transitions = demonstration_transitions(
        npz_path, request.pipeline, request.config, request.geometry
    )
    _assert_lag_free(demonstration)
    return save_trajectory(
        request.output,
        Trajectory(
            episode_id=f"ghost-{file_sha256(request.gbx_path)[:16]}",
            transitions=transitions,
            metadata={
                "source": "ghost",
                "gbx_sha256": meta.sha256,
                "sampling/projected_lap_time_s": demonstration.finish_time_s,
                "dt_s": GHOST_DT_MS / 1_000.0,
            },
        ),
    )


def _assert_lag_free(demonstration: Demonstration) -> None:
    race_times = demonstration.frames[:, 3]
    if np.any(np.diff(race_times) <= 0.0):
        raise ValueError("ghost trajectory race time must increase")
    recorded = demonstration.controls
    actual = demonstration.frames[:-1][:, list(CONTROL_INDICES)]
    actual = np.column_stack(
        (
            np.clip(actual[:, 0], 0.0, 1.0),
            np.clip(actual[:, 1], 0.0, 1.0),
            np.clip(actual[:, 2], -1.0, 1.0),
        )
    )
    if not np.allclose(recorded, actual, atol=1e-5):
        raise ValueError("ghost actions are not taken from the same physics sample as observations")
