"""Bounded, dependency-free OpenPlanet telemetry transport."""

from __future__ import annotations

import socket
import struct
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Never

import numpy as np

DEFAULT_TELEMETRY_FIELD_COUNT = 33
"""Number of float32 values emitted by TrackmaniaRL Connect / SAC_GetData 2.4.0."""

DEFAULT_POSITION_INDICES = (4, 5, 6)
"""``api.Position`` X/Y/Z offsets in the supported 33-field telemetry packet."""

DEFAULT_VELOCITY_INDICES = (7, 8, 9)
"""``api.Velocity`` X/Y/Z offsets in the supported 33-field telemetry packet."""


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """One validated OpenPlanet packet represented as immutable float values."""

    values: np.ndarray
    skipped_frames: int = 0

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or not np.isfinite(self.values).all():
            raise ValueError("Telemetry must be a finite one-dimensional float vector")
        if isinstance(self.skipped_frames, bool) or not isinstance(self.skipped_frames, int):
            raise TypeError("skipped_frames must be an integer")
        if self.skipped_frames < 0:
            raise ValueError("skipped_frames must be non-negative")


@dataclass(frozen=True, slots=True)
class OpenPlanetClientConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    timeout_s: float = 10.0


class OpenPlanetClient:
    """Synchronous latest-frame client with explicit packet and timeout validation."""

    def __init__(self, config: OpenPlanetClientConfig | None = None) -> None:
        config = config or OpenPlanetClientConfig()
        if config.port < 1 or config.timeout_s <= 0:
            raise ValueError("port and timeout_s must be positive")
        self.host = config.host
        self.port = config.port
        self.timeout_s = config.timeout_s
        self._packet = struct.Struct("<" + "f" * DEFAULT_TELEMETRY_FIELD_COUNT)
        self._socket: socket.socket | None = None
        self._buffer = bytearray()

    def connect(self) -> None:
        if self._socket is None:
            connection = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            connection.settimeout(self.timeout_s)
            self._socket = connection

    def read(self) -> TelemetryFrame:
        """Return the newest complete telemetry frame currently available.

        OpenPlanet streams frames continuously rather than responding to
        requests.  After waiting for one complete frame, drain queued socket
        data and retain only its most recent complete frame.  A trailing
        partial frame remains buffered for the next call.
        """
        return self._read(self._read_latest_connected)

    def read_next(self) -> TelemetryFrame:
        """Return the oldest complete frame without dropping queued telemetry."""

        return self._read(self._read_next_connected)

    def _read(self, read_connected: Callable[[], TelemetryFrame]) -> TelemetryFrame:
        reconnect_deadline = monotonic() + self.timeout_s
        while True:
            try:
                return read_connected()
            except ConnectionError as error:
                self.close()
                remaining_s = reconnect_deadline - monotonic()
                if remaining_s <= 0.0:
                    raise ConnectionError(
                        "OpenPlanet telemetry disconnected and did not accept a new "
                        f"connection within {self.timeout_s:g}s"
                    ) from error
                sleep(min(0.1, remaining_s))

    def _read_next_connected(self) -> TelemetryFrame:
        return self._frame(self._receive_packet(), skipped_frames=0)

    def _read_latest_connected(self) -> TelemetryFrame:
        packet = self._receive_packet()
        peer_closed = self._drain_queued_frames()
        packet, skipped_frames = self._take_latest_packet(packet)
        frame = self._frame(packet, skipped_frames)
        if peer_closed:
            self.close()
        return frame

    def _receive_packet(self) -> bytes:
        self.connect()
        assert self._socket is not None
        while len(self._buffer) < self._packet.size:
            try:
                chunk = self._socket.recv(self._packet.size - len(self._buffer))
            except TimeoutError as error:
                self._raise_receive_timeout(error)
            if not chunk:
                self.close()
                raise ConnectionError("OpenPlanet closed the telemetry connection")
            self._buffer.extend(chunk)
        return self._take_oldest_packet()

    def _raise_receive_timeout(self, error: TimeoutError) -> Never:
        self.close()
        raise TimeoutError(
            "OpenPlanet accepted the telemetry connection but sent no complete "
            f"{self._packet.size}-byte frame within {self.timeout_s:g}s. "
            "In Openplanet Plugin Manager, keep only the signed TrackmaniaRL Connect "
            "(SAC_GetData) 2.4.0 plugin enabled, enable School Mode, then enter the "
            "configured local map with a visible vehicle."
        ) from error

    def _take_latest_packet(self, packet: bytes) -> tuple[bytes, int]:
        skipped_frames = 0
        while len(self._buffer) >= self._packet.size:
            packet = self._take_oldest_packet()
            skipped_frames += 1
        return packet, skipped_frames

    def _frame(self, packet: bytes, skipped_frames: int) -> TelemetryFrame:
        values = np.asarray(self._packet.unpack(packet), dtype=np.float32)
        self._validate_grab_data(values)
        return TelemetryFrame(values, skipped_frames=skipped_frames)

    def _validate_grab_data(self, values: np.ndarray) -> None:
        """Apply domain checks for the supported 33-field SAC_GetData schema."""

        speed = float(values[16])
        if not 0.0 <= speed <= 2_500.0:
            raise ValueError(f"telemetry speed is outside the valid range: {speed}")
        # A map's spawn can legitimately be at or close to the world origin.
        # Never substitute an earlier position: position drives both reward and
        # lidar and a silent repair would corrupt the first transition after reset.

    def _take_oldest_packet(self) -> bytes:
        packet = bytes(self._buffer[: self._packet.size])
        del self._buffer[: self._packet.size]
        return packet

    def _drain_queued_frames(self) -> bool:
        """Drain bytes already queued by the telemetry producer without waiting."""

        assert self._socket is not None
        previous_timeout = self._socket.gettimeout()
        self._socket.setblocking(False)
        try:
            peer_closed = self._receive_available()
        finally:
            self._socket.settimeout(previous_timeout)
        return peer_closed

    def _receive_available(self) -> bool:
        assert self._socket is not None
        while True:
            try:
                chunk = self._socket.recv(64 * 1024)
            except BlockingIOError:
                return False
            if not chunk:
                return True
            self._buffer.extend(chunk)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer.clear()
