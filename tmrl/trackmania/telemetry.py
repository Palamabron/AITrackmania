"""Bounded, dependency-free OpenPlanet telemetry transport."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from time import monotonic, sleep

import numpy as np

DEFAULT_TELEMETRY_FIELD_COUNT = 33
"""Number of float32 values emitted by the supported TMRL_GrabData plugin."""

DEFAULT_POSITION_INDICES = (4, 5, 6)
"""``api.Position`` X/Y/Z offsets in the supported 33-field telemetry packet."""

DEFAULT_VELOCITY_INDICES = (7, 8, 9)
"""``api.Velocity`` X/Y/Z offsets in the supported 33-field telemetry packet."""


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """One validated OpenPlanet packet represented as immutable float values."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or not np.isfinite(self.values).all():
            raise ValueError("Telemetry must be a finite one-dimensional float vector")


class OpenPlanetClient:
    """Synchronous latest-frame client with explicit packet and timeout validation."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        *,
        field_count: int = DEFAULT_TELEMETRY_FIELD_COUNT,
        timeout_s: float = 10.0,
    ) -> None:
        if port < 1 or field_count < 1 or timeout_s <= 0:
            raise ValueError("port, field_count, and timeout_s must be positive")
        self.host, self.port, self.field_count, self.timeout_s = host, port, field_count, timeout_s
        self._packet = struct.Struct("<" + "f" * field_count)
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
        reconnect_deadline = monotonic() + self.timeout_s
        while True:
            try:
                return self._read_connected()
            except ConnectionError as error:
                self.close()
                remaining_s = reconnect_deadline - monotonic()
                if remaining_s <= 0.0:
                    raise ConnectionError(
                        "OpenPlanet telemetry disconnected and did not accept a new "
                        f"connection within {self.timeout_s:g}s"
                    ) from error
                sleep(min(0.1, remaining_s))

    def _read_connected(self) -> TelemetryFrame:
        self.connect()
        assert self._socket is not None
        while len(self._buffer) < self._packet.size:
            try:
                chunk = self._socket.recv(self._packet.size - len(self._buffer))
            except TimeoutError as error:
                self.close()
                raise TimeoutError(
                    "OpenPlanet accepted the telemetry connection but sent no complete "
                    f"{self._packet.size}-byte frame within {self.timeout_s:g}s. "
                    "In OpenPlanet, disable every legacy TMRL_GrabData plugin, keep only "
                    "TMRL_GrabData_IQN enabled, reload it, then enter the loaded map with "
                    "a visible vehicle."
                ) from error
            if not chunk:
                self.close()
                raise ConnectionError("OpenPlanet closed the telemetry connection")
            self._buffer.extend(chunk)
        packet = self._take_oldest_packet()
        peer_closed = self._drain_queued_frames()
        while len(self._buffer) >= self._packet.size:
            packet = self._take_oldest_packet()
        values = np.asarray(self._packet.unpack(packet), dtype=np.float32)
        self._validate_grab_data(values)
        if peer_closed:
            self.close()
        return TelemetryFrame(values)

    def _validate_grab_data(self, values: np.ndarray) -> None:
        """Apply domain checks for the supported 33-field TMRL_GrabData schema."""

        if self.field_count != DEFAULT_TELEMETRY_FIELD_COUNT:
            return
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
        peer_closed = False
        self._socket.setblocking(False)
        try:
            while True:
                try:
                    chunk = self._socket.recv(64 * 1024)
                except BlockingIOError:
                    break
                if not chunk:
                    peer_closed = True
                    break
                self._buffer.extend(chunk)
        finally:
            self._socket.settimeout(previous_timeout)
        return peer_closed

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer.clear()
