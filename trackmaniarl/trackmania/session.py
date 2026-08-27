"""Local, versioned OpenPlanet map-readiness command protocol."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import cast

PLUGIN_PROTOCOL_VERSION = "2"


class OpenPlanetSessionError(RuntimeError):
    """Base class for failures reported by the local session protocol."""


class OpenPlanetSessionUnavailableError(OpenPlanetSessionError, ConnectionError):
    """The local Openplanet session endpoint could not complete a command."""


class OpenPlanetSessionProtocolError(OpenPlanetSessionError):
    """The local endpoint did not implement the expected session protocol."""


class OpenPlanetMapMismatchError(OpenPlanetSessionError):
    """The loaded map does not match the configured run."""


class OpenPlanetSessionNotReadyError(OpenPlanetSessionError):
    """The loaded map does not yet have a ready local player."""


@dataclass(frozen=True, slots=True)
class MapSessionReady:
    map_uid: str
    protocol_version: str


class OpenPlanetSessionClient:
    """One-command-per-connection JSONL protocol for a preloaded local map.

    OpenPlanet exposes the active map UID, but does not expose a documented API
    for loading an arbitrary local ``.Map.Gbx``.  The operator loads the map
    once, then this protocol makes every episode fail closed unless that exact
    UID is still active and the local player is ready after reset.
    """

    def __init__(self, host: str, port: int, *, timeout_s: float = 10.0) -> None:
        self.host, self.port, self.timeout_s = host, port, timeout_s

    def _command(self, command: str, **payload: str) -> dict[str, str]:
        request = self._encoded_request(command, payload)
        response = self._exchange(command, request)
        decoded = self._decoded_response(response)
        decoded = self._validated_response(command, decoded)
        return cast(dict[str, str], decoded)

    @staticmethod
    def _encoded_request(command: str, payload: dict[str, str]) -> bytes:
        text = json.dumps(
            {"protocol_version": PLUGIN_PROTOCOL_VERSION, "command": command, **payload},
            separators=(",", ":"),
        )
        return text.encode("utf-8") + b"\n"

    def _exchange(self, command: str, request: bytes) -> bytes:
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_s
            ) as connection:
                connection.settimeout(self.timeout_s)
                connection.sendall(request)
                return self._receive_response(connection)
        except OpenPlanetSessionError:
            raise
        except OSError as exc:
            raise OpenPlanetSessionUnavailableError(
                f"OpenPlanet session command {command!r} failed: {exc}"
            ) from exc

    @staticmethod
    def _receive_response(connection: socket.socket) -> bytes:
        response = b""
        while not response.endswith(b"\n"):
            chunk = connection.recv(4096)
            if not chunk:
                raise OpenPlanetSessionUnavailableError(
                    "OpenPlanet session channel disconnected before response"
                )
            response += chunk
            if len(response) > 16 * 1024:
                raise OpenPlanetSessionProtocolError("OpenPlanet session response exceeds 16 KiB")
        return response

    @staticmethod
    def _decoded_response(response: bytes) -> object:
        try:
            decoded = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenPlanetSessionProtocolError(
                "OpenPlanet session channel returned invalid JSON"
            ) from exc
        return decoded

    @staticmethod
    def _validated_response(command: str, decoded: object) -> dict[object, object]:
        if not isinstance(decoded, dict):
            raise OpenPlanetSessionProtocolError(
                f"OpenPlanet session command {command!r} rejected: {decoded!r}"
            )
        required = {"status", "protocol_version"}
        if required - decoded.keys() or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in decoded.items()
        ):
            raise OpenPlanetSessionProtocolError("OpenPlanet session response is incomplete")
        if decoded["status"] != "ok":
            raise OpenPlanetSessionProtocolError(
                f"OpenPlanet session command {command!r} rejected: {decoded!r}"
            )
        if decoded["protocol_version"] != PLUGIN_PROTOCOL_VERSION:
            raise OpenPlanetSessionProtocolError(
                "OpenPlanet session protocol version does not match Python client"
            )
        return decoded

    def inspect_loaded_map(self) -> MapSessionReady:
        """Return the active map identity after validating the protocol version."""

        response = self._command("verify_loaded_map")
        if "map_uid" not in response:
            raise OpenPlanetSessionProtocolError(
                "OpenPlanet session response did not include an active map UID"
            )
        map_uid = response["map_uid"].strip()
        if not map_uid:
            raise OpenPlanetSessionProtocolError(
                "OpenPlanet session response did not include an active map UID"
            )
        return MapSessionReady(map_uid, response["protocol_version"])

    def verify_loaded_map(self, expected_map_uid: str) -> MapSessionReady:
        """Require that the operator has loaded the configured local map already."""

        active = self.inspect_loaded_map()
        if active.map_uid != expected_map_uid:
            raise OpenPlanetMapMismatchError(
                "OpenPlanet active map UID does not match the configured map: "
                f"expected {expected_map_uid!r}, got {active.map_uid!r}"
            )
        return active

    def confirm_ready(self, expected_map_uid: str) -> MapSessionReady:
        """Confirm the already verified map has an active player after controller reset."""

        response = self._command("confirm_ready")
        required = {"map_uid", "ready"}
        if required - response.keys():
            raise OpenPlanetSessionProtocolError("OpenPlanet readiness response is incomplete")
        if response["map_uid"] != expected_map_uid or response["ready"] != "true":
            raise OpenPlanetSessionNotReadyError(
                f"OpenPlanet did not confirm a ready local player for map UID {expected_map_uid!r}"
            )
        return MapSessionReady(expected_map_uid, response["protocol_version"])
