"""Local, versioned OpenPlanet map-readiness command protocol."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass

PLUGIN_PROTOCOL_VERSION = "2"


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
        request = json.dumps(
            {"protocol_version": PLUGIN_PROTOCOL_VERSION, "command": command, **payload},
            separators=(",", ":"),
        )
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_s
            ) as connection:
                connection.settimeout(self.timeout_s)
                connection.sendall(request.encode("utf-8") + b"\n")
                response = b""
                while not response.endswith(b"\n"):
                    chunk = connection.recv(4096)
                    if not chunk:
                        raise RuntimeError(
                            "OpenPlanet session channel disconnected before response"
                        )
                    response += chunk
                    if len(response) > 16 * 1024:
                        raise RuntimeError("OpenPlanet session response exceeds 16 KiB")
        except OSError as exc:
            raise RuntimeError(f"OpenPlanet session command {command!r} failed: {exc}") from exc
        try:
            decoded = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OpenPlanet session channel returned invalid JSON") from exc
        if not isinstance(decoded, dict) or decoded.get("status") != "ok":
            raise RuntimeError(f"OpenPlanet session command {command!r} rejected: {decoded!r}")
        if decoded.get("protocol_version") != PLUGIN_PROTOCOL_VERSION:
            raise RuntimeError("OpenPlanet session protocol version does not match Python client")
        return {str(key): str(value) for key, value in decoded.items()}

    def verify_loaded_map(self, expected_map_uid: str) -> MapSessionReady:
        """Require that the operator has loaded the configured local map already."""

        response = self._command("verify_loaded_map")
        if response.get("map_uid") != expected_map_uid:
            raise RuntimeError("OpenPlanet active map UID does not match the configured map")
        return MapSessionReady(expected_map_uid, response["protocol_version"])

    def confirm_ready(self, expected_map_uid: str) -> MapSessionReady:
        """Confirm the already verified map has an active player after controller reset."""

        response = self._command("confirm_ready")
        if response.get("map_uid") != expected_map_uid or response.get("ready") != "true":
            raise RuntimeError(
                "OpenPlanet did not confirm a ready session for the requested map UID"
            )
        return MapSessionReady(expected_map_uid, response["protocol_version"])
