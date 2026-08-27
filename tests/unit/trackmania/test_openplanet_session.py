"""Release contracts for the OpenPlanet session and environment."""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from trackmaniarl.trackmania.environment import _validated_live_map_uid
from trackmaniarl.trackmania.session import PLUGIN_PROTOCOL_VERSION, OpenPlanetSessionClient


def _serve_session(server: socket.socket, commands: list[str]) -> None:
    for _ in range(2):
        connection, _ = server.accept()
        with connection:
            request = json.loads(connection.recv(4096).decode("utf-8"))
            commands.append(request["command"])
            response = {
                "status": "ok",
                "protocol_version": PLUGIN_PROTOCOL_VERSION,
                "map_uid": "trackmaniarl-test",
                "ready": "true",
            }
            connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
    server.close()


def test_session_protocol_verifies_preloaded_map_and_ready_state() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    host, port = server.getsockname()

    commands: list[str] = []

    thread = threading.Thread(target=_serve_session, args=(server, commands))
    thread.start()
    client = OpenPlanetSessionClient(host, port, timeout_s=1)
    assert client.verify_loaded_map("trackmaniarl-test").map_uid == "trackmaniarl-test"
    assert client.confirm_ready("trackmaniarl-test").map_uid == "trackmaniarl-test"
    thread.join(timeout=1)
    assert commands == ["verify_loaded_map", "confirm_ready"]


def test_live_map_preflight_rejects_placeholders_and_unbound_geometry() -> None:
    geometry = SimpleNamespace(map_uid="trackmaniarl-test", map_sha256="0" * 64)
    with pytest.raises(ValueError, match="real expected_map_uid"):
        _validated_live_map_uid("REPLACE_WITH_MAP_UID", geometry)
    with pytest.raises(ValueError, match="real expected_map_uid"):
        _validated_live_map_uid("<map-uid>", geometry)

    geometry = SimpleNamespace(map_uid="trackmaniarl-test", map_sha256="")
    with pytest.raises(ValueError, match="source map checksum"):
        _validated_live_map_uid("trackmaniarl-test", geometry)
