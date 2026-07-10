"""Central relay server for distributed TMRL training."""

import pickle
import socket
import sys
import threading
import time
from typing import Any, cast

from loguru import logger
from tlspyo import Relay
from tlspyo.server import Server as TlspyoServer

import tmrl.config as cfg
from tmrl.networking.utils import print_with_timestamp


def _start_relay_windows_tcp(
    port: int,
    password: str,
    local_port: int,
    header_size: int,
    max_workers: int,
):
    """Run tlspyo relay server in a thread on Windows when TLS is disabled.

    The default tlspyo Relay uses a subprocess for the server; on Windows the child's
    stderr is often not visible, so bind failures (e.g. port in use) are silent. This
    runs the same server logic in a thread so any exception is visible in the same process.
    """
    local_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_srv.bind(("127.0.0.1", local_port))
    local_srv.listen()

    accepted_groups = {
        "trainers": {"max_count": 1, "max_consumables": None},
        "workers": {"max_count": max_workers, "max_consumables": None},
    }
    server = TlspyoServer(
        port=port,
        password=password,
        serializer=pickle.dumps,
        deserializer=pickle.loads,
        accepted_groups=accepted_groups,
        local_com_port=local_port,
        header_size=header_size,
        security="TCP",
        keys_dir=None,
    )

    def run_server():
        """Run the tlspyo server inside the Twisted reactor.

        Patches ``reactor.run`` to skip signal-handler installation, which is
        required when running the reactor inside a non-main thread (Python only
        allows signal handlers in the main thread).
        """
        from twisted.internet import reactor

        _reactor = cast(Any, reactor)
        _orig_run = _reactor.run

        def run_without_signals(install_signal_handlers=0):
            """Start the reactor with signal-handler installation disabled."""
            return _orig_run(installSignalHandlers=install_signal_handlers)

        _reactor.run = run_without_signals
        try:
            server.run()
        except Exception as e:
            logger.exception(
                "Relay server thread failed (this is the process that should bind to port {}): {}",
                port,
                e,
            )
            raise

    thread = threading.Thread(target=run_server, daemon=False)
    thread.start()

    conn, _ = local_srv.accept()
    msg = server.serializer(("TEST", None))
    if len(str(len(msg))) > header_size:
        raise ValueError(
            f"Message length {len(msg)} exceeds header capacity ({header_size} digits). "
            "Increase HEADER_SIZE in config."
        )
    header = bytes(f"{len(msg):<{header_size}}", "utf-8")
    conn.sendall(header + msg)

    return type("_WindowsTcpRelay", (), {"_thread": thread, "_conn": conn, "_srv": local_srv})()


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """

    def __init__(
        self,
        port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_SERVER,
        header_size=cfg.HEADER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        max_workers=cfg.NB_WORKERS,
    ):
        """
        Args:
            port (int): tlspyo public port
            password (str): tlspyo password
            local_port (int): tlspyo local communication port
            header_size (int): tlspyo header size (bytes)
            security (Union[str, None]): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory
            max_workers (int): max number of accepted workers
        """
        if sys.platform == "win32" and security is None:
            self.__relay = _start_relay_windows_tcp(
                port=port,
                password=password,
                local_port=local_port,
                header_size=header_size,
                max_workers=max_workers,
            )
        else:
            self.__relay = Relay(
                port=port,
                password=password,
                accepted_groups={
                    "trainers": {"max_count": 1, "max_consumables": None},
                    "workers": {"max_count": max_workers, "max_consumables": None},
                },
                local_com_port=local_port,
                header_size=header_size,
                security=security,
                keys_dir=keys_dir,
            )
        import logging

        logging.getLogger("tlspyo").setLevel(logging.INFO)
        print_with_timestamp(
            f"TMRL server listening on port {port} (trainers + workers). "
            "Leave this process running."
        )
        config_path = str(cfg.LOCAL_OVERRIDE_PATH)
        print_with_timestamp(
            f"Config: {config_path} (ensure server, trainer, worker use this same config)."
        )
        server_startup_delay = 0.5
        port_check_timeout = 2.0
        time.sleep(server_startup_delay)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(port_check_timeout)
                s.connect(("127.0.0.1", port))
            print_with_timestamp(f"Port {port} is open and accepting connections.")
        except OSError as e:
            print_with_timestamp(
                f"WARNING: Could not connect to 127.0.0.1:{port}. "
                f"Relay subprocess may have failed to bind (port in use or firewall). Error: {e}"
            )

    def stop(self):
        """Stop the server so the process can exit (e.g. after Ctrl+C)."""
        relay = getattr(self, "_Server__relay", None)
        if relay is None:
            return
        if hasattr(relay, "_thread"):
            try:
                from twisted.internet import reactor

                _reactor = cast(Any, reactor)
                _reactor.callFromThread(_reactor.stop)
            except Exception:
                pass
            relay_thread_join_timeout = 5.0
            relay._thread.join(timeout=relay_thread_join_timeout)
