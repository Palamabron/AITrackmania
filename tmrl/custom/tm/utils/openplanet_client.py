import math
import socket
import struct
import time
from threading import Lock, Thread


class TM2020OpenPlanetClient:
    """Background socket client for the OpenPlanet TMRL_GrabData plugin.

    Public state read by interfaces:

    - ``_received_once`` — ``False`` until the first frame arrives; used to
      switch from ``first_packet_timeout`` to the shorter steady-state timeout.
    - ``_client_connected`` — ``True`` once ``socket.connect()`` succeeded
      (the first frame may still be pending while the car is in a menu).
    - ``_last_good_pos`` — most recent non-origin ``(x, y, z)``; used to patch
      the ``[0, 0, 0]`` glitch frames the plugin occasionally emits.
    - ``_last_retrieve_position_patched`` — position was patched in the frame
      just returned by :meth:`retrieve_data`.
    - ``_last_retrieve_invalid`` — sanity check failed on the frame just
      returned; the interface should terminate the episode.
    """

    def __init__(self, host="127.0.0.1", port=9000, struct_str=None, nb_floats=19):
        """Configure the struct layout and launch the background receive thread.

        Args:
            host: IP address of the OpenPlanet plugin host. Default ``"127.0.0.1"``.
            port: TCP port the plugin listens on. Default ``9000``.
            struct_str: ``struct.unpack`` format string. Defaults to a little-endian
                sequence of ``nb_floats`` 32-bit floats (``"<fff…"``).
            nb_floats: Number of float fields when ``struct_str`` is omitted.
                Ignored once ``struct_str`` is supplied.
        """
        if struct_str is None:
            struct_str = "<" + "f" * nb_floats
        self._struct_str = struct_str
        self.nb_floats = self._struct_str.count("f")
        self.nb_int32 = self._struct_str.count("i")
        self.nb_uint64 = self._struct_str.count("Q")
        self._nb_bytes = self.nb_floats * 4 + self.nb_uint64 * 8 + self.nb_int32 * 4

        self._host = host
        self._port = port

        self.__lock = Lock()
        self.__data = None
        self._received_once = False
        self._last_good_pos = None
        self._last_retrieve_position_patched = False
        self._last_retrieve_invalid = False
        self._client_connected = False
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """Connect to the plugin (retry until listening) and receive frames forever.

        Only the most recent complete frame is kept in ``self.__data`` — older
        buffered frames are discarded so the interface always reads the freshest
        telemetry even if its step rate lags the plugin's emit rate.
        """
        retry_interval = 2.0
        retry_count = 0
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self._host, self._port))
                    self._client_connected = True
                    print(
                        f"Connected to OpenPlanet plugin at {self._host}:{self._port}. "
                        "Waiting for game data (be in a map with car on track, not main menu).",
                        flush=True,
                    )
                    data_raw = b""
                    while True:
                        while len(data_raw) < self._nb_bytes:
                            chunk = s.recv(1024)
                            if not chunk:
                                self._client_connected = False
                                break
                            data_raw += chunk
                        if len(data_raw) < self._nb_bytes:
                            self._client_connected = False
                            break
                        div = len(data_raw) // self._nb_bytes
                        data_used = data_raw[(div - 1) * self._nb_bytes : div * self._nb_bytes]
                        data_raw = data_raw[div * self._nb_bytes :]
                        self.__lock.acquire()
                        self.__data = data_used
                        self._received_once = True
                        self.__lock.release()
            except (ConnectionRefusedError, OSError):
                self._client_connected = False
                retry_count += 1
                if retry_count == 1 or retry_count % 5 == 0:
                    print(
                        f"Cannot connect to {self._host}:{self._port} (attempt {retry_count}). "
                        "TrackMania running? TQC_GrabData loaded (F3→Developer→Reload)? In map?",
                        flush=True,
                    )
                time.sleep(retry_interval)
                continue

    def retrieve_data(self, sleep_if_empty=0.01, timeout=10.0, first_packet_timeout=60.0):
        """Return the most recent decoded telemetry frame, blocking until one is available.

        The background thread retains only the latest buffered frame, so the caller
        always receives the freshest data even when its step rate lags the plugin emit rate.

        Two timeout regimes apply:
          - Before the first frame: ``first_packet_timeout`` (allows time for the user
            to load a map and put the car on track).
          - After the first frame: ``timeout`` (steady-state gap tolerance).

        After returning, callers should inspect the public flags:
          - ``_last_retrieve_position_patched``: position was replaced with the last good
            sample because the plugin emitted a ``[0, 0, 0]`` glitch frame.
          - ``_last_retrieve_invalid``: speed sanity check failed; the interface should
            terminate the episode and discard the transition.

        Args:
            sleep_if_empty: Seconds to sleep between buffer polls (default 0.01 s).
            timeout: Maximum seconds to wait for subsequent frames (default 10 s).
            first_packet_timeout: Maximum seconds to wait for the very first frame
                (default 60 s).

        Returns:
            Decoded telemetry tuple as returned by ``struct.unpack(struct_str, …)``.

        Raises:
            AssertionError: If no frame arrives within the applicable timeout.
        """
        c = True
        t_start = None
        data = None
        last_hint_log = 0.0
        while c:
            self.__lock.acquire()
            if self.__data is not None:
                data = struct.unpack(self._struct_str, self.__data)
                c = False
                self.__data = None
            self.__lock.release()
            if c:
                if t_start is None:
                    t_start = time.time()
                t_now = time.time()
                elapsed = t_now - t_start
                effective_timeout = first_packet_timeout if not self._received_once else timeout
                if not self._received_once and elapsed - last_hint_log >= 15.0:
                    last_hint_log = elapsed
                    if self._client_connected:
                        print(
                            "Connected but no game data yet. Be IN A MAP with car on track "
                            "(drive or stand), not main menu or loading.",
                            flush=True,
                        )
                    else:
                        print(
                            f"Waiting for OpenPlanet plugin ({self._host}:{self._port}). "
                            "Start TrackMania, load map, TQC_GrabData (F3→Developer→Reload), "
                            "then be in map (car on track).",
                            flush=True,
                        )
                assert elapsed < effective_timeout, (
                    f"OpenPlanet stopped sending data since more than {effective_timeout}s. "
                    "Check: (1) TrackMania running, (2) TQC_GrabData (F3→Developer→Reload), "
                    "(3) IN A MAP with car on track (not menu/loading)."
                )
                time.sleep(sleep_if_empty)

        self._last_retrieve_position_patched = False
        self._last_retrieve_invalid = False
        if data is not None:
            # Position layout by struct size (TMRL_GrabData 33-float: indices 4-6).
            pos_start_idx = 4 if self.nb_floats >= 33 else 3 if self.nb_floats >= 20 else 2
            pos_x, pos_y, pos_z = (
                data[pos_start_idx],
                data[pos_start_idx + 1],
                data[pos_start_idx + 2],
            )

            if math.sqrt(pos_x**2 + pos_y**2 + pos_z**2) < 1.0:
                # Plugin occasionally emits [0,0,0] glitch frames; replace with last good sample.
                if self._last_good_pos is not None:
                    data_list = list(data)
                    data_list[pos_start_idx] = self._last_good_pos[0]
                    data_list[pos_start_idx + 1] = self._last_good_pos[1]
                    data_list[pos_start_idx + 2] = self._last_good_pos[2]
                    data = tuple(data_list)
                    self._last_retrieve_position_patched = True
            else:
                self._last_good_pos = (pos_x, pos_y, pos_z)

            # Speed-in-m/s sanity check (33-float at index 16; legacy TQC 20-float at 2).
            if self.nb_floats >= 33:
                speed_idx = 16
            elif self.nb_floats >= 20:
                speed_idx = 2
            else:
                speed_idx = 0
            if speed_idx < len(data):
                try:
                    speed_val = float(data[speed_idx])
                    if not (0 <= speed_val <= 2500.0):
                        self._last_retrieve_invalid = True
                except (TypeError, ValueError):
                    self._last_retrieve_invalid = True
        return data


def save_ghost(host="127.0.0.1", port=10000):
    """Trigger a ghost save by opening a TCP connection to the ghost-saving server.

    The server saves the current ghost on connection; the connection is closed
    immediately after. Requires a ghost-saving server to be running on the given port.

    Args:
        host: IP address of the ghost-saving server. Default ``"127.0.0.1"``.
        port: TCP port of the ghost-saving server. Default ``10000``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))


if __name__ == "__main__":
    pass
