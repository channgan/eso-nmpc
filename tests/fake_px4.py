"""In-process fake PX4 answering MAVLink heartbeat/param traffic.

Binds an ephemeral UDP port with pymavlink's own packet encoding, so tests
exercise the real handshake: a udpout client must announce itself before
the server streams to it, param reads/writes go through PARAM_VALUE, etc.
"""

from __future__ import annotations

import socket
import threading
from typing import Any


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakePx4:
    """Serve heartbeats and a small parameter table over UDP."""

    def __init__(self, params: dict[str, float] | None = None) -> None:
        import pymavlink  # late import: skip-able when pymavlink is absent

        assert pymavlink
        self.port = free_udp_port()
        self.params: dict[str, float] = dict(params or {})
        self.connection: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "FakePx4":
        from pymavlink import mavutil

        self.connection = mavutil.mavlink_connection(f"udpin:127.0.0.1:{self.port}")
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        # Close the socket inside the serve loop: closing it from here while
        # the loop is blocked in recvfrom raises Bad file descriptor.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "FakePx4":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _serve(self) -> None:
        assert self.connection is not None
        try:
            while not self._stop.is_set():
                message = self.connection.recv_match(blocking=True, timeout=0.2)
                if message is None:
                    continue
                message_type = message.get_type()
                if message_type == "HEARTBEAT":
                    # Only stream once we have heard from the client, like PX4.
                    self.connection.target_system = message.get_srcSystem()
                    self.connection.target_component = message.get_srcComponent()
                    self._send_heartbeat()
                elif message_type == "PARAM_REQUEST_READ":
                    self._send_param_value(message.param_id.rstrip("\0 "))
                elif message_type == "PARAM_SET":
                    self.params[message.param_id.rstrip("\0 ")] = float(
                        message.param_value
                    )
        finally:
            self.connection.close()

    def _send_heartbeat(self) -> None:
        from pymavlink import mavutil

        self.connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_PX4,
            0, 0, 0,
        )

    def _send_param_value(self, name: str) -> None:
        from pymavlink import mavutil

        if name not in self.params:
            return
        self.connection.mav.param_value_send(
            name.encode("ascii"),
            float(self.params[name]),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            len(self.params),
            len(self.params),
        )
