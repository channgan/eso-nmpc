"""ParamGuard behavior against an in-process fake PX4.

These tests pin the two handshake bugs found against the real SITL stack:
a udpout client must announce itself before PX4 streams to it, and a stale
partner address wedges the stream.  See tests/fake_px4.py.
"""

import socket

import pytest

pymavlink = pytest.importorskip("pymavlink")

from integration.mavlink_params import (  # noqa: E402
    DEFAULT_PARAMETERS,
    ParamGuard,
    ParamGuardError,
)
from integration.run_sitl_regression import (  # noqa: E402
    _px4_heartbeat_error,
    _udp_listener_on,
)

from fake_px4 import FakePx4, free_udp_port  # noqa: E402


def test_udp_listener_detection() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        assert _udp_listener_on(port)
    assert not _udp_listener_on(free_udp_port())


def test_heartbeat_check_detects_live_server() -> None:
    with FakePx4() as px4:
        assert _px4_heartbeat_error(timeout=3.0, port=px4.port) is None


def test_heartbeat_check_reports_dead_server() -> None:
    error = _px4_heartbeat_error(timeout=1.0, port=free_udp_port())
    assert error is not None
    assert "heartbeat" in error


def test_guard_applies_and_restores_parameters() -> None:
    # 0.125 is exact in REAL32 (the MAVLink param wire format); 0.1 would not
    # round-trip bit-exactly.
    params = {"SIM_BAT_DRAIN": 0.5, "NAV_DLL_ACT": 0.125}
    with FakePx4(params=params) as px4:
        guard = ParamGuard(
            parameters=DEFAULT_PARAMETERS,
            connection_string=f"udpout:127.0.0.1:{px4.port}",
            heartbeat_timeout=5.0,
        )
        with guard:
            for parameter in DEFAULT_PARAMETERS:
                assert px4.params[parameter.name] == pytest.approx(parameter.value)
        # Restored on exit.
        for name, value in params.items():
            assert px4.params[name] == pytest.approx(value)
        assert guard._connection is None


def test_guard_leaves_untouched_parameters_alone() -> None:
    params = {"NAV_DLL_ACT": 0.0}
    with FakePx4(params=params) as px4:
        guard = ParamGuard(
            parameters=(DEFAULT_PARAMETERS[1],),
            connection_string=f"udpout:127.0.0.1:{px4.port}",
        )
        with guard:
            pass
        # Already at the guarded value: nothing was written, nothing restored.
        assert px4.params == params
        assert guard._changed == set()


def test_guard_fails_without_heartbeat() -> None:
    guard = ParamGuard(
        connection_string=f"udpout:127.0.0.1:{free_udp_port()}",
        heartbeat_timeout=1.0,
    )
    with pytest.raises(ParamGuardError):
        guard.__enter__()
    assert guard._connection is None
