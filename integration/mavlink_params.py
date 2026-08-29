"""MAVLink parameter guard for the SITL baseline suite.

Sets the PX4 parameters the baseline flights depend on, and restores the
previous values on exit -- including when the suite is interrupted -- over
the MAVLink UDP port (udpout:127.0.0.1:14580 by default, the standard SITL
MAVLink endpoint).

The airframe file already bakes the required values into this repository's
SITL build, but a fresh PX4 checkout on another machine may not have them.
Guarding them at runtime makes the suite one-click on an unmodified PX4.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_CONNECTION_STRING = "udpout:127.0.0.1:14580"
HEARTBEAT_TIMEOUT_S = 10.0
PARAM_SET_TIMEOUT_S = 2.0
PARAM_SET_RETRIES = 3


@dataclass(frozen=True)
class GuardedParameter:
    name: str
    value: float
    note: str = ""


# SIM_BAT_DRAIN > 0 drains the simulated battery during long flights and
# eventually trips PX4 failsafes mid-case.  NAV_DLL_ACT arms datalink-loss
# handling, which can abort the mission while the NMPC child (which talks to
# PX4 through uXRCE-DDS, not MAVLink) is flying.
DEFAULT_PARAMETERS = (
    GuardedParameter("SIM_BAT_DRAIN", 0.0, "disable simulated battery drain"),
    GuardedParameter("NAV_DLL_ACT", 0.0, "disable datalink loss actions"),
)


class ParamGuardError(RuntimeError):
    """Raised when a guarded parameter cannot be set or restored."""


def _require_pymavlink() -> Any:
    try:
        from pymavlink import mavutil
    except ImportError as error:
        raise ParamGuardError(
            "pymavlink is not installed. Install it with "
            "`pip install pymavlink` or `pip install -e .[dev]`."
        ) from error
    return mavutil


class ParamGuard:
    """Context manager: set parameters on entry, restore originals on exit."""

    def __init__(
        self,
        parameters: tuple[GuardedParameter, ...] = DEFAULT_PARAMETERS,
        connection_string: str = DEFAULT_CONNECTION_STRING,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT_S,
    ) -> None:
        self.parameters = parameters
        self.connection_string = connection_string
        self.heartbeat_timeout = heartbeat_timeout
        self._connection: Any | None = None
        self._originals: dict[str, float] = {}
        self._changed: set[str] = set()

    def __enter__(self) -> "ParamGuard":
        self._connect()
        try:
            self._apply()
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, *exc_info: object) -> bool:
        try:
            self._restore()
        finally:
            self._close()
        return False

    # -- connection -------------------------------------------------------

    def _connect(self) -> None:
        mavutil = _require_pymavlink()
        self._connection = mavutil.mavlink_connection(self.connection_string)
        start = time.monotonic()
        heartbeat = None
        while time.monotonic() - start < self.heartbeat_timeout:
            heartbeat = self._connection.recv_match(
                type="HEARTBEAT", blocking=True, timeout=1.0
            )
            if heartbeat is not None:
                break
        if heartbeat is None:
            self._close()
            raise ParamGuardError(
                f"no MAVLink heartbeat on {self.connection_string} within "
                f"{self.heartbeat_timeout} s. Is PX4 SITL running? "
                "(make px4_sitl gz_x500)"
            )
        self._connection.target_system = heartbeat.get_srcSystem()
        print(
            f"ParamGuard: MAVLink connected to system {heartbeat.get_srcSystem()}"
            f" (autopilot type {heartbeat.type})", flush=True
        )

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # -- parameter IO -----------------------------------------------------

    def _read(self, name: str, timeout: float = PARAM_SET_TIMEOUT_S) -> float:
        assert self._connection is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._connection.mav.param_request_read_send(
                self._connection.target_system,
                self._connection.target_component,
                name.encode("ascii"),
                -1,
            )
            response = self._connection.recv_match(
                type="PARAM_VALUE", blocking=True, timeout=0.25
            )
            if response is not None and response.param_id.rstrip("\0 ") == name:
                return float(response.param_value)
        raise ParamGuardError(f"no PARAM_VALUE for {name} within {timeout} s")

    def _write(self, name: str, value: float) -> None:
        assert self._connection is not None
        mavutil = _require_pymavlink()
        for attempt in range(PARAM_SET_RETRIES):
            self._connection.mav.param_set_send(
                self._connection.target_system,
                self._connection.target_component,
                name.encode("ascii"),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            if self._read(name) == value:
                return
        raise ParamGuardError(f"parameter {name} still differs after {PARAM_SET_RETRIES} sets")

    # -- apply / restore --------------------------------------------------

    def _apply(self) -> None:
        for parameter in self.parameters:
            original = self._read(parameter.name)
            self._originals[parameter.name] = original
            if original == parameter.value:
                print(
                    f"ParamGuard: {parameter.name} already {original:g} "
                    f"({parameter.note})", flush=True
                )
                continue
            self._write(parameter.name, parameter.value)
            self._changed.add(parameter.name)
            print(
                f"ParamGuard: {parameter.name} {original:g} -> {parameter.value:g} "
                f"({parameter.note})", flush=True
            )

    def _restore(self) -> None:
        failures = []
        for name in self._changed:
            try:
                self._write(name, self._originals[name])
                print(
                    f"ParamGuard: restored {name} to {self._originals[name]:g}",
                    flush=True,
                )
            except ParamGuardError as error:
                failures.append(f"{name}: {error}")
        if failures:
            print(
                "ParamGuard: WARNING failed to restore parameters: "
                + "; ".join(failures),
                file=sys.stderr,
                flush=True,
            )


def _probe_main() -> int:
    """CLI: verify MAVLink connectivity and print the guarded parameters."""
    guard = ParamGuard()
    with guard:
        for parameter in guard.parameters:
            try:
                current = guard._read(parameter.name)
            except ParamGuardError as error:
                print(f"{parameter.name}: read failed ({error})")
                continue
            mark = "OK" if current == parameter.value else f"needs {parameter.value:g}"
            print(f"{parameter.name} = {current:g} [{mark}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_probe_main())
