#!/usr/bin/env python3
"""Run and archive the mandatory four-case NMPC SITL baseline suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.mavlink_params import DEFAULT_PARAMETERS, ParamGuard, ParamGuardError

DEFAULT_ROS_SETUP_SCRIPTS: tuple[Path, ...] = (
    Path("/opt/ros/humble/setup.bash"),
    Path("/home/cy/px4_ros2_ws/install/setup.bash"),
)
DEFAULT_RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
MAVLINK_PORT = 14580
AGENT_PORT = 8888
CASE_TIMEOUT_S = 300.0
_ENV_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _capture_env_after_sourcing(
    scripts: tuple[Path, ...], base_environment: Mapping[str, str]
) -> dict[str, str]:
    """Return the environment resulting from sourcing the ROS setup scripts."""
    command = [
        "bash", "--noprofile", "--norc", "-c",
        'for script in "$@"; do . "$script" || exit 1; done; env',
        "env-capture",
        *[str(script) for script in scripts],
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, env=dict(base_environment)
        )
    except FileNotFoundError as error:
        raise RuntimeError("bash not found; sourcing ROS setup scripts requires bash") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to source ROS setup script(s): {completed.stderr.strip()}"
        )
    return dict(
        line.split("=", 1)
        for line in completed.stdout.splitlines()
        if _ENV_LINE.match(line)
    )


def build_child_environment(
    *,
    acados_source_dir: Path,
    ros_setup_scripts: tuple[Path, ...],
    rmw_implementation: str,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the per-case child environment: acados libs plus ROS 2 overlays."""
    environment = dict(os.environ if base_environment is None else base_environment)
    environment["ACADOS_SOURCE_DIR"] = str(acados_source_dir)
    library_path = str(acados_source_dir / "lib")
    if environment.get("LD_LIBRARY_PATH"):
        library_path += ":" + environment["LD_LIBRARY_PATH"]
    environment["LD_LIBRARY_PATH"] = library_path
    if ros_setup_scripts:
        environment.update(_capture_env_after_sourcing(ros_setup_scripts, environment))
    # Set after capture: sourcing humble's setup.bash picks fastrtps when unset.
    environment["RMW_IMPLEMENTATION"] = rmw_implementation
    return environment


@dataclass(frozen=True)
class BaselineCase:
    name: str
    trajectory: str
    reference_source: str
    arguments: tuple[str, ...] = ()


CASES = (
    BaselineCase("hover", "hover", "px4-smoothed"),
    BaselineCase(
        "point_1m", "step", "px4-smoothed",
        ("--radius", "1.0", "--step-dwell", "2.0"),
    ),
    BaselineCase(
        "circle", "circle", "direct", ("--radius", "2.0", "--speed", "1.0")
    ),
    BaselineCase(
        "figure8", "figure8", "direct", ("--radius", "2.0", "--speed", "1.0")
    ),
)


def _udp_listener_on(port: int) -> bool:
    """True when a process has a UDP socket bound to `port` (Linux /proc)."""
    hex_port = f"{port:04X}"
    try:
        with open("/proc/net/udp", encoding="ascii") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 1 and fields[1].endswith(f":{hex_port}"):
                    return True
    except OSError:
        return False
    return False


def _px4_heartbeat_error(timeout: float = 5.0, port: int = MAVLINK_PORT) -> str | None:
    """Return a problem description, or None when PX4 answers a heartbeat."""
    try:
        from pymavlink import mavutil
    except ImportError:
        return "pymavlink not installed (pip install pymavlink); heartbeat check skipped"
    connection = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}")
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # A udpout connection transmits nothing until it sends a packet,
            # and PX4 only streams to addresses it has heard from.
            connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
            )
            if connection.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0):
                return None
        return f"no MAVLink heartbeat on udp:{MAVLINK_PORT} within {timeout} s"
    finally:
        connection.close()


def check_services(
    *,
    px4_ok: bool,
    px4_heartbeat_error: str | None,
) -> dict[str, tuple[bool, str]]:
    """Probe the SITL services the suite depends on; return status + hints.

    The suite deliberately does not start services itself: PX4 SITL and the
    MicroXRCE agent are long-lived, resource-heavy processes whose lifecycle
    belongs to the developer.  Missing services abort the run with the exact
    commands to start them.

    The PX4 check is normally answered by the parameter guard's persistent
    MAVLink connection (`px4_ok`): transient probe connections are avoided
    because PX4's MAVLink stream can wedge onto a stale client address and
    then keeps transmitting into a dead socket until PX4 is restarted.
    """
    checks: dict[str, tuple[bool, str]] = {}
    if px4_ok:
        checks["px4"] = (True, f"MAVLink heartbeat on udp:{MAVLINK_PORT}")
    else:
        checks["px4"] = (
            False,
            f"{px4_heartbeat_error or 'no MAVLink heartbeat'}. Start with "
            f"`make px4_sitl gz_x500` in the PX4 tree; if PX4 is already "
            f"running, restart it -- its MAVLink stream can wedge onto a "
            f"stale client address.",
        )
    if _udp_listener_on(AGENT_PORT):
        checks["micro_xrce_agent"] = (True, f"UDP listener on port {AGENT_PORT}")
    else:
        checks["micro_xrce_agent"] = (
            False,
            f"no UDP listener on port {AGENT_PORT}. Start with "
            f"`MicroXRCEAgent udp4 -p {AGENT_PORT}`.",
        )
    try:
        subprocess.run(
            ["pgrep", "-f", "gz sim"], check=True, capture_output=True, timeout=5.0
        )
        checks["gz_sim"] = (True, "gz sim process running")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        checks["gz_sim"] = (
            False,
            "no `gz sim` process. Start with `gz sim --verbose=1 -r -s "
            "<px4-tree>/Tools/simulation/gz/worlds/default.sdf`.",
        )
    return checks


def _write_suite_reports(directory: Path, results: dict[str, dict[str, object]]) -> None:
    summary_path = directory / "suite_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# NMPC SITL baseline",
        "",
        "| Case | Interface | Result | Position RMSE (m) | Velocity RMSE (m/s) | "
        "Attitude RMSE (rad) | Solve P99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        status = "PASS" if result.get("success", False) else "FAIL"
        lines.append(
            "| {name} | {source} | {status} | {position:.4f} | {velocity:.4f} | "
            "{attitude:.4f} | {solve:.3f} |".format(
                name=name,
                source=result.get("reference_source", "unknown"),
                status=status,
                position=float(result.get("tracking_position_rmse_m", float("nan"))),
                velocity=float(result.get("velocity_rmse_m_s", float("nan"))),
                attitude=float(result.get("attitude_rmse_rad", float("nan"))),
                solve=float(result.get("solve_p99_ms", float("nan"))),
            )
        )
    lines.extend(("", f"Machine-readable report: `{summary_path.name}`", ""))
    (directory / "suite_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _run_case(
    case: BaselineCase,
    *,
    command: list[str],
    case_directory: Path,
    child_environment: Mapping[str, str],
    timeout: float,
) -> tuple[dict[str, object], int]:
    """Run one case with a hard timeout; return (result, exit_code)."""
    print(f"\n=== NMPC baseline: {case.name} ({case.reference_source}) ===", flush=True)
    result: dict[str, object] | None = None
    timed_out = False
    interrupted = False
    deadline = time.monotonic() + timeout
    log_path = case_directory / "run.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=PROJECT_ROOT, env=child_environment,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    break
                ready, _, _ = select.select([process.stdout], [], [], remaining)
                if not ready:
                    timed_out = True
                    break
                line = process.stdout.readline()
                if not line:
                    break
                print(line, end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
                if line.startswith("NMPC_SITL_RESULT="):
                    result = json.loads(line.split("=", 1)[1])
                    break
        except KeyboardInterrupt:
            interrupted = True
        if timed_out or interrupted:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return_code = process.wait() if not (timed_out or interrupted) else process.returncode

    if interrupted:
        result = {
            "success": False,
            "reason": "interrupted by user",
            "reference_source": case.reference_source,
        }
    elif result is None:
        if timed_out:
            result = {
                "success": False,
                "reason": f"case timeout after {timeout:g} s",
                "reference_source": case.reference_source,
            }
        else:
            result = {
                "success": False,
                "reason": "missing NMPC_SITL_RESULT",
                "reference_source": case.reference_source,
            }
    result["case"] = case.name
    result["return_code"] = return_code
    result["run_log"] = str(log_path.resolve())
    (case_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, 130 if interrupted else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--acados-source-dir",
        type=Path,
        default=Path(os.environ.get("ACADOS_SOURCE_DIR", Path.home() / "acados")),
    )
    parser.add_argument("--altitude", type=float, default=1.0)
    parser.add_argument(
        "--cases", nargs="+", choices=[case.name for case in CASES],
        help="run only selected cases (default: all four mandatory cases)",
    )
    parser.add_argument(
        "--ros-setup-scripts", action="append", type=Path,
        help="ROS setup scripts to source for the child environment "
        "(default: /opt/ros/humble/setup.bash and the px4_msgs overlay)",
    )
    parser.add_argument(
        "--rmw-implementation", default=DEFAULT_RMW_IMPLEMENTATION,
        help="RMW implementation for the child environment",
    )
    parser.add_argument(
        "--case-timeout", type=float, default=CASE_TIMEOUT_S,
        help=f"per-case hard timeout in seconds (default {CASE_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--skip-service-check", action="store_true",
        help="do not probe px4/gz sim/MicroXRCEAgent before running",
    )
    parser.add_argument(
        "--skip-params", action="store_true",
        help="do not apply/restore the MAVLink parameter guard",
    )
    arguments = parser.parse_args()
    if arguments.altitude <= 0.0:
        parser.error("altitude must be positive")
    if arguments.case_timeout <= 0.0:
        parser.error("case timeout must be positive")
    acados_directory = arguments.acados_source_dir.resolve()
    if not (acados_directory / "lib/libacados.so").is_file():
        parser.error(f"acados shared libraries not found under {acados_directory}")
    ros_setup_scripts = (
        tuple(path.expanduser() for path in arguments.ros_setup_scripts)
        if arguments.ros_setup_scripts is not None
        else DEFAULT_ROS_SETUP_SCRIPTS
    )
    try:
        child_environment = build_child_environment(
            acados_source_dir=acados_directory,
            ros_setup_scripts=ros_setup_scripts,
            rmw_implementation=arguments.rmw_implementation,
        )
    except RuntimeError as error:
        parser.error(f"cannot build the case environment: {error}")

    # Guard the parameters the flights depend on FIRST: its MAVLink connection
    # stays open for the whole suite and the service check reuses it, because
    # transient probe connections can wedge PX4's MAVLink stream onto a stale
    # client address.  The guard restores the parameters even when the suite
    # is interrupted.  The airframe file already bakes the values, so with
    # --skip-service-check a guard failure degrades to a warning.
    guard: ParamGuard | None = None
    guard_error: ParamGuardError | None = None
    if not arguments.skip_params:
        print(f"\nParameter guard ({len(DEFAULT_PARAMETERS)} parameters):", flush=True)
        try:
            guard = ParamGuard()
            guard.__enter__()
        except ParamGuardError as error:
            guard_error = error

    if not arguments.skip_service_check:
        checks = check_services(
            px4_ok=guard is not None,
            px4_heartbeat_error=str(guard_error) if guard_error is not None else None,
        )
        print("SITL service check:", flush=True)
        for service, (available, note) in checks.items():
            print(f"  [{'OK' if available else 'MISSING'}] {service}: {note}", flush=True)
        if not all(available for available, _ in checks.values()):
            print(
                "\nRequired SITL services are not all running; the suite will not "
                "start them itself. Start the missing services above and re-run, "
                "or pass --skip-service-check to attempt the run anyway.",
                file=sys.stderr, flush=True,
            )
            if guard is not None:
                guard.__exit__(None, None, None)
            return 2
    if guard is None and not arguments.skip_params:
        print(
            f"warning: {guard_error} -- continuing without parameter guard",
            file=sys.stderr, flush=True,
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = arguments.output_directory or PROJECT_ROOT / "background/baseline_runs" / run_id
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = set(arguments.cases) if arguments.cases else None
    cases = [case for case in CASES if selected is None or case.name in selected]

    results: dict[str, dict[str, object]] = {}
    suite_success = True
    interrupted = False
    try:
        for case in cases:
            case_directory = output_directory / case.name
            case_directory.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "integration/px4_sitl_hover.py"),
                "--trajectory", case.trajectory,
                "--reference-source", case.reference_source,
                "--altitude", str(arguments.altitude),
                "--log-directory", str(case_directory),
                "--validate-model",
                *case.arguments,
            ]
            result, exit_code = _run_case(
                case,
                command=command,
                case_directory=case_directory,
                child_environment=child_environment,
                timeout=arguments.case_timeout,
            )
            results[case.name] = result
            _write_suite_reports(output_directory, results)
            if exit_code == 130:
                interrupted = True
                break
            if result.get("return_code") != 0 or not result.get("success", False):
                suite_success = False
                print(f"Baseline case failed: {case.name}; continuing suite", file=sys.stderr)
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)

    if interrupted:
        print("\nRun interrupted; results so far are archived.", file=sys.stderr)
        print(f"Baseline reports written to {output_directory}", flush=True)
        return 130
    markdown_path = output_directory / "suite_summary.md"
    print("\n" + markdown_path.read_text(encoding="utf-8"), flush=True)
    print(f"Baseline reports written to {output_directory}", flush=True)
    return 0 if suite_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
