#!/usr/bin/env python3
"""Run the ESO disturbance matrix sequentially with SITL restart/retry."""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PX4 = Path("/home/cy/apx")
BUILD = PX4 / "build/px4_sitl_default"
MODEL = PX4 / "Tools/simulation/gz/models/x500_base/model.sdf"
WORLD = PX4 / "Tools/simulation/gz/worlds/default.sdf"
TRAJECTORIES = ("hover", "point_1m", "circle", "figure8")
CONDITIONS = {
    "wind": (False, True, False),
    "cg": (True, False, False),
    "noise": (False, False, True),
    "wind_cg": (True, True, False),
    "wind_noise": (False, True, True),
    "cg_noise": (True, False, True),
    "all": (True, True, True),
}


def _pids_for_process(kind: str) -> list[int]:
    """Find only the SITL processes owned by this supervisor.

    ``pgrep -x gz`` does not match the real Gazebo command (``gz sim ...``),
    which previously left the old simulator alive across matrix cases.  Use
    the complete command line and explicit PX4 build paths instead.
    """
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    pids: list[int] = []
    for line in listing:
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        command = fields[1]
        if pid == os.getpid():
            continue
        if kind == "px4" and (
            command.rstrip().endswith("/bin/px4")
            or command.rstrip() == "px4"
        ):
            pids.append(pid)
        elif kind == "gazebo" and command.startswith("gz sim"):
            pids.append(pid)
        elif kind == "ninja" and "ninja -C " in command and " gz_x500" in command:
            pids.append(pid)
    return pids


def _terminate_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.1)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _cleanup_sitl() -> None:
    """Stop stale PX4/Gazebo launchers before starting a fresh case."""
    _terminate_pids(_pids_for_process("px4"))
    _terminate_pids(_pids_for_process("ninja"))
    _terminate_pids(_pids_for_process("gazebo"))


def _udp_listener_on(port: int) -> bool:
    """Return whether Linux has a UDP listener on ``port``."""
    hex_port = f"{port:04X}"
    try:
        with open("/proc/net/udp", encoding="ascii") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) > 1 and fields[1].endswith(f":{hex_port}"):
                    return True
    except OSError:
        return False
    return False


def restart_sitl(log: Path) -> subprocess.Popen[str]:
    _cleanup_sitl()
    time.sleep(2)
    proc = subprocess.Popen(
        ["ninja", "-C", str(BUILD), "gz_x500"],
        stdout=log.open("w"), stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            text = log.read_text(errors="replace")
        except OSError:
            text = ""
        if (
            "Gazebo world is ready" in text
            and "Spawning Gazebo model" in text
            and "Startup script returned successfully" in text
        ):
            # Give uXRCE-DDS and EKF a short settling interval before the
            # controller is launched.  Starting immediately after Gazebo
            # spawn is a recurrent source of a false first-case timeout.
            time.sleep(5.0)
            return proc
        time.sleep(1.0)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise RuntimeError("SITL did not report Gazebo world ready within 60 s")


def configure(condition: str) -> None:
    cg, wind, _noise = CONDITIONS[condition]
    model = MODEL.read_text()
    model = re.sub(r"(<inertial>\s*<pose>)[^<]+(</pose>)", rf"\g<1>{'0.02 0 0 0 0 0' if cg else '0 0 0 0 0 0'}\2", model, count=1)
    MODEL.write_text(model)
    world = WORLD.read_text()
    world = re.sub(r"\s*<wind>\s*<linear_velocity>[^<]+</linear_velocity>\s*</wind>", "", world)
    if wind:
        world = world.replace("  </world>", "    <wind>\n      <linear_velocity>0.5 0 0</linear_velocity>\n    </wind>\n  </world>")
    WORLD.write_text(world)


def run_case(
    condition: str,
    eso_on: bool,
    output: Path,
    seed: int,
    cases: tuple[str, ...],
    noise_options: tuple[str, ...],
    rmw_implementation: str,
    eso_bandwidth: float,
    case_timeout: float,
) -> int:
    cg, wind, noise = CONDITIONS[condition]
    command = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "integration/run_sitl_regression.py"),
        "--output-directory", str(output), "--eso-bandwidth", str(eso_bandwidth),
        "--cg-offset-x-m", "0.02" if cg else "0",
        "--noise-seed", str(seed), "--wind-velocity-x-m-s", "0.5" if wind else "0",
        "--case-timeout", str(case_timeout),
        "--rmw-implementation", rmw_implementation,
        # The service probe can miss the localhost MAVLink heartbeat even
        # while PX4's DDS/state stream is healthy.  The child controller still
        # enforces its own odometry/status/preflight gates.
        "--skip-service-check", "--skip-params",
        "--cases", *cases,
    ]
    if noise:
        command.extend(noise_options)
    else:
        command.extend((
            "--position-bias-rw-std-m-sqrt-s", "0.0",
            "--velocity-bias-rw-std-m-s-sqrt-s", "0.0",
        ))
    if not eso_on:
        command.append("--disable-eso")
    # PX4 in this WSL setup advertises DDS on eth1 (192.168.31.164), so the
    # ROS side must not force localhost-only discovery.  The caller supplies
    # PX4_UXRCE_DDS_AGENT_IP when needed; all other launch settings are
    # inherited so the mixed ROS/.venv/acados environment is preserved.
    child_environment = os.environ.copy()
    child_environment["ROS_LOCALHOST_ONLY"] = "0"
    child_environment["RMW_IMPLEMENTATION"] = rmw_implementation
    return subprocess.run(command, cwd=ROOT, env=child_environment).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "background/baseline_runs/eso_comparison/matrix_supervised")
    parser.add_argument("--only-eso", choices=("on", "off", "both"), default="both")
    parser.add_argument("--eso-bandwidth", type=float, default=3.0,
                        help="ESO bandwidth in rad/s (default: current baseline 3.0)")
    parser.add_argument("--case-timeout", type=float, default=180.0,
                        help="per-trajectory child timeout in seconds (default: 180)")
    parser.add_argument("--rmw-implementation", default="rmw_fastrtps_cpp")
    parser.add_argument("--position-bias-rw-std-m-sqrt-s", type=float, default=0.002)
    parser.add_argument("--velocity-bias-rw-std-m-s-sqrt-s", type=float, default=0.005)
    parser.add_argument("--cases", nargs="+", choices=TRAJECTORIES, default=list(TRAJECTORIES),
                        help="only run these trajectories; useful for resuming incomplete cases")
    args = parser.parse_args()
    noise_options = (
        "--position-bias-rw-std-m-sqrt-s", str(args.position_bias_rw_std_m_sqrt_s),
        "--velocity-bias-rw-std-m-s-sqrt-s", str(args.velocity_bias_rw_std_m_s_sqrt_s),
    )
    original_model, original_world = MODEL.read_bytes(), WORLD.read_bytes()
    args.output_root.mkdir(parents=True, exist_ok=True)
    # The supervisor owns PX4/Gazebo lifecycle, while the Agent is an
    # intentionally long-lived external service.  Fail before creating a
    # matrix of guaranteed odometry timeouts when it is absent.
    if not _udp_listener_on(8888):
        raise RuntimeError(
            "MicroXRCEAgent is not listening on UDP 8888; start "
            "`MicroXRCEAgent udp4 -p 8888` first"
        )
    modes = (True, False) if args.only_eso == "both" else (args.only_eso == "on",)
    try:
        for condition in args.conditions:
            for eso_on in modes:
                label = f"{condition}_eso_{'on' if eso_on else 'off'}"
                output = args.output_root / label
                configure(condition)
                # Restart PX4/Gazebo for every trajectory.  A failed or
                # safety-aborted child can otherwise leave stale DDS clients
                # behind and make the next trajectory report solve_count=0.
                for case in args.cases:
                    sitl_log = args.output_root / f"{label}_{case}_sitl.log"
                    proc = restart_sitl(sitl_log)
                    try:
                        rc = run_case(
                            condition, eso_on, output, args.seed, (case,),
                            noise_options, args.rmw_implementation, args.eso_bandwidth,
                            args.case_timeout,
                        )
                        print(
                            f"MATRIX_RESULT {label} case={case} return_code={rc}",
                            flush=True,
                        )
                    finally:
                        if proc.poll() is None:
                            try:
                                os.killpg(proc.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
    finally:
        _cleanup_sitl()
        MODEL.write_bytes(original_model)
        WORLD.write_bytes(original_world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
