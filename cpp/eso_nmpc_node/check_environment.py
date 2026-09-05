#!/usr/bin/env python3
"""Check the dependencies required to build and run eso_nmpc_node.

Run this after sourcing the ROS 2 and px4_msgs workspaces.  The checker is
deliberately read-only: it does not install packages, modify PX4 parameters,
or start any process.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")

    def command(self, name: str, required: bool = True) -> Path | None:
        path = shutil.which(name)
        if path:
            self.ok(f"command {name}: {path}")
            return Path(path)
        (self.fail if required else self.warn)(f"command {name} is not installed")
        return None

    def run(self, command: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None

    def ros_package(self, package: str) -> None:
        result = self.run(["ros2", "pkg", "prefix", package])
        if result is not None and result.returncode == 0:
            self.ok(f"ROS package {package}: {result.stdout.strip()}")
        else:
            self.fail(f"ROS package {package} is not visible; source the matching ROS/PX4 workspace")

    def python_module(self, module: str, required: bool) -> None:
        result = self.run([sys.executable, "-c", f"import {module}"])
        if result is not None and result.returncode == 0:
            self.ok(f"Python module {module}: available")
        else:
            (self.fail if required else self.warn)(f"Python module {module} is unavailable")


def _existing_file(checker: Checker, path: Path, label: str, required: bool = True) -> None:
    if path.is_file():
        checker.ok(f"{label}: {path}")
    else:
        (checker.fail if required else checker.warn)(f"{label} missing: {path}")


def _existing_dir(checker: Checker, path: Path, label: str, required: bool = True) -> None:
    if path.is_dir():
        checker.ok(f"{label}: {path}")
    else:
        (checker.fail if required else checker.warn)(f"{label} missing: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="eso_nmpc repository root")
    parser.add_argument("--acados", type=Path, default=None,
                        help="acados source/build prefix")
    parser.add_argument("--executable", type=Path,
                        help="optional built eso_nmpc_node executable to inspect with ldd")
    parser.add_argument("--target-aarch64", action="store_true",
                        help="treat non-aarch64 hosts as an error instead of a warning")
    parser.add_argument("--require-generator", action="store_true",
                        help="also require CasADi and acados_template for solver regeneration")
    args = parser.parse_args()

    root = args.root.resolve()
    acados = args.acados.resolve() if args.acados else (root.parent / "acados").resolve()
    checker = Checker()
    print(f"eso_nmpc deployment check\nroot: {root}\nacados: {acados}\n")

    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        checker.ok(f"target architecture: {machine}")
    elif args.target_aarch64:
        checker.fail(f"target architecture is {machine}, expected aarch64/arm64")
    else:
        checker.warn(f"target architecture is {machine}; WSL/x86 is valid only for a build smoke test")

    for command in ("python3", "cmake", "colcon", "ros2", "gcc", "g++"):
        checker.command(command)
    checker.command("MicroXRCEAgent")

    ros_distro = os.environ.get("ROS_DISTRO")
    if ros_distro:
        checker.ok(f"ROS_DISTRO={ros_distro}")
    else:
        checker.fail("ROS_DISTRO is unset; source /opt/ros/<distro>/setup.bash")
    if os.environ.get("RMW_IMPLEMENTATION") == "rmw_cyclonedds_cpp":
        checker.ok("RMW_IMPLEMENTATION=rmw_cyclonedds_cpp")
    else:
        checker.warn("RMW_IMPLEMENTATION is not rmw_cyclonedds_cpp")

    for package in ("rclcpp", "std_msgs", "px4_msgs", "rmw_cyclonedds_cpp"):
        checker.ros_package(package)
    msg = checker.run(["ros2", "interface", "show", "px4_msgs/msg/NmpcTrajectorySetpoint"])
    if msg is not None and msg.returncode == 0 and "uint64 timestamp" in msg.stdout and "float32[153] position" in msg.stdout:
        checker.ok("px4_msgs/msg/NmpcTrajectorySetpoint: custom complete-trajectory interface")
    else:
        checker.fail("custom NmpcTrajectorySetpoint is not available in the sourced px4_msgs")
    for interface, required_field in (
        ("px4_msgs/msg/OffboardControlMode", "bool body_rate"),
        ("px4_msgs/msg/VehicleRatesSetpoint", "float32[3] thrust_body"),
    ):
        interface_result = checker.run(["ros2", "interface", "show", interface])
        if interface_result is not None and interface_result.returncode == 0 and required_field in interface_result.stdout:
            checker.ok(f"{interface}: required field {required_field}")
        else:
            checker.fail(f"{interface} is missing required field {required_field}")

    _existing_file(checker, root / "cpp/eso_nmpc_node/package.xml", "C++ package manifest")
    _existing_file(checker, root / "cpp/eso_nmpc_node/CMakeLists.txt", "C++ CMake file")
    _existing_file(checker, root / "cpp/eso_nmpc_node/config/eso_nmpc_cpp.yaml", "C++ parameter file")
    _existing_file(checker, root / "solver/generate_solver.py", "solver generation script")
    _existing_dir(checker, acados, "acados tree")
    _existing_file(checker, acados / "include/acados_c/ocp_nlp_interface.h", "acados headers")
    for library in ("libacados.so", "libhpipm.so"):
        _existing_file(checker, acados / "lib" / library, library)
    blasfeo = list((acados / "lib").glob("libblasfeo.so*")) if (acados / "lib").is_dir() else []
    if blasfeo:
        checker.ok(f"BLASFEO library: {blasfeo[0]}")
    else:
        checker.fail(f"BLASFEO library missing under {acados / 'lib'}")

    generated = root / "generated/quadrotor_nmpc"
    _existing_file(checker, generated / "acados_solver_ocp_quadrotor_nmpc_1c2d851e.h", "generated solver header")
    _existing_file(checker, generated / "libacados_ocp_solver_ocp_quadrotor_nmpc_1c2d851e.so", "generated solver library")
    checker.python_module("numpy", required=False)
    checker.python_module("yaml", required=False)
    checker.python_module("casadi", required=args.require_generator)
    checker.python_module("acados_template", required=args.require_generator)

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if str(acados / "lib") in ld_library_path:
        checker.ok("LD_LIBRARY_PATH contains acados/lib")
    else:
        checker.warn("LD_LIBRARY_PATH does not explicitly contain acados/lib; RPATH may still be sufficient")

    if args.executable:
        executable = args.executable.resolve()
        _existing_file(checker, executable, "NMPC executable")
        if executable.is_file() and shutil.which("ldd"):
            result = checker.run(["ldd", str(executable)])
            missing = [line.strip() for line in (result.stdout if result else "").splitlines() if "not found" in line]
            if missing:
                checker.fail("executable has unresolved shared libraries: " + "; ".join(missing))
            else:
                checker.ok("NMPC executable shared-library dependencies resolved")

    print(f"\nresult: {len(checker.failures)} failure(s), {len(checker.warnings)} warning(s)")
    if checker.failures:
        print("Fix all failures before arming or starting the NMPC node.")
        return 1
    print("Environment is ready for build/runtime smoke testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
