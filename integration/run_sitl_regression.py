#!/usr/bin/env python3
"""Run and archive the mandatory four-case NMPC SITL baseline suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROS_SETUP_SCRIPTS: tuple[Path, ...] = (
    Path("/opt/ros/humble/setup.bash"),
    Path("/home/cy/px4_ros2_ws/install/setup.bash"),
)
DEFAULT_RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
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
    arguments = parser.parse_args()
    if arguments.altitude <= 0.0:
        parser.error("altitude must be positive")
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = arguments.output_directory or PROJECT_ROOT / "background/baseline_runs" / run_id
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = set(arguments.cases) if arguments.cases else None
    cases = [case for case in CASES if selected is None or case.name in selected]

    results: dict[str, dict[str, object]] = {}
    suite_success = True
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
        print(f"\n=== NMPC baseline: {case.name} ({case.reference_source}) ===", flush=True)
        result: dict[str, object] | None = None
        log_path = case_directory / "run.log"
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=PROJECT_ROOT, env=child_environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
                if line.startswith("NMPC_SITL_RESULT="):
                    result = json.loads(line.split("=", 1)[1])
            return_code = process.wait()

        if result is None:
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
        results[case.name] = result
        _write_suite_reports(output_directory, results)
        if return_code != 0 or not result.get("success", False):
            suite_success = False
            print(f"Baseline case failed: {case.name}; continuing suite", file=sys.stderr)

    markdown_path = output_directory / "suite_summary.md"
    print("\n" + markdown_path.read_text(encoding="utf-8"), flush=True)
    print(f"Baseline reports written to {output_directory}", flush=True)
    return 0 if suite_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
