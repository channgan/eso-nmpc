#!/usr/bin/env python3
"""Run the documented four-case C++ NMPC SITL regression suite.

PX4 SITL and one MicroXRCEAgent are intentionally managed outside this script.
The script owns the C++ NMPC process and the case supervisor, keeps each case's
raw C++ logs together, and writes the machine-readable suite report only under
``background/json`` as required by ``docs/文件整理规则.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.run_sitl_regression import (  # noqa: E402
    CASES,
    DEFAULT_RMW_IMPLEMENTATION,
    DEFAULT_ROS_SETUP_SCRIPTS,
    _write_combined_trajectory_plot,
    build_child_environment,
    check_services,
)
from integration.mavlink_params import (  # noqa: E402
    DEFAULT_PARAMETERS,
    GuardedParameter,
    ParamGuard,
    ParamGuardError,
)

DEFAULT_NODE = PROJECT_ROOT / "install_cpp/eso_nmpc_node/lib/eso_nmpc_node/eso_nmpc_node"
DEFAULT_PARAMS = PROJECT_ROOT / "cpp/eso_nmpc_node/config/eso_nmpc_cpp.yaml"
CASE_TIMEOUT_S = 300.0


def _terminate(process: subprocess.Popen[str] | None, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _latest_ulog(search_root: Path, started_at: float) -> Path | None:
    candidates = [
        path for path in search_root.rglob("*.ulg")
        if path.is_file() and path.stat().st_mtime >= started_at - 2.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _timing_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"timing_samples": 0, "timing_error": f"missing {path.name}"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, object] = {"timing_samples": len(rows)}
    for column, label in (
        ("rx_to_pub_ms", "rx_to_pub"),
        ("acados_solve_wall_ms", "acados_solve"),
        ("time_qp_ms", "qp"),
    ):
        values = np.asarray(
            [float(row[column]) for row in rows if row.get(column, "") not in ("", "nan")],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size:
            result[f"{label}_median_p95_p99_max_ms"] = [
                float(np.percentile(values, q)) for q in (50, 95, 99)
            ] + [float(np.max(values))]
    success_values = [int(row.get("solve_success", "0")) for row in rows]
    result["solve_success_count"] = int(sum(success_values))
    result["solve_failure_count"] = int(len(success_values) - sum(success_values))
    return result


def _run_supervisor(
    *, command: list[str], environment: Mapping[str, str], case_directory: Path,
    timeout: float,
) -> tuple[dict[str, object], int]:
    log_path = case_directory / "supervisor.log"
    result: dict[str, object] | None = None
    process = subprocess.Popen(
        command, cwd=PROJECT_ROOT, env=dict(environment), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
    )
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    with log_path.open("w", encoding="utf-8") as stream:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                break
            line = process.stdout.readline()
            if not line:
                break
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
            if line.startswith("CPP_NMPC_SITL_RESULT="):
                result = json.loads(line.split("=", 1)[1])
    if process.poll() is None:
        _terminate(process)
    return_code = process.wait()
    if result is None:
        result = {"success": False, "reason": "missing CPP_NMPC_SITL_RESULT"}
    result["return_code"] = return_code
    result["supervisor_log"] = str(log_path.resolve())
    return result, return_code


def _write_markdown(path: Path, results: dict[str, dict[str, object]], json_path: Path) -> None:
    lines = [
        "# C++ NMPC SITL regression",
        "",
        "| Case | Result | Position RMSE (m) | Position max (m) | rx→pub Median/P95/P99 (ms) | Solve Median/P95/P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        result = results.get(case.name)
        if result is None:
            continue
        rx = result.get("rx_to_pub_median_p95_p99_max_ms", [float("nan")] * 4)
        solve = result.get("acados_solve_median_p95_p99_max_ms", [float("nan")] * 4)
        lines.append(
            f"| {case.name} | {'PASS' if result.get('success') else 'FAIL'} | "
            f"{float(result.get('tracking_position_rmse_m', float('nan'))):.4f} | "
            f"{float(result.get('tracking_position_max_m', float('nan'))):.4f} | "
            f"{rx[0]:.2f}/{rx[1]:.2f}/{rx[2]:.2f} | "
            f"{solve[0]:.2f}/{solve[1]:.2f}/{solve[2]:.2f} |"
        )
    lines.extend(("", f"Machine-readable report: `{json_path}`", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--node-executable", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--acados-source-dir", type=Path,
                        default=Path(os.environ.get("ACADOS_SOURCE_DIR", Path.home() / "acados")))
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--case-timeout", type=float, default=CASE_TIMEOUT_S)
    parser.add_argument("--ros-setup-scripts", action="append", type=Path)
    parser.add_argument("--rmw-implementation", default=DEFAULT_RMW_IMPLEMENTATION)
    parser.add_argument("--skip-service-check", action="store_true")
    parser.add_argument("--skip-params", action="store_true",
                        help="使用 PX4 当前参数；正式运行建议由外层统一参数流程管理")
    parser.add_argument("--position-bias-rw-std-m-sqrt-s", type=float, default=0.0)
    parser.add_argument("--velocity-bias-rw-std-m-s-sqrt-s", type=float, default=0.0)
    args = parser.parse_args()

    node = args.node_executable.expanduser().resolve()
    params = args.params_file.expanduser().resolve()
    if not node.is_file() or not os.access(node, os.X_OK):
        parser.error(f"C++ NMPC executable not found or not executable: {node}")
    if not params.is_file():
        parser.error(f"C++ parameter file not found: {params}")
    if not (args.acados_source_dir / "lib/libacados.so").is_file():
        parser.error(f"acados shared libraries not found under {args.acados_source_dir}")
    if args.position_bias_rw_std_m_sqrt_s < 0.0 or args.velocity_bias_rw_std_m_s_sqrt_s < 0.0:
        parser.error("random-walk diffusion parameters must be non-negative")
    scripts = tuple(path.expanduser() for path in args.ros_setup_scripts) \
        if args.ros_setup_scripts is not None else DEFAULT_ROS_SETUP_SCRIPTS
    try:
        environment = build_child_environment(
            acados_source_dir=args.acados_source_dir.resolve(),
            ros_setup_scripts=scripts,
            rmw_implementation=args.rmw_implementation,
        )
    except RuntimeError as error:
        parser.error(str(error))

    if not args.skip_service_check:
        # The C++ runner does not open a MAVLink parameter connection.  The
        # PX4 process and simulator checks are sufficient here; the DDS node
        # startup below is the definitive end-to-end probe.
        checks = check_services(px4_ok=True, px4_heartbeat_error=None)
        missing = [f"{name}: {note}" for name, (ok, note) in checks.items() if not ok]
        if missing:
            parser.error("SITL services unavailable:\n" + "\n".join(missing))

    guard: ParamGuard | None = None
    if not args.skip_params:
        parameters = DEFAULT_PARAMETERS + (
            GuardedParameter("SIM_GZ_ODOM_RW_P", args.position_bias_rw_std_m_sqrt_s,
                             "external odometry position random walk"),
            GuardedParameter("SIM_GZ_ODOM_RW_V", args.velocity_bias_rw_std_m_s_sqrt_s,
                             "external odometry velocity random walk"),
        )
        try:
            guard = ParamGuard(parameters=parameters)
            guard.__enter__()
        except ParamGuardError as error:
            parser.error(f"PX4 parameter guard failed: {error}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_directory or PROJECT_ROOT / "background/sitl_regression_cpp" / run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.cases) if args.cases else None
    cases = [case for case in CASES if selected is None or case.name in selected]
    results: dict[str, dict[str, object]] = {}
    px4_log_root = PROJECT_ROOT.parent / "apx"

    try:
        for case in cases:
            case_dir = output / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== C++ NMPC regression: {case.name} ===", flush=True)
            node_log = case_dir / "cpp_node.log"
            node_cmd = [
                str(node), "--ros-args", "--params-file", str(params),
                "-p", f"flight_log_path:={case_dir / 'nmpc_flight.csv'}",
                "-p", f"timing_log_path:={case_dir / 'nmpc_timing.csv'}",
                "-p", "control_enabled_at_start:=false",
            ]
            started_at = time.time()
            with node_log.open("w", encoding="utf-8") as node_stream:
                node_process = subprocess.Popen(
                    node_cmd, cwd=PROJECT_ROOT, env=environment, stdout=node_stream,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
            try:
                time.sleep(2.0)
                if node_process.poll() is not None:
                    result = {"success": False, "reason": "C++ NMPC node exited during startup", "return_code": node_process.returncode}
                else:
                    supervisor_cmd = [
                        sys.executable, str(PROJECT_ROOT / "integration/run_cpp_sitl_hover.py"),
                        "--trajectory", "point_1m" if case.name == "point_1m" else case.name,
                        "--output-directory", str(case_dir), "--skip-params",
                    ]
                    if case.name in ("circle", "figure8"):
                        supervisor_cmd += ["--radius", "2.0", "--speed", "1.0"]
                    result, _ = _run_supervisor(
                        command=supervisor_cmd, environment=environment,
                        case_directory=case_dir, timeout=args.case_timeout,
                    )
            finally:
                _terminate(node_process)

            result.update(_timing_summary(case_dir / "nmpc_timing.csv"))
            plot_command = [sys.executable, str(PROJECT_ROOT / "integration/plot_cpp_run.py"), str(case_dir)]
            subprocess.run(plot_command, cwd=PROJECT_ROOT, env=environment, check=False,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if (case_dir / "trajectory.png").is_file():
                result["trajectory_plot"] = str((case_dir / "trajectory.png").resolve())
            if (case_dir / "controller_timing.png").is_file():
                result["controller_timing_plot"] = str((case_dir / "controller_timing.png").resolve())
            ulog = _latest_ulog(px4_log_root, started_at)
            if ulog is not None:
                shutil.copy2(ulog, case_dir / ulog.name)
                result["px4_ulog"] = str((case_dir / ulog.name).resolve())
            result["case"] = case.name
            result["cpp_node_log"] = str(node_log.resolve())
            result["case_directory"] = str(case_dir.resolve())
            results[case.name] = result
            # Supervisor's JSON is a transient implementation artifact; the suite
            # report below is the single retained machine-readable source.
            (case_dir / "summary.json").unlink(missing_ok=True)
            if not result.get("success", False):
                print(f"C++ regression case failed: {case.name}: {result.get('reason')}", file=sys.stderr)
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)

    combined = _write_combined_trajectory_plot(output, results)
    if combined is not None:
        results["_suite"] = {"trajectory_suite_long": str(combined)}
    json_path = PROJECT_ROOT / "background/json" / f"cpp_regression_{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output / "suite_summary.md", {k: v for k, v in results.items() if k != "_suite"}, json_path)
    print(f"C++ regression report: {output / 'suite_summary.md'}", flush=True)
    print(f"C++ regression JSON: {json_path}", flush=True)
    return 0 if all(result.get("success", False) for name, result in results.items() if name != "_suite") else 1


if __name__ == "__main__":
    raise SystemExit(main())
