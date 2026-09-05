#!/usr/bin/env python3
"""Run the documented four-case C++ NMPC SITL regression suite.

PX4 SITL and one MicroXRCEAgent are intentionally managed outside this script.
The script owns the C++ NMPC process and the case supervisor, keeps each case's
raw runtime logs under a temporary directory outside the repository, and writes
the machine-readable suite report only under ``background/json`` as required by
``docs/文件整理规则.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
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

DEFAULT_NODE = Path(
    os.environ.get(
        "ESO_NMPC_CPP_NODE",
        str(Path.home() / "px4_ros2_ws/install/eso_nmpc_node/lib/eso_nmpc_node/eso_nmpc_node"),
    )
)
DEFAULT_PARAMS = PROJECT_ROOT / "cpp/eso_nmpc_node/config/eso_nmpc_cpp.yaml"
DEFAULT_LOG_ROOT = Path(os.environ.get("ESO_NMPC_SITL_LOG_DIR", "/tmp/eso_nmpc_sitl_logs"))
CASE_TIMEOUT_S = 300.0
SOLVER_HASH = "1c2d851e"
DEFAULT_REFERENCE_SAMPLE_TIME = 0.01


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_solver_hash(path: Path) -> str:
    matches = set(re.findall(rb"ocp_quadrotor_nmpc_([0-9a-f]{8})_acados_solve", path.read_bytes()))
    if len(matches) == 1:
        return next(iter(matches)).decode("ascii")
    return SOLVER_HASH


def _yaml_bool(path: Path, key: str, fallback: bool) -> bool:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(true|false)\s*$", path.read_text(), re.MULTILINE)
    if match is None:
        return fallback
    return match.group(1) == "true"


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
    *, command: list[str], environment: Mapping[str, str], log_directory: Path,
    timeout: float,
) -> tuple[dict[str, object], int]:
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "supervisor.log"
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
            # Drain the child pipe continuously, but do not mirror every ROS
            # log line to the runner's stdout.  A burst of DDS reorder
            # warnings can otherwise fill the outer terminal pipe, block this
            # reader, and starve the supervisor's ROS executor.  The complete
            # stream is retained in supervisor.log below.
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


def _write_markdown(
    path: Path, results: dict[str, dict[str, object]], json_path: Path,
    metadata: dict[str, object],
) -> None:
    lines = [
        "# C++ NMPC SITL regression",
        "",
        f"- Disturbance profile: `{metadata.get('disturbance_profile', 'unspecified')}`",
        f"- ESO: `{'enabled' if metadata.get('eso_enabled', True) else 'disabled'}`; "
        f"bandwidth override: `{metadata.get('eso_bandwidth_override_rad_s', 'none')} rad/s`",
        f"- Warm start: `{'enabled' if metadata.get('warm_start', True) else 'disabled'}`",
        f"- Parameter snapshot: `{metadata.get('params_snapshot', 'not recorded')}`",
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


def _write_failure_record(
    path: Path, *, run_directory: Path, json_path: Path,
    results: dict[str, dict[str, object]],
    position_rw: float, velocity_rw: float, eso_enabled: bool,
) -> None:
    failures = [(name, result) for name, result in results.items() if not result.get("success", False)]
    if not failures:
        return
    lines = [
        f"## {run_directory.name}",
        "",
        f"- 结果目录：`{run_directory}`",
        f"- JSON 报告：`{json_path}`",
        f"- 后端：ROS 2 + C++ 单进程 NMPC；ESO：`{'开启' if eso_enabled else '关闭'}`",
        f"- 位置/速度随机游走：`{position_rw}/{velocity_rw}`",
        "- 结论：本轮不得作为 nominal 或 best 基线",
        "",
        "| 用例 | failure reason | landed_disarmed | Position RMSE (m) | Max (m) |",
        "|---|---|---:|---:|---:|",
    ]
    for name, result in failures:
        lines.append(
            f"| {name} | {result.get('reason', 'unknown')} | "
            f"{result.get('landed_disarmed', False)} | "
            f"{float(result.get('tracking_position_rmse_m', float('nan'))):.4f} | "
            f"{float(result.get('tracking_position_max_m', float('nan'))):.4f} |"
        )
    lines.extend(("", "临时节点、监督器和 PX4 ULog 位于结果报告记录的 `/tmp/eso_nmpc_sitl_logs/` 路径；", ""))
    with path.open("a", encoding="utf-8") as stream:
        if path.stat().st_size == 0:
            stream.write("# 四项 C++ 回归失败记录\n\n")
        stream.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--log-directory", type=Path, default=DEFAULT_LOG_ROOT,
        help="临时节点/监督器/ULog 目录；默认在仓库外的 /tmp/eso_nmpc_sitl_logs",
    )
    parser.add_argument("--node-executable", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--acados-source-dir", type=Path,
                        default=Path(os.environ.get("ACADOS_SOURCE_DIR", Path.home() / "acados")))
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--case-timeout", type=float, default=CASE_TIMEOUT_S)
    parser.add_argument(
        "--reference-sample-time", type=float, default=DEFAULT_REFERENCE_SAMPLE_TIME,
        help="reference trajectory step; default matches the generated C++ solver at 0.01 s",
    )
    parser.add_argument(
        "--skip-reference-sample-time-check", action="store_true",
        help="diagnostic only: accept a reference step different from the C++ solver step",
    )
    parser.add_argument("--ros-setup-scripts", action="append", type=Path)
    parser.add_argument("--rmw-implementation", default=DEFAULT_RMW_IMPLEMENTATION)
    parser.add_argument("--skip-service-check", action="store_true")
    parser.add_argument("--skip-params", action="store_true",
                        help="使用 PX4 当前参数；正式运行建议由外层统一参数流程管理")
    parser.add_argument("--best-mode", type=str,
                        help="完成四项且全部通过后，自动更新该仿真模式的 best 历史")
    eso_group = parser.add_mutually_exclusive_group()
    eso_group.add_argument("--disable-eso", action="store_true",
                           help="关闭 C++ 节点 ESO；用于与 ESO 开启的扰动回归对照")
    eso_group.add_argument("--enable-eso", action="store_true",
                           help="启用 C++ 节点 ESO；覆盖节点 YAML 默认值")
    parser.add_argument("--disable-warm-start", action="store_true",
                        help="关闭 C++ Acados warm start；用于冷启动对照")
    parser.add_argument(
        "--disable-rates-output", action="store_true",
        help="diagnostic only: keep solving/logging but suppress PX4 rate setpoints",
    )
    parser.add_argument(
        "--odometry-gap-threshold", type=float,
        help="diagnostic only: override the C++ receive-gap threshold; never use for formal flight",
    )
    parser.add_argument("--eso-bandwidth", type=float,
                        help="临时覆盖 C++ ESO 带宽（rad/s），并写入回归及 best 记录")
    parser.add_argument("--disturbance-profile", default="unspecified",
                        help="记录本轮仿真扰动配置名称，不负责修改 Gazebo 模型")
    parser.add_argument("--cg-bias-m", type=float, default=0.0,
                        help="记录的重心偏置（m），不负责修改 Gazebo 模型")
    parser.add_argument("--wind-x-m-s", type=float, default=0.0,
                        help="记录的 X 向风速（m/s），不负责修改 Gazebo 世界")
    parser.add_argument("--position-bias-rw-std-m-sqrt-s", type=float, default=0.0)
    parser.add_argument("--velocity-bias-rw-std-m-s-sqrt-s", type=float, default=0.0)
    args = parser.parse_args()

    node = args.node_executable.expanduser().resolve()
    params = args.params_file.expanduser().resolve()
    if not node.is_file() or not os.access(node, os.X_OK):
        parser.error(f"C++ NMPC executable not found or not executable: {node}")
    if not params.is_file():
        parser.error(f"C++ parameter file not found: {params}")
    eso_enabled = args.enable_eso or (
        not args.disable_eso and _yaml_bool(params, "eso_enabled", True)
    )
    warm_start_enabled = (
        not args.disable_warm_start and _yaml_bool(params, "warm_start", True)
    )
    if not (args.acados_source_dir / "lib/libacados.so").is_file():
        parser.error(f"acados shared libraries not found under {args.acados_source_dir}")
    if args.position_bias_rw_std_m_sqrt_s < 0.0 or args.velocity_bias_rw_std_m_s_sqrt_s < 0.0:
        parser.error("random-walk diffusion parameters must be non-negative")
    if args.eso_bandwidth is not None and args.eso_bandwidth <= 0.0:
        parser.error("ESO bandwidth must be positive")
    if args.reference_sample_time <= 0.0:
        parser.error("reference-sample-time must be positive")
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
    params_snapshot = output / "cpp_params.yaml"
    shutil.copy2(params, params_snapshot)
    log_root = args.log_directory.expanduser().resolve() / output.name
    selected = set(args.cases) if args.cases else None
    cases = [case for case in CASES if selected is None or case.name in selected]
    results: dict[str, dict[str, object]] = {}
    px4_log_root = PROJECT_ROOT.parent / "apx"

    try:
        for case in cases:
            case_dir = output / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            case_log_dir = log_root / case.name
            print(f"\n=== C++ NMPC regression: {case.name} ===", flush=True)
            case_log_dir.mkdir(parents=True, exist_ok=True)
            node_log = case_log_dir / "cpp_node.log"
            node_cmd = [
                str(node), "--ros-args", "--params-file", str(params),
                "-p", f"flight_log_path:={case_dir / 'nmpc_flight.csv'}",
                "-p", f"timing_log_path:={case_dir / 'nmpc_timing.csv'}",
                "-p", "control_enabled_at_start:=false",
            ]
            if args.skip_reference_sample_time_check:
                node_cmd.extend(["-p", "enforce_reference_sample_time:=false"])
            if args.disable_eso:
                node_cmd.extend(["-p", "eso_enabled:=false"])
            if args.enable_eso:
                node_cmd.extend(["-p", "eso_enabled:=true"])
            if args.disable_warm_start:
                node_cmd.extend(["-p", "warm_start:=false"])
            if args.disable_rates_output:
                node_cmd.extend(["-p", "publish_rates_enabled:=false"])
            if args.odometry_gap_threshold is not None:
                if args.odometry_gap_threshold <= 0.0:
                    parser.error("odometry-gap-threshold must be positive")
                node_cmd.extend(["-p", f"odometry_timestamp_gap_threshold:={args.odometry_gap_threshold}"])
            if args.eso_bandwidth is not None:
                node_cmd.extend(["-p", f"eso_bandwidth:={args.eso_bandwidth}"])
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
                        "--reference-sample-time", str(args.reference_sample_time),
                    ]
                    if case.name == "point_1m":
                        supervisor_cmd += ["--step-dwell", "2.0", "--radius", "1.0"]
                    if case.name in ("circle", "figure8"):
                        supervisor_cmd += ["--radius", "2.0", "--speed", "1.0"]
                    result, _ = _run_supervisor(
                        command=supervisor_cmd, environment=environment,
                        log_directory=case_log_dir, timeout=args.case_timeout,
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
                shutil.copy2(ulog, case_log_dir / ulog.name)
                result["px4_ulog"] = str((case_log_dir / ulog.name).resolve())
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

    case_results = {k: v for k, v in results.items() if k in {case.name for case in CASES}}
    metadata: dict[str, object] = {
        "backend": "ROS 2 + C++ single-process NMPC",
        "solver_hash": _detect_solver_hash(node),
        "node_sha256": _file_sha256(node),
        "control_period_s": 0.01,
        "reference_sample_time_s": args.reference_sample_time,
        "reference_sample_time_check_enforced": not args.skip_reference_sample_time_check,
        "horizon_steps": 30,
        "disturbance_profile": args.disturbance_profile,
        "disturbances": {
            "cg_bias_m": args.cg_bias_m,
            "wind_x_m_s": args.wind_x_m_s,
            "position_random_walk_std_m_sqrt_s": args.position_bias_rw_std_m_sqrt_s,
            "velocity_random_walk_std_m_s_sqrt_s": args.velocity_bias_rw_std_m_s_sqrt_s,
        },
        "eso_enabled": eso_enabled,
        "warm_start": warm_start_enabled,
        "eso_bandwidth_override_rad_s": args.eso_bandwidth,
        "eso_configured_defaults": {
            "bandwidth_rad_s": 2.5,
            "activation_delay_s": 3.0,
            "clamp_m_s2": 1.0,
            "innovation_limit_m_s": 0.5,
        },
        "params_snapshot": str(params_snapshot),
        "node_executable": str(node),
    }
    results["_metadata"] = metadata
    combined = _write_combined_trajectory_plot(output, case_results)
    if combined is not None:
        results["_suite"] = {"trajectory_suite_long": str(combined)}
    json_path = PROJECT_ROOT / "background/json" / f"cpp_regression_{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output / "suite_summary.md", case_results, json_path, metadata)
    _write_failure_record(
        PROJECT_ROOT / "docs/四项回归失败记录.md",
        run_directory=output,
        json_path=json_path,
        results=case_results,
        position_rw=args.position_bias_rw_std_m_sqrt_s,
        velocity_rw=args.velocity_bias_rw_std_m_s_sqrt_s,
        eso_enabled=eso_enabled,
    )
    if args.best_mode and all(
        result.get("success", False) for result in case_results.values()
    ):
        best_command = [
            sys.executable,
            str(PROJECT_ROOT / "integration/manage_best_sitl_history.py"),
            "update",
            "--mode", args.best_mode,
            "--report", str(json_path),
        ]
        subprocess.run(best_command, cwd=PROJECT_ROOT, check=True)
    print(f"C++ regression report: {output / 'suite_summary.md'}", flush=True)
    print(f"C++ regression JSON: {json_path}", flush=True)
    return 0 if all(result.get("success", False) for result in case_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
