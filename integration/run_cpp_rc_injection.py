#!/usr/bin/env python3
"""Run the C++ RC-NMPC injection test and archive its complete flight log."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.mavlink_params import DEFAULT_PARAMETERS, ParamGuard, ParamGuardError
from integration.run_cpp_regression import (
    DEFAULT_LOG_ROOT,
    DEFAULT_NODE,
    DEFAULT_PARAMS,
    DEFAULT_RMW_IMPLEMENTATION,
    DEFAULT_ROS_SETUP_SCRIPTS,
    SOLVER_HASH,
    _latest_ulog,
    _terminate,
    _timing_summary,
    build_child_environment,
    check_services,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--log-directory", type=Path, default=DEFAULT_LOG_ROOT,
                        help="临时节点/注入器/监督器/ULog 目录")
    parser.add_argument("--node-executable", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--acados-source-dir", type=Path,
                        default=Path(os.environ.get("ACADOS_SOURCE_DIR", Path.home() / "acados")))
    parser.add_argument("--rc-aux-channel", type=int, default=6,
                        help="C++ node/injector AUX channel; default matches deployment AUX6")
    parser.add_argument("--drop-after", type=float, default=-1.0)
    parser.add_argument("--case-timeout", type=float, default=180.0)
    parser.add_argument("--ros-setup-scripts", action="append", type=Path)
    parser.add_argument("--rmw-implementation", default=DEFAULT_RMW_IMPLEMENTATION)
    parser.add_argument("--skip-service-check", action="store_true")
    parser.add_argument("--skip-params", action="store_true")
    args = parser.parse_args()
    node = args.node_executable.expanduser().resolve()
    params = args.params_file.expanduser().resolve()
    if not node.is_file() or not os.access(node, os.X_OK):
        parser.error(f"C++ NMPC executable not found: {node}")
    if not params.is_file():
        parser.error(f"C++ parameter file not found: {params}")
    if args.rc_aux_channel < 1 or args.rc_aux_channel > 6:
        parser.error("rc-aux-channel must be in [1, 6]")
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
        checks = check_services(px4_ok=True, px4_heartbeat_error=None)
        missing = [f"{name}: {note}" for name, (ok, note) in checks.items() if not ok]
        if missing:
            parser.error("SITL services unavailable:\n" + "\n".join(missing))
    guard = None
    if not args.skip_params:
        try:
            guard = ParamGuard(parameters=DEFAULT_PARAMETERS)
            guard.__enter__()
        except ParamGuardError as error:
            parser.error(f"PX4 parameter guard failed: {error}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_directory or PROJECT_ROOT / "background/sitl_regression_cpp" /
              f"rc_injection_{run_id}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    params_snapshot = output / "cpp_params.yaml"
    shutil.copy2(params, params_snapshot)
    log_directory = args.log_directory.expanduser().resolve() / output.name
    log_directory.mkdir(parents=True, exist_ok=True)
    node_log = log_directory / "cpp_node.log"
    injector_log = log_directory / "rc_injector.log"
    supervisor_log = log_directory / "supervisor.log"
    flight_log = output / "nmpc_flight.csv"
    timing_log = output / "nmpc_timing.csv"
    px4_log_root = PROJECT_ROOT.parent / "apx"
    node_cmd = [
        str(node), "--ros-args", "--params-file", str(params),
        "-p", f"flight_log_path:={flight_log}",
        "-p", f"timing_log_path:={timing_log}",
        "-p", f"rc_aux_channel:={args.rc_aux_channel}",
        "-p", "manual_control_topic:=/nmpc/test/manual_control_setpoint",
        "-p", "control_enabled_at_start:=false",
    ]
    supervisor_cmd = [
        sys.executable, str(PROJECT_ROOT / "integration/run_cpp_sitl_hover.py"),
        "--trajectory", "hover", "--output-directory", str(output), "--skip-params",
        "--safety-drift-limit", "5.0",
    ]
    if args.drop_after >= 0.0:
        supervisor_cmd.append("--expect-rc-timeout-fallback")
    injector_cmd = [
        sys.executable, str(PROJECT_ROOT / "integration/inject_rc_sitl.py"),
        "--aux-channel", str(args.rc_aux_channel),
        "--topic", "/nmpc/test/manual_control_setpoint",
    ]
    if args.drop_after >= 0.0:
        injector_cmd += ["--drop-after", str(args.drop_after)]
    started_at = time.time()
    node_process = injector_process = supervisor_process = None
    result: dict[str, object] = {"success": False, "reason": "not started"}
    try:
        with node_log.open("w", encoding="utf-8") as node_stream:
            node_process = subprocess.Popen(
                node_cmd, cwd=PROJECT_ROOT, env=environment, stdout=node_stream,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
        time.sleep(2.0)
        if node_process.poll() is not None:
            result = {"success": False, "reason": "C++ NMPC node exited during startup"}
        else:
            with injector_log.open("w", encoding="utf-8") as injector_stream:
                injector_process = subprocess.Popen(
                    injector_cmd, cwd=PROJECT_ROOT, env=environment, stdout=injector_stream,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
            with supervisor_log.open("w", encoding="utf-8") as supervisor_stream:
                supervisor_process = subprocess.Popen(
                    supervisor_cmd, cwd=PROJECT_ROOT, env=environment, stdout=supervisor_stream,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
            try:
                supervisor_process.wait(timeout=args.case_timeout)
            except subprocess.TimeoutExpired:
                result = {"success": False, "reason": f"RC case timeout after {args.case_timeout:g}s"}
            supervisor_result = None
            if supervisor_log.is_file():
                for line in supervisor_log.read_text(encoding="utf-8").splitlines():
                    if line.startswith("CPP_NMPC_SITL_RESULT="):
                        supervisor_result = json.loads(line.split("=", 1)[1])
            if supervisor_result is None:
                result = {"success": False, "reason": "missing CPP_NMPC_SITL_RESULT"}
            else:
                supervisor_reason = supervisor_result.get("reason", "unknown")
                supervisor_safe_complete = bool(
                    supervisor_result.get("landed_disarmed", False) and
                    supervisor_reason == "trajectory completed"
                )
                result = {
                    # The supervisor's hover RMSE is intentionally not an RC
                    # acceptance criterion: RC-NMPC is expected to move away
                    # from the external hover reference.  Keep that RMSE as a
                    # diagnostic field, but judge the RC case by safe
                    # completion and the explicit RC/solver checks below.
                    "success": supervisor_safe_complete,
                    "supervisor_success": bool(supervisor_result.get("success", False)),
                    "supervisor_safe_complete": supervisor_safe_complete,
                    "reason": supervisor_reason,
                    "landed_disarmed": supervisor_result.get("landed_disarmed", False),
                    "supervisor_position_rmse_m": supervisor_result.get("tracking_position_rmse_m"),
                    "rc_timeout_seen": supervisor_result.get("rc_timeout_seen", False),
                    "rc_timeout_position_seen": supervisor_result.get("rc_timeout_position_seen", False),
                    "rc_timeout_fallback_seen": supervisor_result.get("rc_timeout_fallback_seen", False),
                }
    finally:
        _terminate(supervisor_process)
        _terminate(injector_process)
        _terminate(node_process)
        if guard is not None:
            guard.__exit__(None, None, None)

    result.update(_timing_summary(timing_log))
    rc_active_count = 0
    if flight_log.is_file():
        with flight_log.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rc_active_count = sum(int(row.get("rc_mode_active", "0")) for row in rows)
    result["rc_active_sample_count"] = rc_active_count
    result["solve_failure_count"] = int(result.get("solve_failure_count", 0))
    result["success"] = bool(
        result.get("supervisor_safe_complete", False) and
        rc_active_count > 0 and
        result["solve_failure_count"] == 0
    )
    if rc_active_count == 0:
        result["reason"] = "RC-NMPC was not activated (no rc_mode_active samples)"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "integration/plot_cpp_run.py"), str(output),
         "--source", "nmpc_flight"],
        cwd=PROJECT_ROOT, env=environment, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if (output / "trajectory.png").is_file():
        result["trajectory_plot"] = str((output / "trajectory.png").resolve())
    if (output / "controller_timing.png").is_file():
        result["controller_timing_plot"] = str((output / "controller_timing.png").resolve())
    ulog = _latest_ulog(px4_log_root, started_at)
    if ulog is not None:
        destination = log_directory / ulog.name
        shutil.copy2(ulog, destination)
        result["px4_ulog"] = str(destination.resolve())
    result["backend"] = "cpp_rc_injection"
    result["rc_aux_channel"] = args.rc_aux_channel
    result["run_configuration"] = {
        "backend": "ROS 2 + C++ single-process NMPC",
        "solver_hash": SOLVER_HASH,
        "control_period_s": 0.01,
        "horizon_steps": 30,
        "rc_aux_channel": args.rc_aux_channel,
        "manual_control_topic": "/nmpc/test/manual_control_setpoint",
        "rc_timeout_s": 0.50,
        "drop_after_s": args.drop_after if args.drop_after >= 0.0 else None,
        "params_snapshot": str(params_snapshot),
        "node_executable": str(node),
        "rc_timeout_action": "latch_nmpc_output_off_and_request_px4_auto_loiter",
        "plot_source": "nmpc_flight.csv",
    }
    result["cpp_node_log"] = str(node_log.resolve())
    result["rc_injector_log"] = str(injector_log.resolve())
    result["case_directory"] = str(output)
    json_path = PROJECT_ROOT / "background/json" / f"cpp_rc_injection_{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(
        "# C++ RC-NMPC injection\n\n"
        f"- Result: {'PASS' if result.get('success') else 'FAIL'}\n"
        f"- AUX channel: {args.rc_aux_channel}\n"
        f"- RC timeout test: {'drop after ' + str(args.drop_after) + ' s' if args.drop_after >= 0.0 else 'normal stream'}\n"
        f"- RC timeout fallback: {result.get('rc_timeout_fallback_seen', 'not applicable')}\n"
        f"- RC active samples: {result.get('rc_active_sample_count', 0)}\n"
        f"- Solve failures: {result.get('solve_failure_count', 0)}\n"
        "- Supervisor hover RMSE: diagnostic only; not an RC acceptance criterion\n"
        f"- Solve Median/P95/P99/Max (ms): {result.get('acados_solve_median_p95_p99_max_ms', [])}\n"
        f"- rx→pub Median/P95/P99/Max (ms): {result.get('rx_to_pub_median_p95_p99_max_ms', [])}\n"
        f"- Parameter snapshot: `{params_snapshot}`\n"
        f"- Temporary logs: `{log_directory}`\n"
        "- Plot source: `nmpc_flight.csv`\n"
        f"- JSON: `{json_path}`\n",
        encoding="utf-8",
    )
    print("CPP_RC_NMPC_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
