#!/usr/bin/env python3
"""Run the C++ ESO/no-ESO disturbance cross-validation matrix.

Each condition/ESO/trajectory combination gets a fresh MicroXRCE-DDS Agent,
PX4, and Gazebo process.  The C++
regression runner owns the C++ NMPC node and the Python flight supervisor;
this file only owns the external SITL lifecycle and the Gazebo disturbance
configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PX4 = Path("/home/cy/apx")
BUILD = PX4 / "build/px4_sitl_default"
MODEL = PX4 / "Tools/simulation/gz/models/x500_base/model.sdf"
WORLD = PX4 / "Tools/simulation/gz/worlds/default.sdf"
NODE = Path("/home/cy/px4_ros2_ws/build/eso_nmpc_node/eso_nmpc_node")
PARAMS = ROOT / "cpp/eso_nmpc_node/config/eso_nmpc_cpp.yaml"
TRAJECTORIES = ("hover", "point_1m", "circle", "figure8")

from integration.run_eso_matrix_supervisor import (  # noqa: E402
    _cleanup_sitl,
    _udp_listener_on,
    restart_sitl,
)


CONDITIONS = {
    "nominal": {
        "profile": "nominal_no_disturbance",
        "cg_bias_m": 0.0,
        "wind_x_m_s": 0.0,
        "position_rw": 0.0,
        "velocity_rw": 0.0,
    },
    "cg": {
        "profile": "single_cg_bias",
        "cg_bias_m": 0.02,
        "wind_x_m_s": 0.0,
        "position_rw": 0.0,
        "velocity_rw": 0.0,
    },
    "wind": {
        "profile": "single_wind",
        "cg_bias_m": 0.0,
        "wind_x_m_s": 0.5,
        "position_rw": 0.0,
        "velocity_rw": 0.0,
    },
    "noise": {
        "profile": "single_odom_random_walk",
        "cg_bias_m": 0.0,
        "wind_x_m_s": 0.0,
        "position_rw": 0.002,
        "velocity_rw": 0.005,
    },
    "cg_wind": {
        "profile": "pair_cg_wind",
        "cg_bias_m": 0.02,
        "wind_x_m_s": 0.5,
        "position_rw": 0.0,
        "velocity_rw": 0.0,
    },
    "cg_noise": {
        "profile": "pair_cg_odom_random_walk",
        "cg_bias_m": 0.02,
        "wind_x_m_s": 0.0,
        "position_rw": 0.002,
        "velocity_rw": 0.005,
    },
    "wind_noise": {
        "profile": "pair_wind_odom_random_walk",
        "cg_bias_m": 0.0,
        "wind_x_m_s": 0.5,
        "position_rw": 0.002,
        "velocity_rw": 0.005,
    },
    "three_disturbances": {
        "profile": "three_disturbances_all_on",
        "cg_bias_m": 0.02,
        "wind_x_m_s": 0.5,
        "position_rw": 0.002,
        "velocity_rw": 0.005,
    },
}

CONDITION_LABELS = {
    "nominal": "无扰动",
    "cg": "单项：重心偏置",
    "wind": "单项：X 向风",
    "noise": "单项：里程计随机游走",
    "cg_wind": "两项：重心偏置 + X 向风",
    "cg_noise": "两项：重心偏置 + 随机游走",
    "wind_noise": "两项：X 向风 + 随机游走",
    "three_disturbances": "三项全开",
}


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3.0)


def _controller_pids() -> list[int]:
    """Find only this project's C++ runner, supervisor, and node processes."""
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    matches = (
        "/home/cy/eso_nmpc/integration/run_cpp_regression.py",
        "/home/cy/eso_nmpc/integration/run_cpp_sitl_hover.py",
        "/home/cy/px4_ros2_ws/build/eso_nmpc_node/eso_nmpc_node",
    )
    pids: list[int] = []
    for line in listing:
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid != os.getpid() and any(marker in fields[1] for marker in matches):
            pids.append(pid)
    return pids


def _cleanup_controller_processes() -> None:
    """Prevent an interrupted prior case from publishing into the next SITL."""
    pids = _controller_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        alive: list[int] = []
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


def _start_agent(log_path: Path) -> tuple[subprocess.Popen[str] | None, bool]:
    if _udp_listener_on(8888):
        return None, False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            ["MicroXRCEAgent", "udp4", "-p", "8888"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        stream.close()
        raise
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stream.close()
            raise RuntimeError("MicroXRCEAgent exited during startup")
        if _udp_listener_on(8888):
            # Keep the file descriptor owned by the child process; closing the
            # parent copy avoids leaking one descriptor for every run.
            stream.close()
            return process, True
        time.sleep(0.2)
    _stop_process(process)
    stream.close()
    raise RuntimeError("MicroXRCEAgent did not bind UDP 8888 within 15 s")


def _agent_pids() -> list[int]:
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
        if pid != os.getpid() and "MicroXRCEAgent" in command and "-p 8888" in command:
            pids.append(pid)
    return pids


def _stop_existing_agent() -> None:
    """Drop an exact old Agent session before starting the next case."""
    pids = _agent_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        alive: list[int] = []
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


def _configure_sitl(condition: dict[str, float | str]) -> None:
    """Apply the physical CG and wind for the next fresh PX4/Gazebo case."""
    model = MODEL.read_text(encoding="utf-8")
    cg = float(condition["cg_bias_m"])
    marker = "<inertial>\n        <pose>"
    if marker not in model:
        raise RuntimeError(f"cannot locate x500 inertial pose in {MODEL}")
    start = model.index(marker) + len(marker)
    end = model.index("</pose>", start)
    values = model[start:end].strip().split()
    if len(values) != 6:
        raise RuntimeError(f"unexpected inertial pose in {MODEL}: {model[start:end]!r}")
    values[0] = f"{cg:.12g}"
    model = model[:start] + " ".join(values) + model[end:]
    MODEL.write_text(model, encoding="utf-8")

    world = WORLD.read_text(encoding="utf-8")
    start_tag = "    <wind>"
    start = world.find(start_tag)
    if start < 0:
        raise RuntimeError(f"cannot locate wind block in {WORLD}")
    end = world.find("    </wind>", start)
    if end < 0:
        raise RuntimeError(f"cannot close wind block in {WORLD}")
    end += len("    </wind>")
    wind = float(condition["wind_x_m_s"])
    block = (
        "    <wind>\n"
        f"      <linear_velocity>{wind:.12g} 0 0</linear_velocity>\n"
        "    </wind>"
    )
    WORLD.write_text(world[:start] + block + world[end:], encoding="utf-8")


def _case_command(
    *,
    output: Path,
    log_directory: Path,
    case: str,
    condition_name: str,
    condition: dict[str, float | str],
    eso_on: bool,
    case_timeout: float,
) -> list[str]:
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "integration/run_cpp_regression.py"),
        "--output-directory", str(output),
        "--log-directory", str(log_directory),
        "--node-executable", str(NODE),
        "--params-file", str(PARAMS),
        "--cases", case,
        "--case-timeout", str(case_timeout),
        "--reference-sample-time", "0.01",
        "--disturbance-profile", str(condition["profile"]),
        "--cg-bias-m", str(condition["cg_bias_m"]),
        "--wind-x-m-s", str(condition["wind_x_m_s"]),
        "--position-bias-rw-std-m-sqrt-s", str(condition["position_rw"]),
        "--velocity-bias-rw-std-m-s-sqrt-s", str(condition["velocity_rw"]),
    ]
    if not eso_on:
        command.append("--disable-eso")
    return command


def _read_case_result(
    output: Path, case: str, runner_log: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Read the runner's global JSON report named in its captured output."""
    report_path: Path | None = None
    if runner_log.is_file():
        for line in runner_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("C++ regression JSON:"):
                report_path = Path(line.split(":", 1)[1].strip())
    if report_path is None or not report_path.is_file():
        return (
            {"success": False, "reason": "C++ runner produced no background/json report"},
            {},
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = dict(report.get(case, {}))
    if not result:
        result = {"success": False, "reason": f"suite report has no case {case}"}
    return result, dict(report.get("_metadata", {}))


def _write_cross_plot(root: Path, results: dict[str, dict[str, object]]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = [
        (f"{CONDITION_LABELS[condition]} / ESO {mode}", condition, mode)
        for condition in CONDITIONS
        for mode in ("off", "on")
        if any(key.startswith(f"{condition}/{mode}/") for key in results)
    ]
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(24, max(22, 5.5 * len(rows))), squeeze=False
    )
    for row_index, (label, condition, mode) in enumerate(rows):
        for col_index, case in enumerate(TRAJECTORIES):
            axis = axes[row_index, col_index]
            item = results.get(f"{condition}/{mode}/{case}", {})
            image_path = Path(str(item.get("trajectory_plot", "")))
            if image_path.is_file():
                axis.imshow(plt.imread(image_path))
            else:
                axis.text(0.5, 0.5, "NO PLOT", ha="center", va="center")
            axis.set_title(f"{label}\n{case}", fontsize=9)
            axis.axis("off")
    figure.suptitle(
        "C++ NMPC ESO cross-validation | nominal + singles + pairs + all disturbances "
        "| 100 Hz | warm start | N=30 | ESO bandwidth 2.5 rad/s",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output = root / "cross_validation_trajectory_long.png"
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def _write_summary(
    root: Path,
    results: dict[str, dict[str, object]],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    json_path = root / "cross_validation_summary.json"
    payload = {"_metadata": metadata, "results": results}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# C++ NMPC：ESO 有无 × 全扰动组合交叉验证",
        "",
        "覆盖无扰动、3 个单扰动、3 个两两组合和三项全开；固定：100 Hz、"
        "sample/reference/control `0.01 s`、`N=30`、原版 Q、热启动开启、ESO 带宽 `2.5 rad/s`；"
        "每个案例独立重启 Agent、PX4/Gazebo、C++ 节点并独立关闭。",
        "",
        "| 条件 | 用例 | ESO | 结果 | Position RMSE (m) | Max (m) | "
        "rx→pub Median/P95/P99 (ms) | Solve Median/P95/P99 (ms) | 失败原因 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, result in results.items():
        condition, mode, case = key.split("/")
        rx = result.get("rx_to_pub_median_p95_p99_max_ms", [np.nan] * 4)
        solve = result.get("acados_solve_wall_median_p95_p99_max_ms")
        if solve is None:
            solve = result.get("acados_solve_median_p95_p99_max_ms", [np.nan] * 4)
        lines.append(
            f"| {condition} | {case} | {mode} | "
            f"{'PASS' if result.get('success') else 'FAIL'} | "
            f"{float(result.get('tracking_position_rmse_m', np.nan)):.4f} | "
            f"{float(result.get('tracking_position_max_m', np.nan)):.4f} | "
            f"{float(rx[0]):.2f}/{float(rx[1]):.2f}/{float(rx[2]):.2f} | "
            f"{float(solve[0]):.2f}/{float(solve[1]):.2f}/{float(solve[2]):.2f} | "
            f"{result.get('reason', '')} |"
        )
    lines.extend(("", f"机器汇总：`{json_path}`", ""))
    markdown_path = root / "cross_validation_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-root", type=Path,
                        default=Path("/tmp/eso_nmpc_sitl_logs/cpp_eso_cross_validation"))
    parser.add_argument("--case-timeout", type=float, default=180.0)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS),
                        default=list(CONDITIONS))
    parser.add_argument("--only-eso", choices=("on", "off", "both"), default="both")
    parser.add_argument("--cases", nargs="+", choices=TRAJECTORIES, default=list(TRAJECTORIES))
    args = parser.parse_args()

    if not NODE.is_file() or not os.access(NODE, os.X_OK):
        parser.error(f"C++ executable not found/executable: {NODE}")
    if not PARAMS.is_file():
        parser.error(f"C++ params not found: {PARAMS}")
    if not BUILD.is_dir():
        parser.error(f"PX4 build directory not found: {BUILD}")
    if args.case_timeout <= 0.0:
        parser.error("case-timeout must be positive")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.output_root or
            ROOT / "background/sitl_regression_cpp" /
            f"cpp_eso_disturbance_cross_validation_{run_stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_root = args.log_root.resolve() / run_stamp
    original_model = MODEL.read_bytes()
    original_world = WORLD.read_bytes()
    results: dict[str, dict[str, object]] = {}
    modes = (False, True) if args.only_eso == "both" else (args.only_eso == "on",)
    metadata: dict[str, object] = {
        "backend": "ROS 2 + C++ single-process NMPC",
        "validation": "ESO on/off crossed with nominal/single/pair/all disturbance conditions",
        "execution_order": [
            f"{condition}/eso_{'on' if eso_on else 'off'}"
            for condition in args.conditions
            for eso_on in modes
        ],
        "independent_sitl_restart_per_case": True,
        "agent_restarted_per_case": True,
        "node_executable": str(NODE.resolve()),
        "params_file": str(PARAMS.resolve()),
        "reference_sample_time_s": 0.01,
        "control_period_s": 0.01,
        "horizon_steps": 30,
        "warm_start": True,
        "eso_bandwidth_rad_s": 2.5,
        "conditions": CONDITIONS,
        "cases": args.cases,
    }
    try:
        for condition_name in args.conditions:
            condition = CONDITIONS[condition_name]
            for eso_on in modes:
                mode = "on" if eso_on else "off"
                for case in args.cases:
                    key = f"{condition_name}/{mode}/{case}"
                    output = root / condition_name / f"eso_{mode}" / case
                    output.mkdir(parents=True, exist_ok=True)
                    _configure_sitl(condition)
                    sitl_log = log_root / condition_name / f"eso_{mode}_{case}_sitl.log"
                    sitl_process = None
                    runner_log = log_root / condition_name / f"eso_{mode}_{case}_runner.log"
                    runner_log.parent.mkdir(parents=True, exist_ok=True)
                    print(f"CROSS_START {key}", flush=True)
                    agent_process: subprocess.Popen[str] | None = None
                    sitl_process: subprocess.Popen[str] | None = None
                    try:
                        _cleanup_controller_processes()
                        _stop_existing_agent()
                        time.sleep(1.0)
                        agent_process, _ = _start_agent(
                            log_root / condition_name / f"eso_{mode}_{case}_agent.log"
                        )
                        sitl_process = restart_sitl(sitl_log)
                        command = _case_command(
                            output=output,
                            log_directory=log_root / condition_name / f"eso_{mode}",
                            case=case,
                            condition_name=condition_name,
                            condition=condition,
                            eso_on=eso_on,
                            case_timeout=args.case_timeout,
                        )
                        environment = os.environ.copy()
                        environment["ROS_LOCALHOST_ONLY"] = "0"
                        environment["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
                        with runner_log.open("w", encoding="utf-8") as stream:
                            completed = subprocess.run(
                                command, cwd=ROOT, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, text=True,
                                timeout=args.case_timeout + 90.0,
                            )
                        result, run_metadata = _read_case_result(output, case, runner_log)
                        result["cross_runner_return_code"] = completed.returncode
                        result["cross_runner_log"] = str(runner_log.resolve())
                        result["condition"] = condition_name
                        result["eso_mode"] = mode
                        result["trajectory"] = case
                        result["physical_disturbances"] = dict(condition)
                        if run_metadata:
                            metadata.setdefault("source_metadata", run_metadata)
                    except subprocess.TimeoutExpired:
                        result = {
                            "success": False,
                            "reason": f"cross runner timeout after {args.case_timeout + 90.0:.0f}s",
                            "condition": condition_name,
                            "eso_mode": mode,
                            "trajectory": case,
                            "physical_disturbances": dict(condition),
                            "cross_runner_log": str(runner_log.resolve()),
                        }
                    except Exception as error:  # keep matrix evidence after one case fails
                        result = {
                            "success": False,
                            "reason": f"cross orchestration error: {error}",
                            "condition": condition_name,
                            "eso_mode": mode,
                            "trajectory": case,
                            "physical_disturbances": dict(condition),
                            "cross_runner_log": str(runner_log.resolve()),
                        }
                    finally:
                        _cleanup_controller_processes()
                        _stop_process(sitl_process)
                        _cleanup_sitl()
                        _stop_process(agent_process)
                        _stop_existing_agent()
                    results[key] = result
                    print(
                        f"CROSS_RESULT {key} success={result.get('success', False)} "
                        f"reason={result.get('reason', '')}", flush=True,
                    )
    finally:
        _cleanup_controller_processes()
        _cleanup_sitl()
        _stop_existing_agent()
        MODEL.write_bytes(original_model)
        WORLD.write_bytes(original_world)

    metadata["completed_cases"] = len(results)
    metadata["passed_cases"] = sum(bool(item.get("success")) for item in results.values())
    metadata["failed_cases"] = len(results) - int(metadata["passed_cases"])
    json_path, markdown_path = _write_summary(root, results, metadata)
    plot_path = _write_cross_plot(root, results)
    metadata["summary_json"] = str(json_path.resolve())
    metadata["summary_markdown"] = str(markdown_path.resolve())
    metadata["trajectory_long_plot"] = str(plot_path.resolve()) if plot_path else None
    (root / "cross_validation_summary.json").write_text(
        json.dumps({"_metadata": metadata, "results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"CROSS_SUMMARY {markdown_path}", flush=True)
    print(f"CROSS_JSON {json_path}", flush=True)
    if plot_path:
        print(f"CROSS_PLOT {plot_path}", flush=True)
    return 0 if results and all(item.get("success", False) for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
