#!/usr/bin/env python3
"""Validate ESO against a known Gazebo world-frame external force.

The force is applied to the x500 model's canonical link through Gazebo's
ApplyLinkWrench system.  Inputs are expressed in PX4 NED; Gazebo expects ENU.
"""
from __future__ import annotations

import csv
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GZ = "/usr/bin/gz"
WORLD = "/world/default"
ENTITY = "x500_0"
ENTITY_ID = 10  # Gazebo dynamic_pose/info currently reports x500_0 as id 10.
FLIGHT_RE = re.compile(r"flight t=([0-9]+(?:\.[0-9]+)?)")


def publish_force(force_ned: tuple[float, float, float]) -> None:
    # PX4 NED -> Gazebo ENU: [N, E, D] -> [E, N, U].
    force_enu = (force_ned[1], force_ned[0], -force_ned[2])
    text = (
        f"entity {{id: {ENTITY_ID}}} "
        f"wrench {{force {{x: {force_enu[0]:.9g} y: {force_enu[1]:.9g} z: {force_enu[2]:.9g}}}}}"
    )
    subprocess.run(
        [GZ, "topic", "-t", f"{WORLD}/wrench/persistent", "-m", "gz.msgs.EntityWrench", "-p", text],
        check=True,
        timeout=5.0,
    )


def clear_force() -> None:
    text = f"id: {ENTITY_ID}"
    subprocess.run(
        [GZ, "topic", "-t", f"{WORLD}/wrench/clear", "-m", "gz.msgs.Entity", "-p", text],
        check=False,
        timeout=5.0,
    )


def latest_flight_time(log: Path) -> float | None:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = [float(match.group(1)) for match in FLIGHT_RE.finditer(text)]
    return values[-1] if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ESO against a known external force")
    parser.add_argument("output", nargs="?", type=Path,
                        default=ROOT / "background" / "baseline_runs" / "eso_force_validation")
    parser.add_argument("--force-ned-x-n", type=float, default=1.0)
    parser.add_argument("--force-ned-y-n", type=float, default=0.0)
    parser.add_argument("--force-ned-z-n", type=float, default=0.0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    force_ned = (args.force_ned_x_n, args.force_ned_y_n, args.force_ned_z_n)
    mass_kg = 2.0
    delay_after_flight_s = 2.0
    force_duration_s = 8.0
    command = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "integration/run_sitl_regression.py"),
        "--output-directory", str(output), "--cases", "hover", "--case-timeout", "90",
        "--cg-offset-x-m", "0", "--wind-velocity-x-m-s", "0",
    ]
    run_log = output / "run.log"
    flight_log = output / "hover" / "run.log"
    flight_detected_at = None
    force_start_at = None
    force_start_trajectory_s = None
    force_stop_at = None
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=run_log.open("w"), stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and process.poll() is None:
            if flight_log.exists() and "phase=FLIGHT" in flight_log.read_text(errors="replace"):
                flight_detected_at = time.monotonic()
                break
            time.sleep(0.2)
        if flight_detected_at is None:
            raise RuntimeError("hover flight phase was not detected")
        time.sleep(delay_after_flight_s)
        force_start_at = time.monotonic()
        force_start_trajectory_s = latest_flight_time(flight_log)
        publish_force(force_ned)
        time.sleep(force_duration_s)
        force_stop_at = time.monotonic()
        clear_force()
        return_code = process.wait(timeout=100.0)
    except Exception as error:
        clear_force()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15.0)
        (output / "force_validation_error.txt").write_text(str(error) + "\n", encoding="utf-8")
        print(f"force validation failed: {error}", file=sys.stderr)
        return 1
    finally:
        clear_force()

    trajectory = output / "hover" / "trajectory.csv"
    estimates: list[dict[str, float]] = []
    if trajectory.exists():
        with trajectory.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    item = {
                        "t": float(row["trajectory_time_s"]),
                        "x": float(row["disturbance_x"]),
                        "y": float(row["disturbance_y"]),
                        "z": float(row["disturbance_z"]),
                    }
                    for axis in "xyz":
                        key = f"model_residual_{axis}_m_s2"
                        if key in row:
                            item[f"truth_{axis}"] = float(row[key])
                    estimates.append(item)
                except (KeyError, TypeError, ValueError):
                    continue
    start_t = force_start_trajectory_s
    stop_t = start_t + force_duration_s if start_t is not None else None
    window = [r for r in estimates if start_t is not None and stop_t is not None and start_t <= r["t"] <= stop_t]
    baseline = [r for r in estimates if start_t is not None and r["t"] < start_t - 0.5]
    def mean(rows: list[dict[str, float]], axis: str) -> float | None:
        values = [r[axis] for r in rows if axis in r and math.isfinite(r[axis])]
        return sum(values) / len(values) if values else None
    report = {
        "return_code": return_code,
        "force_ned_N": force_ned,
        "mass_kg": mass_kg,
        "expected_disturbance_ned_m_s2": [v / mass_kg for v in force_ned],
        "force_start_after_flight_s": delay_after_flight_s,
        "force_duration_s": force_duration_s,
        "force_start_trajectory_time_s": start_t,
        "force_stop_trajectory_time_s": stop_t,
        "baseline_estimate_mean_m_s2": [mean(baseline, a) for a in "xyz"],
        "force_window_estimate_mean_m_s2": [mean(window, a) for a in "xyz"],
        "baseline_sample_count": len(baseline),
        "force_window_sample_count": len(window),
        "trajectory_summary": str((output / "hover" / "summary.json").resolve()),
    }
    if window and all(f"truth_{axis}" in window[0] for axis in "xyz"):
        report["force_window_model_consistent_truth_mean_m_s2"] = [
            mean(window, f"truth_{axis}") for axis in "xyz"
        ]
        report["model_consistent_validation"] = (
            "The ESO estimate is assessed against the truth residual generated "
            "by the unchanged command-thrust model; force/mass is shown only "
            "as a physical reference."
        )
    (output / "force_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if return_code == 0 else return_code


if __name__ == "__main__":
    raise SystemExit(main())
