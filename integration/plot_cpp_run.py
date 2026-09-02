#!/usr/bin/env python3
"""Create the same post-run trajectory figure used by the Python baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty trajectory log: {path}")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.percentile(values, q)) for q in (50, 95, 99))


def _quaternion_error_angle(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    actual_norm = np.linalg.norm(actual, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    actual = actual / np.maximum(actual_norm, 1.0e-12)
    reference = reference / np.maximum(reference_norm, 1.0e-12)
    cosine = np.clip(np.abs(np.sum(actual * reference, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(cosine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    is_flight_log = (run / "nmpc_flight.csv").exists()
    data_path = run / ("nmpc_flight.csv" if is_flight_log else "trajectory.csv")
    data = _read_csv(data_path)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    if is_flight_log:
        timestamp = data["px4_timestamp_sample_us"]
        if not np.any(timestamp > 0.0):
            timestamp = data["t_rx_steady_s"] * 1.0e6
        time_s = (timestamp - timestamp[0]) * 1.0e-6
        position = np.column_stack([data[f"measured_p_{axis}"] for axis in "xyz"])
        reference_position = np.column_stack([data[f"reference_p_{axis}"] for axis in "xyz"])
        velocity = np.column_stack([data[f"measured_v_{axis}"] for axis in "xyz"])
        reference_velocity = np.column_stack([data[f"reference_v_{axis}"] for axis in "xyz"])
        attitude_error = _quaternion_error_angle(
            np.column_stack([data[f"measured_q_{axis}"] for axis in ("w", "x", "y", "z")]),
            np.column_stack([data[f"reference_q_{axis}"] for axis in ("w", "x", "y", "z")]),
        )
        position_rmse = float(np.sqrt(np.mean((position - reference_position) ** 2)))
        velocity_rmse = float(np.sqrt(np.mean((velocity - reference_velocity) ** 2)))
        attitude_rmse = float(np.sqrt(np.mean(attitude_error**2)))
    else:
        time_s = data["time_s"]
        position = np.column_stack([data[f"position_{axis}"] for axis in "xyz"])
        reference_position = np.column_stack([data[f"reference_{axis}"] for axis in "xyz"])
        velocity = np.column_stack([data[f"velocity_{axis}"] for axis in "xyz"])
        reference_velocity = None
        attitude_error = None
        position_rmse = float(summary.get("tracking_position_rmse_m", np.sqrt(np.mean((position - reference_position) ** 2))))
        velocity_rmse = float(summary.get("tracking_velocity_rmse_m_s", "nan"))
        attitude_rmse = float(summary.get("tracking_attitude_rmse_rad", "nan"))
    position_error = np.linalg.norm(position - reference_position, axis=1)
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
    # The C++ supervisor currently logs actual velocity only. Reconstruct the
    # known horizontal reference speed for the supported post-run maneuvers.
    if "figure8" in run.name:
        theta = 0.5 * np.maximum(time_s - 7.0, 0.0)
        reference_horizontal_speed = np.where(
            time_s >= 7.0, np.hypot(np.cos(theta), np.cos(2.0 * theta)), 0.0
        )
    else:
        reference_horizontal_speed = np.where(time_s >= 7.0, 1.0, 0.0)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axis_xy, axis_position, axis_speed, axis_error = axes.flat
    axis_xy.plot(position[:, 0], position[:, 1], label="actual")
    axis_xy.plot(reference_position[:, 0], reference_position[:, 1], "--", label="reference")
    axis_xy.set_title("Horizontal trajectory")
    axis_xy.set_xlabel("North x [m]")
    axis_xy.set_ylabel("East y [m]")
    axis_xy.axis("equal")
    axis_xy.grid(True, alpha=0.3)
    axis_xy.legend()
    axis_xy.text(
        0.02, 0.98,
        "RMSE: pos={:.4f} m | total={:.3f} ms | rx→pub={:.3f} ms".format(
            position_rmse,
            float(summary.get("acados_time_tot_median_p95_p99_ms", [float("nan")])[0]),
            float(summary.get("rx_to_pub_median_p95_p99_ms", [float("nan")])[0]),
        ),
        transform=axis_xy.transAxes, va="top", fontsize="small",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    for index, label in enumerate(("x", "y", "z")):
        axis_position.plot(time_s, position[:, index], label=f"actual {label}")
        axis_position.plot(time_s, reference_position[:, index], "--", label=f"reference {label}")
    axis_position.set_title("Position")
    axis_position.set_xlabel("Trajectory time [s]")
    axis_position.set_ylabel("Position [m]")
    axis_position.grid(True, alpha=0.3)
    axis_position.legend(ncol=2, fontsize="small")

    axis_speed.plot(time_s, horizontal_speed, label="actual")
    if reference_velocity is not None:
        reference_horizontal_speed = np.linalg.norm(reference_velocity[:, :2], axis=1)
    axis_speed.plot(time_s, reference_horizontal_speed, "--", label="reference")
    axis_speed.axhline(2.0, color="tab:red", linestyle=":", label="limit")
    axis_speed.set_title("Horizontal speed")
    axis_speed.set_xlabel("Trajectory time [s]")
    axis_speed.set_ylabel("Speed [m/s]")
    axis_speed.grid(True, alpha=0.3)
    axis_speed.legend()

    axis_error.plot(time_s, position_error, color="tab:purple")
    axis_error.set_title("Position tracking error")
    axis_error.set_xlabel("Trajectory time [s]")
    axis_error.set_ylabel("Error norm [m]")
    axis_error.grid(True, alpha=0.3)

    if is_flight_log:
        axis_position.text(
            0.02, 0.98,
            "RMSE: pos={:.4f} m | vel={:.4f} m/s | attitude={:.3f} deg".format(
                position_rmse, velocity_rmse, np.rad2deg(attitude_rmse)),
            transform=axis_position.transAxes, va="top", fontsize="small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    trajectory_output = args.output or run / "trajectory.png"
    figure.savefig(trajectory_output, dpi=150)
    plt.close(figure)

    timing_path = run / ("nmpc_timing.csv" if is_flight_log else "controller_timing.csv")
    if timing_path.exists():
        timing = _read_csv(timing_path)
        solve = timing["acados_solve_wall_ms"]
        qp = timing["time_qp_ms"]
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        index = np.arange(solve.size)
        ax.plot(index, timing["rx_to_pub_ms"], label="rx→publish")
        ax.plot(index, solve, label="Acados solve")
        ax.plot(index, qp, label="QP total")
        ax.plot(index, timing["time_qp_xcond_ms"], label="QP condensing")
        ax.plot(index, timing["time_qp_solver_call_ms"], label="HPIPM solver")
        ax.set_title("Controller timing (post-run, one point per solve)")
        ax.set_xlabel("Solve index")
        ax.set_ylabel("Time [ms]")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        med, p95, p99 = _stats(solve)
        qmed, qp95, qp99 = _stats(qp)
        ax.text(
            0.99, 0.97,
            f"solve Median/P95/P99: {med:.2f}/{p95:.2f}/{p99:.2f} ms\n"
            f"QP Median/P95/P99: {qmed:.2f}/{qp95:.2f}/{qp99:.2f} ms",
            transform=ax.transAxes, ha="right", va="top", fontsize="small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.savefig(run / "controller_timing.png", dpi=150)
        plt.close(fig)

    print(trajectory_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
