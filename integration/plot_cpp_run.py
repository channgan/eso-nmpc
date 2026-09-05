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


def _csv_has_rows(path: Path) -> bool:
    """Return whether a CSV contains at least one data row after its header."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            return next(reader, None) is not None
    except (OSError, StopIteration):
        return False


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.percentile(values, q)) for q in (50, 95, 99))


def _finite_column(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    values = data.get(name, np.asarray([], dtype=float))
    return values[np.isfinite(values)]


def _median_column(data: dict[str, np.ndarray] | None, name: str) -> float:
    if data is None:
        return float("nan")
    values = _finite_column(data, name)
    return float(np.median(values)) if values.size else float("nan")


def _time_gap_indices(time_s: np.ndarray) -> tuple[np.ndarray, float]:
    """Return gaps that must not be bridged by a plotted line.

    The supervisor trajectory log is sampled from its odometry callback.  A
    temporary executor/DDS stall can therefore leave a multi-second hole even
    though the C++ controller kept solving and logging.  Connecting the two
    endpoints would manufacture a false chord in circle/figure-eight plots.
    """
    if time_s.size < 2:
        return np.asarray([], dtype=int), float("nan")
    delta = np.diff(time_s)
    positive = delta[np.isfinite(delta) & (delta > 0.0)]
    if positive.size == 0:
        return np.asarray([], dtype=int), float("nan")
    threshold = max(0.1, 5.0 * float(np.median(positive)))
    return np.flatnonzero(delta > threshold), threshold


def _break_plot_lines(
    time_s: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Insert NaNs after large time gaps so matplotlib leaves a visible break."""
    gaps, _ = _time_gap_indices(time_s)
    outputs = [np.asarray(array, dtype=float).copy() for array in arrays]
    for index in gaps:
        for array in outputs:
            array[index + 1] = np.nan
    return tuple(outputs)


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
    parser.add_argument(
        "--source",
        choices=("auto", "trajectory", "nmpc_flight"),
        default="auto",
        help=(
            "trajectory data source; auto prefers the supervisor trajectory.csv "
            "when it exists, and only falls back to nmpc_flight.csv for a "
            "standalone C++ flight log"
        ),
    )
    args = parser.parse_args()
    run = args.run_directory.resolve()
    trajectory_path = run / "trajectory.csv"
    flight_log_path = run / "nmpc_flight.csv"
    supervisor_data = _read_csv(trajectory_path) if trajectory_path.exists() else None
    visualization_note = ""
    if args.source == "trajectory":
        data_path = trajectory_path
    elif args.source == "nmpc_flight":
        data_path = flight_log_path
    elif trajectory_path.exists():
        # Regression directories contain both files.  trajectory.csv remains
        # the canonical source for the supervisor metric, but it can contain
        # observer gaps.  In that case use the same-run C++ flight log for the
        # plotted samples so a missing arc is not replaced by a false chord.
        supervisor_gaps, supervisor_gap_threshold = _time_gap_indices(
            supervisor_data.get("time_s", np.asarray([], dtype=float))
        )
        if (
            flight_log_path.exists()
            and supervisor_gaps.size
            and _csv_has_rows(flight_log_path)
        ):
            data_path = flight_log_path
            visualization_note = (
                "same-run nmpc_flight.csv; supervisor trajectory.csv has "
                f"{supervisor_gaps.size} gap(s) > {supervisor_gap_threshold:.3f} s"
            )
        else:
            data_path = trajectory_path
    else:
        data_path = flight_log_path
    if not data_path.exists():
        raise FileNotFoundError(f"no trajectory log found: {trajectory_path} or {flight_log_path}")
    is_flight_log = data_path.name == "nmpc_flight.csv"
    data = _read_csv(data_path)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    canonical_position_rmse = float("nan")
    if supervisor_data is not None:
        supervisor_time = supervisor_data.get("time_s", np.asarray([], dtype=float))
        supervisor_error = supervisor_data.get("position_error_m", np.asarray([], dtype=float))
        evaluation_mask = supervisor_time >= 5.0
        evaluation_mask &= np.isfinite(supervisor_error)
        if np.any(evaluation_mask):
            canonical_position_rmse = float(
                np.sqrt(np.mean(supervisor_error[evaluation_mask] ** 2))
            )

    if is_flight_log:
        timestamp = data["px4_timestamp_sample_us"]
        # PX4 timestamps are the canonical cross-log clock, but a restarted
        # SITL can emit an occasional stale/short timestamp during DDS
        # handover.  Do not let one such sample stretch the plot to 1e9 s;
        # use the monotonic host receive clock for visualization in that case.
        timestamp_delta = np.diff(timestamp)
        timestamp_valid = (
            np.all(np.isfinite(timestamp))
            and np.all(timestamp > 0.0)
            and np.all(timestamp_delta > 0.0)
            and np.all(timestamp_delta < 1.0e7)
        )
        if not timestamp_valid:
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
        raw_position_rmse = float(np.sqrt(np.mean((position - reference_position) ** 2)))
        raw_velocity_rmse = float(np.sqrt(np.mean((velocity - reference_velocity) ** 2)))
        raw_attitude_rmse = float(np.sqrt(np.mean(attitude_error**2)))
        # Keep the number displayed on a regression plot identical to the
        # formal supervisor metric even when the same-run C++ log is used only
        # to restore missing visualization samples.
        position_rmse = (
            canonical_position_rmse
            if np.isfinite(canonical_position_rmse)
            else float(summary.get("tracking_position_rmse_m", raw_position_rmse))
        )
        velocity_rmse = float(summary.get("tracking_velocity_rmse_m_s", raw_velocity_rmse))
        attitude_rmse = float(summary.get("tracking_attitude_rmse_rad", raw_attitude_rmse))
    else:
        time_s = data["time_s"]
        position = np.column_stack([data[f"position_{axis}"] for axis in "xyz"])
        reference_position = np.column_stack([data[f"reference_{axis}"] for axis in "xyz"])
        velocity = np.column_stack([data[f"velocity_{axis}"] for axis in "xyz"])
        reference_velocity_columns = [f"reference_velocity_{axis}" for axis in "xyz"]
        if all(column in data for column in reference_velocity_columns):
            reference_velocity = np.column_stack(
                [data[column] for column in reference_velocity_columns]
            )
        else:
            # Older trajectory.csv files did not store reference velocity.
            # Derive it from the same sampled reference position as a
            # backwards-compatible fallback.
            reference_velocity = np.gradient(reference_position, time_s, axis=0)
        attitude_error = None
        # Keep the fallback definition identical to the C++ supervisor:
        # RMSE is computed from the Euclidean position-error norm per sample,
        # not from the three Cartesian components as independent samples.  The
        # supervisor evaluates after the 4 s ascent plus a 1 s settling window.
        position_error_norm = np.linalg.norm(position - reference_position, axis=1)
        evaluation_mask = time_s >= 5.0
        if not np.any(evaluation_mask):
            evaluation_mask = np.ones(time_s.shape, dtype=bool)
        position_rmse = float(
            summary.get(
                "tracking_position_rmse_m",
                np.sqrt(np.mean(position_error_norm[evaluation_mask] ** 2)),
            )
        )
        velocity_rmse = float(summary.get("tracking_velocity_rmse_m_s", "nan"))
        attitude_rmse = float(summary.get("tracking_attitude_rmse_rad", "nan"))
    position_error = np.linalg.norm(position - reference_position, axis=1)
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
    reference_horizontal_speed = np.linalg.norm(reference_velocity[:, :2], axis=1)

    position_plot, reference_position_plot, velocity_plot, reference_velocity_plot = (
        _break_plot_lines(
            time_s, position, reference_position, velocity, reference_velocity
        )
    )
    position_error_plot = np.linalg.norm(position_plot - reference_position_plot, axis=1)

    # C++ timing is written to nmpc_timing.csv, not to the supervisor's
    # summary.json.  Load it before drawing the trajectory annotation so the
    # displayed values come from the same per-solve data as controller_timing.png.
    timing_path = run / "controller_timing.csv"
    if not timing_path.exists():
        timing_path = run / "nmpc_timing.csv"
    timing = (
        _read_csv(timing_path)
        if timing_path.exists() and _csv_has_rows(timing_path)
        else None
    )
    solve_median = _median_column(timing, "acados_solve_wall_ms")
    total_median = _median_column(timing, "time_tot_ms")
    rx_median = _median_column(timing, "rx_to_pub_ms")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    source_label = data_path.name
    if visualization_note:
        source_label += f" ({visualization_note})"
    figure.suptitle(f"{run.name} | visualization source: {source_label}", fontsize="medium")
    axis_xy, axis_position, axis_speed, axis_error = axes.flat
    axis_xy.plot(position_plot[:, 0], position_plot[:, 1], label="actual")
    axis_xy.plot(
        reference_position_plot[:, 0], reference_position_plot[:, 1], "--", label="reference"
    )
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
            total_median if np.isfinite(total_median) else solve_median,
            rx_median,
        ),
        transform=axis_xy.transAxes, va="top", fontsize="small",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    for index, label in enumerate(("x", "y", "z")):
        axis_position.plot(time_s, position_plot[:, index], label=f"actual {label}")
        axis_position.plot(
            time_s, reference_position_plot[:, index], "--", label=f"reference {label}"
        )
    axis_position.set_title("Position")
    axis_position.set_xlabel("Trajectory time [s]")
    axis_position.set_ylabel("Position [m]")
    axis_position.grid(True, alpha=0.3)
    axis_position.legend(ncol=2, fontsize="small")

    axis_speed.plot(time_s, np.linalg.norm(velocity_plot[:, :2], axis=1), label="actual")
    if reference_velocity_plot is not None:
        reference_horizontal_speed = np.linalg.norm(reference_velocity_plot[:, :2], axis=1)
    axis_speed.plot(time_s, reference_horizontal_speed, "--", label="reference")
    axis_speed.axhline(2.0, color="tab:red", linestyle=":", label="limit")
    axis_speed.set_title("Horizontal speed")
    axis_speed.set_xlabel("Trajectory time [s]")
    axis_speed.set_ylabel("Speed [m/s]")
    axis_speed.grid(True, alpha=0.3)
    axis_speed.legend()

    axis_error.plot(time_s, position_error_plot, color="tab:purple")
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

    # Timing is independent of the trajectory source.  Regression directories
    # retain nmpc_timing.csv alongside trajectory.csv, while older standalone
    # SITL logs use controller_timing.csv.
    if timing is not None:
        solve = _finite_column(timing, "acados_solve_wall_ms")
        qp = _finite_column(timing, "time_qp_ms")
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        index = np.arange(solve.size)
        ax.plot(index, _finite_column(timing, "rx_to_pub_ms"), label="rx→publish")
        ax.plot(index, solve, label="Acados solve")
        ax.plot(index, qp, label="QP total")
        ax.plot(index, _finite_column(timing, "time_qp_xcond_ms"), label="QP condensing")
        ax.plot(index, _finite_column(timing, "time_qp_solver_call_ms"), label="HPIPM solver")
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

    print(f"plot source: {data_path}")
    print(trajectory_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
