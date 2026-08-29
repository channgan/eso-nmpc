#!/usr/bin/env python3
"""Run and evaluate closed-loop tracking of a smooth horizontal circle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmpc.config import load_config
from nmpc.model.quadrotor import QuadrotorModel
from nmpc.reference import circular_reference
from nmpc.solver.acados_solver import AcadosNmpc


def _attitude_errors(actual: np.ndarray, desired: np.ndarray) -> np.ndarray:
    dots = np.sum(actual * desired, axis=1)
    return 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nmpc.yaml")
    parser.add_argument("--duration", type=float, default=26.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--altitude", type=float, default=1.0, help="height above NED origin [m]")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--disturbance-x", type=float, default=0.0)
    parser.add_argument(
        "--estimate-disturbance-x",
        type=float,
        help="NMPC disturbance estimate; defaults to the true x disturbance",
    )
    parser.add_argument("--save", help="optional .npz output path")
    parser.add_argument(
        "--strict", action="store_true", help="exit nonzero when an acceptance check fails"
    )
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("duration must be positive")

    config = load_config(args.config)
    controller = AcadosNmpc(config)
    model = QuadrotorModel(config.model.mass, config.model.gravity)
    dt = config.controller.sample_time
    steps = int(round(args.duration / dt))
    center = np.array([0.0, 0.0, -args.altitude])
    disturbance = np.array([args.disturbance_x, 0.0, 0.0])
    estimated_disturbance = np.array(
        [
            args.disturbance_x
            if args.estimate_disturbance_x is None
            else args.estimate_disturbance_x,
            0.0,
            0.0,
        ]
    )

    def make_reference(time_now: float):
        return circular_reference(
            time=time_now,
            horizon_steps=config.controller.horizon_steps,
            sample_time=dt,
            center=center,
            radius=args.radius,
            speed=args.speed,
            mass=config.model.mass,
            gravity=config.model.gravity,
            yaw=args.yaw,
            disturbance=estimated_disturbance,
        )

    initial_reference = make_reference(0.0)
    state = initial_reference.states[0].copy()
    states = np.empty((steps + 1, 10))
    references = np.empty((steps + 1, 10))
    controls = np.empty((steps, 4))
    solve_times = np.empty(steps)
    statuses = np.empty(steps, dtype=int)
    states[0] = state
    references[0] = initial_reference.states[0]

    wall_start = perf_counter()
    for index in range(steps):
        time_now = index * dt
        reference = make_reference(time_now)
        control = controller.solve(state, reference, estimated_disturbance).as_array()
        state = model.step_rk4(state, control, dt, disturbance)
        states[index + 1] = state
        references[index + 1] = make_reference(time_now + dt).states[0]
        controls[index] = control
        solve_times[index] = controller.last_solve_time
        statuses[index] = controller.last_status
    wall_elapsed = perf_counter() - wall_start

    position_error = np.linalg.norm(states[:, :3] - references[:, :3], axis=1)
    velocity_error = np.linalg.norm(states[:, 3:6] - references[:, 3:6], axis=1)
    attitude_error = _attitude_errors(states[:, 6:10], references[:, 6:10])
    quaternion_norm_error = np.max(
        np.abs(np.linalg.norm(states[:, 6:10], axis=1) - 1.0)
    )
    position_rmse = float(np.sqrt(np.mean(position_error**2)))
    velocity_rmse = float(np.sqrt(np.mean(velocity_error**2)))
    attitude_rmse = float(np.sqrt(np.mean(attitude_error**2)))
    p99_solve_time = float(np.percentile(solve_times, 99))
    max_solve_time = float(np.max(solve_times))

    thrust_violation = np.any(
        (controls[:, 0] < config.limits.thrust_min - 1.0e-9)
        | (controls[:, 0] > config.limits.thrust_max + 1.0e-9)
    )
    rate_violation = np.any(
        np.abs(controls[:, 1:4]) > config.limits.body_rate_max + 1.0e-9
    )
    thrust_margin = 0.01 * (
        config.limits.thrust_max - config.limits.thrust_min
    )
    thrust_saturated = (
        (controls[:, 0] <= config.limits.thrust_min + thrust_margin)
        | (controls[:, 0] >= config.limits.thrust_max - thrust_margin)
    )
    rate_saturated = np.any(
        np.abs(controls[:, 1:4]) >= 0.99 * config.limits.body_rate_max,
        axis=1,
    )
    saturation_fraction = float(np.mean(thrust_saturated | rate_saturated))

    checks = {
        "solver status": bool(np.all(statuses == 0)),
        "finite values": bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(controls))
            and np.all(np.isfinite(solve_times))
        ),
        "quaternion norm": quaternion_norm_error < 1.0e-6,
        "position RMSE": position_rmse < 0.10,
        "maximum position error": float(np.max(position_error)) < 0.25,
        "velocity RMSE": velocity_rmse < 0.20,
        "attitude RMSE": attitude_rmse < np.deg2rad(5.0),
        "maximum attitude error": float(np.max(attitude_error)) < np.deg2rad(15.0),
        "input constraints": not (thrust_violation or rate_violation),
        "saturation fraction": saturation_fraction < 0.05,
        "p99 solve time": p99_solve_time < 0.005,
        "maximum solve time": max_solve_time < dt,
    }

    print("trajectory: horizontal circle")
    print(f"duration / radius / speed: {args.duration:.2f} s / {args.radius:.2f} m / {args.speed:.2f} m/s")
    print(f"disturbance / estimate x: {disturbance[0]:.3f} / {estimated_disturbance[0]:.3f} m/s^2")
    print(f"position RMSE / max [m]: {position_rmse:.4f} / {np.max(position_error):.4f}")
    print(f"velocity RMSE / max [m/s]: {velocity_rmse:.4f} / {np.max(velocity_error):.4f}")
    print(f"attitude RMSE / max [deg]: {np.rad2deg(attitude_rmse):.3f} / {np.rad2deg(np.max(attitude_error)):.3f}")
    print(f"quaternion max norm error: {quaternion_norm_error:.3e}")
    print(f"thrust range [N]: {np.min(controls[:, 0]):.3f} / {np.max(controls[:, 0]):.3f}")
    print(f"max absolute body rate [rad/s]: {np.max(np.abs(controls[:, 1:4]), axis=0)}")
    print(f"input saturation fraction: {100.0 * saturation_fraction:.3f}%")
    print(f"solve mean / p99 / max [ms]: {1e3*np.mean(solve_times):.3f} / {1e3*p99_solve_time:.3f} / {1e3*max_solve_time:.3f}")
    print(f"wall time [s]: {wall_elapsed:.3f}")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    accepted = all(checks.values())
    print(f"acceptance: {'PASS' if accepted else 'FAIL'}")

    if args.save:
        output = Path(args.save)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            states=states,
            references=references,
            controls=controls,
            solve_times=solve_times,
            statuses=statuses,
            dt=dt,
        )
        print(f"saved: {output}")
    if args.strict and not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
