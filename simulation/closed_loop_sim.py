#!/usr/bin/env python3
"""Run a simple NED hover/position-step closed-loop simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

# Permit direct execution without requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmpc.config import load_config
from nmpc.model.quadrotor import QuadrotorModel
from nmpc.reference import stationary_reference
from nmpc.solver.acados_solver import AcadosNmpc
from nmpc.solver.scipy_solver import ScipyNmpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nmpc.yaml")
    parser.add_argument("--backend", choices=("acados", "scipy"), default="scipy")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--step-time", type=float, default=0.5)
    parser.add_argument("--disturbance-x", type=float, default=0.0)
    parser.add_argument(
        "--estimate-disturbance-x",
        type=float,
        help="ESO estimate; default uses the true simulated x disturbance",
    )
    parser.add_argument("--scipy-max-iterations", type=int, default=8)
    parser.add_argument("--save", help="optional .npz output path")
    args = parser.parse_args()

    config = load_config(args.config)
    controller = (
        AcadosNmpc(config)
        if args.backend == "acados"
        else ScipyNmpc(config, max_iterations=args.scipy_max_iterations)
    )
    model = QuadrotorModel(config.model.mass, config.model.gravity)
    dt = config.controller.sample_time
    steps = int(round(args.duration / dt))
    state = np.r_[np.zeros(3), np.zeros(3), [1.0, 0.0, 0.0, 0.0]]
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
    states = np.empty((steps + 1, 10))
    controls = np.empty((steps, 4))
    solve_times = np.empty(steps)
    states[0] = state

    wall_start = perf_counter()
    for index in range(steps):
        time_now = index * dt
        target = np.array([0.0, 0.0, -1.0])
        if time_now >= args.step_time:
            target[0] = 1.0
        reference = stationary_reference(
            target, config.controller.horizon_steps, config.hover_thrust
        )
        control = controller.solve(state, reference, estimated_disturbance).as_array()
        state = model.step_rk4(state, control, dt, disturbance)
        states[index + 1] = state
        controls[index] = control
        solve_times[index] = controller.last_solve_time
    wall_elapsed = perf_counter() - wall_start

    target = np.array([1.0, 0.0, -1.0]) if args.duration > args.step_time else np.array([0.0, 0.0, -1.0])
    error = states[-1, :3] - target
    print(f"backend: {args.backend}")
    print(f"final position NED [m]: {states[-1, :3]}")
    print(f"final position error [m]: {error}, norm={np.linalg.norm(error):.4f}")
    print(f"max quaternion norm error: {np.max(np.abs(np.linalg.norm(states[:, 6:10], axis=1) - 1.0)):.3e}")
    print(f"solve mean/p99/max [ms]: {1e3*np.mean(solve_times):.3f} / {1e3*np.percentile(solve_times, 99):.3f} / {1e3*np.max(solve_times):.3f}")
    print(f"wall time [s]: {wall_elapsed:.3f}")
    if args.save:
        output = Path(args.save)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, states=states, controls=controls, solve_times=solve_times, dt=dt)
        print(f"saved: {output}")


if __name__ == "__main__":
    main()
