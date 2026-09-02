#!/usr/bin/env python3
"""Benchmark the generated acados solver using warm-started hover solves."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmpc.config import load_config
from nmpc.reference import stationary_reference
from nmpc.solver.acados_solver import AcadosNmpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nmpc.yaml")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--warm-start",
        choices=("config", "on", "off"),
        default="config",
        help="override controller.warm_start for an A/B benchmark",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.warm_start != "config":
        config = replace(
            config,
            controller=replace(config.controller, warm_start=args.warm_start == "on"),
        )
    controller = AcadosNmpc(config)
    state = np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
    reference = stationary_reference(
        np.array([0.0, 0.0, -1.0]), config.controller.horizon_steps, config.hover_thrust
    )
    samples = []
    warm_start_used = 0
    for index in range(args.warmup + args.iterations):
        controller.solve(state, reference, np.zeros(3))
        if index >= args.warmup:
            samples.append(controller.last_solve_time)
            warm_start_used += int(controller.warm_start_used)
    milliseconds = 1.0e3 * np.asarray(samples)
    print(f"warm start enabled: {config.controller.warm_start}")
    print(f"warm start used: {warm_start_used}/{len(milliseconds)}")
    print(f"samples: {len(milliseconds)}")
    print(f"mean [ms]: {np.mean(milliseconds):.4f}")
    print(f"p99 [ms]: {np.percentile(milliseconds, 99):.4f}")
    print(f"p99.9 [ms]: {np.percentile(milliseconds, 99.9):.4f}")
    print(f"max [ms]: {np.max(milliseconds):.4f}")


if __name__ == "__main__":
    main()
