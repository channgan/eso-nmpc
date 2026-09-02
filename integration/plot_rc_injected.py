#!/usr/bin/env python3
"""Plot the actual response from the deterministic RC injection test."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    rows = list(csv.DictReader((run / "trajectory.csv").open()))
    t = np.array([float(row["time_s"]) for row in rows])
    p = np.array([[float(row[f"position_{axis}"]) for axis in "xyz"] for row in rows])
    v = np.array([[float(row[f"velocity_{axis}"]) for axis in "xyz"] for row in rows])
    speed = np.linalg.norm(v[:, :2], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(p[:, 0], p[:, 1], lw=1.8, label="actual")
    axes[0, 0].scatter(p[0, 0], p[0, 1], c="green", label="start")
    axes[0, 0].scatter(p[-1, 0], p[-1, 1], c="red", label="end")
    axes[0, 0].set_title("RC injected horizontal trajectory")
    axes[0, 0].set_xlabel("x [m]"); axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal"); axes[0, 0].grid(True); axes[0, 0].legend()
    axes[0, 1].plot(t, p, label=["x", "y", "z"])
    axes[0, 1].axvspan(1, 4, color="orange", alpha=0.18, label="forward stick")
    axes[0, 1].axvline(4, color="k", ls="--", lw=1, label="release")
    axes[0, 1].set_title("Position"); axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("m")
    axes[0, 1].grid(True); axes[0, 1].legend()
    axes[1, 0].plot(t, v, label=["vx", "vy", "vz"])
    axes[1, 0].plot(t, speed, "k--", lw=1.5, label="horizontal speed")
    axes[1, 0].axvspan(1, 4, color="orange", alpha=0.18)
    axes[1, 0].axvline(4, color="k", ls="--", lw=1)
    axes[1, 0].set_title("Velocity and braking after release")
    axes[1, 0].set_xlabel("time [s]"); axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True); axes[1, 0].legend()
    axes[1, 1].plot(t[t >= 8], p[t >= 8, 0] - np.mean(p[t >= 8, 0]), label="x deviation")
    axes[1, 1].plot(t[t >= 8], p[t >= 8, 1] - np.mean(p[t >= 8, 1]), label="y deviation")
    axes[1, 1].axhline(0, color="k", lw=0.8)
    axes[1, 1].set_title("Hold deviation (t ≥ 8 s)")
    axes[1, 1].set_xlabel("time [s]"); axes[1, 1].set_ylabel("m")
    axes[1, 1].grid(True); axes[1, 1].legend()
    fig.suptitle("C++ RC-NMPC injection: AUX takeover, forward stick, release")
    output = run / "rc_injected_trajectory.png"
    fig.savefig(output, dpi=150)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
