#!/usr/bin/env python3
"""Plot known-force ESO validation data."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} VALIDATION_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    report = json.loads((root / "force_validation.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((root / "hover" / "trajectory.csv").open(encoding="utf-8")))
    time_s = [float(r["trajectory_time_s"]) for r in rows]
    estimates = [[float(r[f"disturbance_{axis}"]) for r in rows] for axis in "xyz"]
    has_model_residual = all(
        f"model_residual_{axis}_m_s2" in rows[0] for axis in "xyz"
    )
    model_residual = (
        [[float(r[f"model_residual_{axis}_m_s2"]) for r in rows] for axis in "xyz"]
        if has_model_residual else None
    )
    start = float(report["force_start_trajectory_time_s"])
    stop = float(report["force_stop_trajectory_time_s"])
    expected = report["expected_disturbance_ned_m_s2"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for index, (axis, values, target) in enumerate(zip("xyz", estimates, expected)):
        ax = axes[index]
        ax.plot(time_s, values, label=f"ESO estimate {axis}")
        if model_residual is not None:
            ax.plot(
                time_s, model_residual[index], color="tab:blue", alpha=0.8,
                label=f"model-consistent truth {axis}",
            )
        ax.axhline(
            target, color="tab:green", linestyle="--",
            label=f"physical F/m target {target:.3f} m/s²",
        )
        ax.axvspan(start, stop, color="tab:orange", alpha=0.15, label="1 N force applied")
        ax.set_ylabel(f"d_{axis} [m/s²]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("trajectory time [s]")
    force = report["force_ned_N"]
    fig.suptitle(
        "ESO known-force validation (model-consistent residual): "
        f"[{force[0]:g}, {force[1]:g}, {force[2]:g}] N NED on 2 kg vehicle"
    )
    out = root / "eso_force_validation.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
