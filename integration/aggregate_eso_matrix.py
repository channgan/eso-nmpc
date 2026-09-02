#!/usr/bin/env python3
"""Aggregate paired ESO on/off NMPC matrix results into machine-readable reports."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TRAJECTORIES = ("hover", "point_1m", "circle", "figure8")
# Seven non-empty disturbance combinations. The nominal/no-disturbance
# reference is kept in the historical baseline directory, not this matrix.
CONDITIONS = ("wind", "cg", "noise", "wind_cg", "wind_noise", "cg_noise", "all")


def load_case(root: Path, condition: str, eso: str, trajectory: str) -> dict:
    path = root / f"{condition}_eso_{eso}" / trajectory / "summary.json"
    if not path.exists():
        return {"state": "missing", "summary_path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "invalid", "error": str(exc), "summary_path": str(path)}
    return {
        "state": "complete",
        "success": bool(data.get("success", False)),
        "reason": data.get("reason"),
        "position_rmse_m": data.get("tracking_position_rmse_m"),
        "velocity_rmse_m_s": data.get("velocity_rmse_m_s"),
        "attitude_rmse_rad": data.get("attitude_rmse_rad"),
        "solve_p99_ms": data.get("solve_p99_ms"),
        "solve_count": data.get("solve_count"),
        "summary_path": str(path),
        "trajectory_plot": data.get("trajectory_plot"),
    }


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def csv_value(value: object) -> object:
    return value if value is not None else "-"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} OUTPUT_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    rows: list[dict] = []
    for condition in CONDITIONS:
        for trajectory in TRAJECTORIES:
            on = load_case(root, condition, "on", trajectory)
            off = load_case(root, condition, "off", trajectory)
            row = {"condition": condition, "trajectory": trajectory, "eso_on": on, "eso_off": off}
            for metric, key in (
                ("position_rmse_delta_off_minus_on_m", "position_rmse_m"),
                ("velocity_rmse_delta_off_minus_on_m_s", "velocity_rmse_m_s"),
                ("attitude_rmse_delta_off_minus_on_rad", "attitude_rmse_rad"),
                ("solve_p99_delta_off_minus_on_ms", "solve_p99_ms"),
            ):
                a, b = on.get(key), off.get(key)
                row[metric] = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    (root / "eso_comparison.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "condition", "trajectory", "eso_on_success", "eso_off_success",
        "on_pos_rmse_m", "off_pos_rmse_m", "delta_pos_off_minus_on_m",
        "on_vel_rmse_m_s", "off_vel_rmse_m_s", "delta_vel_off_minus_on_m_s",
        "on_att_rmse_rad", "off_att_rmse_rad", "delta_att_off_minus_on_rad",
        "on_solve_p99_ms", "off_solve_p99_ms", "delta_solve_p99_off_minus_on_ms",
    ]
    with (root / "eso_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            on, off = row["eso_on"], row["eso_off"]
            writer.writerow({
                "condition": row["condition"], "trajectory": row["trajectory"],
                "eso_on_success": on.get("success", "-"), "eso_off_success": off.get("success", "-"),
                "on_pos_rmse_m": on.get("position_rmse_m", "-"), "off_pos_rmse_m": off.get("position_rmse_m", "-"),
                "delta_pos_off_minus_on_m": csv_value(row["position_rmse_delta_off_minus_on_m"]),
                "on_vel_rmse_m_s": on.get("velocity_rmse_m_s", "-"), "off_vel_rmse_m_s": off.get("velocity_rmse_m_s", "-"),
                "delta_vel_off_minus_on_m_s": csv_value(row["velocity_rmse_delta_off_minus_on_m_s"]),
                "on_att_rmse_rad": on.get("attitude_rmse_rad", "-"), "off_att_rmse_rad": off.get("attitude_rmse_rad", "-"),
                "delta_att_off_minus_on_rad": csv_value(row["attitude_rmse_delta_off_minus_on_rad"]),
                "on_solve_p99_ms": on.get("solve_p99_ms", "-"), "off_solve_p99_ms": off.get("solve_p99_ms", "-"),
                "delta_solve_p99_off_minus_on_ms": csv_value(row["solve_p99_delta_off_minus_on_ms"]),
            })

    lines = [
        "# ESO 开关成对对比",
        "",
        "差值定义：`ESO off - ESO on`；RMSE 差值为正表示 ESO 开启后误差更小。",
        "",
        "| 扰动 | 轨迹 | ESO 开 | ESO 关 | 位置开/关 (m) | 位置差值 | 速度开/关 (m/s) | 速度差值 | 姿态开/关 (rad) | 姿态差值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        on, off = row["eso_on"], row["eso_off"]
        lines.append(
            f"| {row['condition']} | {row['trajectory']} | {on.get('success', '-')} | {off.get('success', '-')} | "
            f"{fmt(on.get('position_rmse_m'))} / {fmt(off.get('position_rmse_m'))} | {fmt(row['position_rmse_delta_off_minus_on_m'])} | "
            f"{fmt(on.get('velocity_rmse_m_s'))} / {fmt(off.get('velocity_rmse_m_s'))} | {fmt(row['velocity_rmse_delta_off_minus_on_m_s'])} | "
            f"{fmt(on.get('attitude_rmse_rad'))} / {fmt(off.get('attitude_rmse_rad'))} | {fmt(row['attitude_rmse_delta_off_minus_on_rad'])} |"
        )
    (root / "eso_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {root / 'eso_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
