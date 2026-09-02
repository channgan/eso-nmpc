#!/usr/bin/env python3
"""Create a compact visual summary of the paired ESO matrix."""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("wind", "cg", "noise", "wind_cg", "wind_noise", "cg_noise", "all")
TRAJECTORIES = ("hover", "point_1m", "circle", "figure8")
METRICS = (
    ("position_rmse_delta_off_minus_on_m", "Position RMSE delta (m)"),
    ("velocity_rmse_delta_off_minus_on_m_s", "Velocity RMSE delta (m/s)"),
    ("attitude_rmse_delta_off_minus_on_rad", "Attitude RMSE delta (rad)"),
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} MATRIX_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    rows = json.loads((root / "eso_comparison.json").read_text(encoding="utf-8"))
    lookup = {(r["condition"], r["trajectory"]): r for r in rows}

    # Static PNG for easy inspection outside the inline view.
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes, METRICS):
        matrix = np.array([[lookup[(c, t)][key] for t in TRAJECTORIES] for c in CONDITIONS], dtype=float)
        vmax = max(float(np.nanmax(np.abs(matrix))), 1e-9)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(TRAJECTORIES)), ["hover", "1 m step", "circle", "figure-8"], rotation=30, ha="right")
        ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
        ax.set_xlabel("trajectory")
        ax.set_ylabel("disturbance")
        for i in range(len(CONDITIONS)):
            for j in range(len(TRAJECTORIES)):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.75, label="off − on (positive is improvement)")
    fig.suptitle("ESO A/B comparison — seed 42", fontsize=15)
    png = root / "eso_matrix_overview.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    # Inline SVG heatmaps. Positive (green) means ESO-on is better.
    width, height = 1080, 650
    left, top = 175, 76
    cell_w, cell_h = 55, 34
    gap = 335
    svg: list[str] = [
        f'<svg class="eso-chart" role="img" aria-labelledby="eso-title eso-desc" viewBox="0 0 {width} {height}">',
        '<title id="eso-title">ESO 开关对比热力图</title>',
        '<desc id="eso-desc">三组误差指标的 ESO 关闭减去 ESO 开启差值；绿色表示开启 ESO 后误差降低。</desc>',
        '<text x="18" y="28" class="chart-title">ESO 开关对比（seed 42）</text>',
        '<text x="18" y="50" class="chart-subtitle">差值 = ESO 关 − ESO 开；绿色表示 ESO 开启后误差更小</text>',
    ]
    for panel, (key, title) in enumerate(METRICS):
        x0 = left + panel * gap
        values = [lookup[(c, t)][key] for c in CONDITIONS for t in TRAJECTORIES]
        vmax = max(max(abs(float(v)) for v in values), 1e-9)
        svg.append(f'<text x="{x0 + 110}" y="70" text-anchor="middle" class="panel-title">{html.escape(title)}</text>')
        for j, t in enumerate(TRAJECTORIES):
            svg.append(f'<text x="{x0 + j * cell_w + cell_w / 2}" y="92" text-anchor="middle" class="axis-label">{html.escape(t)}</text>')
        for i, c in enumerate(CONDITIONS):
            y = top + i * cell_h
            if panel == 0:
                svg.append(f'<text x="{x0 - 10}" y="{y + 22}" text-anchor="end" class="axis-label">{html.escape(c)}</text>')
            for j, t in enumerate(TRAJECTORIES):
                value = float(lookup[(c, t)][key])
                opacity = 0.18 + 0.72 * min(abs(value) / vmax, 1.0)
                cls = "positive" if value >= 0 else "negative"
                x = x0 + j * cell_w
                svg.append(f'<rect class="cell {cls}" x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" style="fill-opacity:{opacity:.3f}" data-tooltip="{html.escape(c + ", " + t + ": " + f"{value:+.4f}")}"/>')
                svg.append(f'<text x="{x + (cell_w - 3) / 2}" y="{y + 20}" text-anchor="middle" class="cell-value">{value:+.3f}</text>')
    legend_y = top + len(CONDITIONS) * cell_h + 28
    svg += [
        f'<rect class="positive" x="{left}" y="{legend_y}" width="18" height="18" style="fill-opacity:.8"/><text x="{left + 25}" y="{legend_y + 14}" class="axis-label">ESO 开启改善</text>',
        f'<rect class="negative" x="{left + 155}" y="{legend_y}" width="18" height="18" style="fill-opacity:.8"/><text x="{left + 180}" y="{legend_y + 14}" class="axis-label">ESO 开启变差</text>',
        '</svg>',
    ]
    fragment = """<div id="eso-matrix-overview" aria-label="ESO 开关对比可视化">
<style>
#eso-matrix-overview { font-family: system-ui, sans-serif; color: var(--foreground); }
#eso-matrix-overview .eso-chart { width: 100%; height: auto; display: block; }
#eso-matrix-overview .chart-title { fill: var(--foreground); font-size: 18px; font-weight: 500; }
#eso-matrix-overview .chart-subtitle, #eso-matrix-overview .axis-label { fill: var(--muted-foreground); font-size: 12px; }
#eso-matrix-overview .panel-title { fill: var(--foreground); font-size: 13px; font-weight: 500; }
#eso-matrix-overview .cell-value { fill: var(--foreground); font-size: 11px; }
#eso-matrix-overview .cell { stroke: var(--border); stroke-width: 1; }
#eso-matrix-overview .positive { fill: var(--green); }
#eso-matrix-overview .negative { fill: var(--red); }
</style>
__SVG__
</div>
""".replace("__SVG__", "\n".join(svg))
    (root / "eso-matrix-overview.html").write_text(fragment, encoding="utf-8")
    print(png)
    print(root / "eso-matrix-overview.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
