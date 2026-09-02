#!/usr/bin/env python3
"""Periodically summarize NMPC baseline result completeness."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "background" / "baseline_runs" / "eso_comparison"
OUT = BASE / "progress.json"
TRAJECTORIES = ("hover", "point_1m", "circle", "figure8")


def snapshot() -> dict[str, object]:
    groups: dict[str, object] = {}
    for directory in sorted(BASE.iterdir() if BASE.exists() else ()):
        if not directory.is_dir() or directory.name == "verified":
            continue
        cases = {}
        for trajectory in TRAJECTORIES:
            summary = directory / trajectory / "summary.json"
            if not summary.exists():
                cases[trajectory] = {"state": "missing"}
                continue
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                cases[trajectory] = {"state": "invalid", "error": str(exc)}
                continue
            cases[trajectory] = {
                "state": "complete",
                "success": bool(data.get("success", False)),
                "reason": data.get("reason"),
            }
        groups[directory.name] = cases
    return {"updated_at": datetime.now().astimezone().isoformat(), "groups": groups}


def main() -> None:
    interval = float(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 30.0
    while True:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(snapshot(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        time.sleep(max(1.0, interval))


if __name__ == "__main__":
    main()
