#!/usr/bin/env python3
"""Keep one lowest-error four-case SITL history per simulation mode.

The input is a C++ regression JSON report written by
``integration/run_cpp_regression.py``.  A record is eligible only when all
four mandatory cases completed successfully and landed disarmed.  The score
is the sum of the four position RMSE values, so lower is better.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_ROOT = PROJECT_ROOT / "background/best_sitl_history"
CASES = ("hover", "point_1m", "circle", "figure8")
MODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _mode(value: str) -> str:
    if not MODE_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "mode 只能包含字母、数字、点、下划线和短横线，且不能以符号开头"
        )
    return value


def _report_path(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("suite_summary.json"))
        if len(candidates) != 1:
            raise SystemExit(f"目录中未找到唯一 suite_summary.json: {path}")
        return candidates[0]
    if not path.is_file():
        raise SystemExit(f"回归 JSON 不存在: {path}")
    return path


def _read_report(path: Path) -> tuple[dict[str, dict[str, Any]], Path, float, dict[str, Any]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"无法读取回归 JSON {path}: {error}") from error

    missing = [case for case in CASES if not isinstance(report.get(case), dict)]
    if missing:
        raise SystemExit(f"回归报告缺少完整四项: {', '.join(missing)}")

    results = {case: report[case] for case in CASES}
    run_metadata = report.get("_metadata", {})
    if not isinstance(run_metadata, dict):
        run_metadata = {}
    invalid: list[str] = []
    for case, result in results.items():
        rmse = result.get("tracking_position_rmse_m")
        if (
            result.get("success") is not True
            or result.get("landed_disarmed") is not True
            or not isinstance(rmse, (int, float))
            or not math.isfinite(float(rmse))
            or not float(rmse) >= 0.0
        ):
            invalid.append(case)
    if invalid:
        raise SystemExit(
            "只有四项全部成功且 landed_disarmed=true 才能归档 best；无效项: "
            + ", ".join(invalid)
        )

    case_dirs = [Path(results[case]["case_directory"]).expanduser().resolve() for case in CASES]
    source_root = case_dirs[0].parent
    if any(case_dir.parent != source_root for case_dir in case_dirs):
        raise SystemExit("四项 case_directory 不属于同一个回归目录")
    if not source_root.is_dir():
        raise SystemExit(f"回归原始目录不存在: {source_root}")

    score = sum(float(results[case]["tracking_position_rmse_m"]) for case in CASES)
    return results, source_root, score, run_metadata


def _read_existing_score(target: Path) -> float | None:
    metadata = target / "best_record.json"
    if not metadata.is_file():
        return None
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))["position_rmse_total_m"]
        return float(value)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_metadata(
    path: Path,
    *,
    mode: str,
    report_path: Path,
    source_root: Path,
    results: dict[str, dict[str, Any]],
    score: float,
    run_metadata: dict[str, Any],
) -> None:
    metadata = {
        "mode": mode,
        "criterion": "sum of four tracking_position_rmse_m; lower is better",
        "position_rmse_total_m": score,
        "position_rmse_by_case_m": {
            case: float(results[case]["tracking_position_rmse_m"]) for case in CASES
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "source_run_directory": str(source_root),
        "run_configuration": run_metadata,
        "parameter_snapshot_in_best": "cpp_params.yaml"
        if (path.parent / "cpp_params.yaml").is_file() else None,
        "cases": list(CASES),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update(mode: str, report: Path, dry_run: bool) -> int:
    report = _report_path(report)
    results, source_root, score, run_metadata = _read_report(report)
    target = BEST_ROOT / mode
    old_score = _read_existing_score(target) if target.is_dir() else None

    if old_score is not None and score >= old_score:
        print(f"KEEP {mode}: new={score:.6f} m >= best={old_score:.6f} m")
        return 0
    if dry_run:
        print(
            f"DRY-RUN REPLACE {mode}: "
            f"new={score:.6f} m, old={'none' if old_score is None else f'{old_score:.6f} m'}"
        )
        return 0

    BEST_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = BEST_ROOT / f".{mode}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source_root, temporary)
    shutil.copy2(report, temporary / "source_report.json")
    _write_metadata(
        temporary / "best_record.json",
        mode=mode,
        report_path=report,
        source_root=source_root,
        results=results,
        score=score,
        run_metadata=run_metadata,
    )
    if target.exists():
        shutil.rmtree(target)
    os.replace(temporary, target)
    print(
        f"UPDATED {mode}: {score:.6f} m "
        f"({target.relative_to(PROJECT_ROOT)})"
    )
    return 0


def status() -> int:
    if not BEST_ROOT.is_dir():
        print("best history: empty")
        return 0
    records = sorted(path for path in BEST_ROOT.iterdir() if path.is_dir())
    if not records:
        print("best history: empty")
        return 0
    for path in records:
        score = _read_existing_score(path)
        print(f"{path.name}: {'unknown' if score is None else f'{score:.6f} m'}")
    return 0


def remove(mode: str | None, remove_all: bool, confirm: bool) -> int:
    if not confirm:
        raise SystemExit("删除 best 记录需要显式添加 --yes")
    if remove_all:
        targets = [path for path in BEST_ROOT.iterdir() if path.is_dir()] if BEST_ROOT.is_dir() else []
    elif mode is not None:
        targets = [BEST_ROOT / mode]
    else:
        raise SystemExit("remove 需要 --mode MODE 或 --all")
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            print(f"REMOVED {target.relative_to(PROJECT_ROOT)}")
        else:
            print(f"NOT FOUND {target.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update", help="用新回归报告更新某模式的 best")
    update_parser.add_argument("--mode", required=True, type=_mode)
    update_parser.add_argument("--report", required=True, type=Path)
    update_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("status", help="查看各模式当前 best 分数")

    remove_parser = subparsers.add_parser("remove", help="删除冒烟确认后的 best 记录")
    remove_parser.add_argument("--mode", type=_mode)
    remove_parser.add_argument("--all", action="store_true")
    remove_parser.add_argument("--yes", action="store_true")

    args = parser.parse_args()
    if args.command == "update":
        return update(args.mode, args.report, args.dry_run)
    if args.command == "status":
        return status()
    return remove(args.mode, args.all, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
