import json

from integration.run_sitl_regression import CASES, _write_suite_reports


def test_mandatory_baseline_uses_requested_interfaces() -> None:
    mapping = {case.name: case.reference_source for case in CASES}
    assert mapping == {
        "hover": "direct",
        "point_1m": "direct",
        "circle": "direct",
        "figure8": "direct",
    }


def test_suite_report_contains_machine_and_human_readable_results(tmp_path) -> None:
    results = {
        "hover": {
            "success": True,
            "reference_source": "direct",
            "tracking_position_rmse_m": 0.01,
            "velocity_rmse_m_s": 0.02,
            "attitude_rmse_rad": 0.03,
            "solve_p99_ms": 3.0,
        }
    }
    _write_suite_reports(tmp_path, results)
    assert json.loads((tmp_path / "suite_summary.json").read_text()) == results
    markdown = (tmp_path / "suite_summary.md").read_text()
    assert "hover" in markdown
    assert "direct" in markdown
    assert "PASS" in markdown
