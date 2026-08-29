import numpy as np
import pytest

from nmpc.model.quadrotor import QuadrotorModel
from nmpc.validation import ModelValidationRecorder


def _hover_state() -> np.ndarray:
    return np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]


def test_exact_hover_model_has_zero_prediction_residuals() -> None:
    model = QuadrotorModel(2.0)
    recorder = ModelValidationRecorder(model)
    state = _hover_state()
    control = np.array([model.mass * model.gravity, 0.0, 0.0, 0.0])
    for index in range(101):
        recorder.add(index * 10_000, state, np.zeros(3), control, "hold")
        state = model.step_rk4(state, control, 0.01)

    summary = recorder.summary((0.01, 0.1, 0.5))
    assert summary["rejected_interval_count"] == 0
    assert summary["segments"] == {"hold": 101}
    for horizon in summary["prediction"].values():
        assert horizon["position_error_m"]["rmse_norm"] == pytest.approx(0.0)
        assert horizon["velocity_error_m_s"]["rmse_norm"] == pytest.approx(0.0)
        assert horizon["orientation_error_rad"]["rmse"] == pytest.approx(0.0)


def test_rate_tracking_error_compares_command_to_next_measurement() -> None:
    model = QuadrotorModel(2.0)
    recorder = ModelValidationRecorder(model)
    state = _hover_state()
    control = np.array([model.mass * model.gravity, 0.2, -0.1, 0.05])
    recorder.add(0, state, np.zeros(3), control)
    following = model.step_rk4(state, control, 0.01)
    recorder.add(10_000, following, np.zeros(3), control)

    rate_error = recorder.summary((0.01,))["body_rate_tracking_error_rad_s"]
    np.testing.assert_allclose(rate_error["bias"], control[1:4])
    assert rate_error["count"] == 1


def test_timestamp_gap_is_rejected_from_dynamics_statistics() -> None:
    model = QuadrotorModel(2.0)
    recorder = ModelValidationRecorder(model, maximum_interval=0.05)
    state = _hover_state()
    control = np.array([model.mass * model.gravity, 0.0, 0.0, 0.0])
    recorder.add(0, state, np.zeros(3), control)
    recorder.add(1_000_000, state, np.zeros(3), control)
    summary = recorder.summary((0.01,))
    assert summary["valid_interval_count"] == 0
    assert summary["rejected_interval_count"] == 1
    assert summary["prediction"]["0.010s"]["position_error_m"]["count"] == 0


def test_rate_alignment_recovers_known_command_delay() -> None:
    model = QuadrotorModel(2.0)
    recorder = ModelValidationRecorder(model)
    state = _hover_state()
    delay = 0.03
    time_step = 0.01
    timestamps = np.arange(0.0, 1.01, time_step)
    command_history = np.column_stack(
        (
            np.sin(2.0 * np.pi * 2.0 * timestamps),
            np.sin(2.0 * np.pi * 3.0 * timestamps + 0.2),
            np.sin(2.0 * np.pi * 4.0 * timestamps - 0.1),
        )
    )
    for index, timestamp in enumerate(timestamps):
        delayed_time = timestamp - delay
        measured = np.array(
            [
                np.interp(delayed_time, timestamps, command_history[:, axis])
                for axis in range(3)
            ]
        )
        control = np.r_[model.mass * model.gravity, command_history[index]]
        recorder.add(round(timestamp * 1e6), state, measured, control)

    aligned = recorder.summary((0.01,))[
        "delay_aligned_body_rate_tracking_error_rad_s"
    ]
    np.testing.assert_allclose(aligned["best_delay_s_per_axis"], delay, atol=0.002)
    np.testing.assert_allclose(aligned["rmse_per_axis"], 0.0, atol=1e-12)
