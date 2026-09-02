from dataclasses import replace

import pytest
import numpy as np

from nmpc.config import load_config


def test_default_config_is_consistent() -> None:
    config = load_config()
    assert config.controller.sample_time == pytest.approx(0.01)
    assert config.controller.control_period == pytest.approx(0.01)
    assert config.controller.horizon_steps == 30
    assert config.hover_thrust == pytest.approx(2.0 * 9.80665)
    assert config.limits.thrust_min < config.hover_thrust < config.limits.thrust_max
    assert config.limits.horizontal_speed_max == pytest.approx(2.0)
    assert config.limits.vertical_speed_max_up == pytest.approx(3.0)
    assert config.limits.vertical_speed_max_down == pytest.approx(1.5)
    assert config.limits.tilt_max_deg == pytest.approx(45.0)
    assert config.limits.horizontal_acceleration_max == pytest.approx(2.0)
    assert config.limits.jerk_max == pytest.approx(4.0)
    assert config.manual_control.max_horizontal_speed == pytest.approx(2.0)
    assert config.manual_control.timeout == pytest.approx(0.5)
    assert config.cost_scales.weight_factor == pytest.approx(0.7)
    assert config.eso.enabled is True
    assert config.eso.bandwidth_rad_s == pytest.approx(3.0)
    assert config.eso.disturbance_clamp_m_s2 == pytest.approx(1.0)
    assert config.eso.activation_delay_s == pytest.approx(3.0)
    assert config.eso.innovation_limit_m_s == pytest.approx(0.5)
    np.testing.assert_allclose(
        config.cost_scales.position_error_m, [0.1, 0.1, 0.1 / np.sqrt(2.0)]
    )
    np.testing.assert_allclose(config.cost_scales.velocity_error_m_s, 0.1)
    np.testing.assert_allclose(config.cost_scales.attitude_error_deg, 5.0)
    assert config.cost_scales.thrust_correction_n == pytest.approx(0.418330013)
    np.testing.assert_allclose(config.cost_scales.body_rate_correction_deg_s, 5.0)
    np.testing.assert_allclose(
        config.cost_scales.state_weights,
        0.7
        * np.r_[
            [100.0, 100.0, 200.0],
            [100.0] * 3,
            [1.0 / np.deg2rad(5.0) ** 2] * 3,
        ],
    )
    np.testing.assert_allclose(
        config.cost_scales.control_weights,
        0.7 * np.r_[5.714285714285714, [1.0 / np.deg2rad(5.0) ** 2] * 3],
    )


def test_config_arrays_are_not_aliased_by_yaml() -> None:
    first = load_config()
    second = load_config()
    first.cost_scales.position_error_m[0] = 999.0
    assert second.cost_scales.position_error_m[0] == 0.1


def test_real_airframe_thrust_limits_contain_hover() -> None:
    config = load_config()
    assert config.limits.thrust_min < config.hover_thrust < config.limits.thrust_max
