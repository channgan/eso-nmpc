from dataclasses import replace

import pytest
import numpy as np

from nmpc.config import load_config


def test_default_config_is_consistent() -> None:
    config = load_config()
    assert config.controller.sample_time == pytest.approx(0.01)
    assert config.controller.horizon_steps == 30
    assert config.hover_thrust == pytest.approx(2.0643076923 * 9.80665)
    assert config.limits.thrust_min < config.hover_thrust < config.limits.thrust_max
    assert config.limits.horizontal_speed_max == pytest.approx(2.0)
    assert config.limits.vertical_speed_max_up == pytest.approx(3.0)
    assert config.limits.vertical_speed_max_down == pytest.approx(1.5)
    assert config.limits.tilt_max_deg == pytest.approx(45.0)
    assert config.limits.horizontal_acceleration_max == pytest.approx(2.0)
    assert config.limits.jerk_max == pytest.approx(4.0)
    assert config.manual_control.max_horizontal_speed == pytest.approx(2.0)
    assert config.manual_control.timeout == pytest.approx(0.5)


def test_config_arrays_are_not_aliased_by_yaml() -> None:
    first = load_config()
    second = load_config()
    first.weights.position[0] = 999.0
    assert second.weights.position[0] == 20.0


def test_x500_thrust_limits_match_px4_throttle() -> None:
    config = load_config()
    hover = config.hover_thrust

    np.testing.assert_allclose(
        config.limits.thrust_min / hover,
        config.px4.throttle_min / config.px4.hover_throttle,
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        config.limits.thrust_max / hover,
        config.px4.throttle_max / config.px4.hover_throttle,
        rtol=0.0,
        atol=1.0e-5,
    )
