import numpy as np
import pytest

from nmpc.trajectory import quintic_segment, smooth_profile


def test_quintic_profile_has_zero_endpoint_rates() -> None:
    assert smooth_profile(0.0, 4.0) == (0.0, 0.0, 0.0)
    assert smooth_profile(4.0, 4.0) == (1.0, 0.0, 0.0)


def test_quintic_profile_midpoint() -> None:
    position, velocity, acceleration = smooth_profile(2.0, 4.0)
    assert position == pytest.approx(0.5)
    assert velocity > 0.0
    assert acceleration == pytest.approx(0.0)


def test_quintic_profile_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        smooth_profile(0.0, 0.0)


def test_quintic_segment_matches_all_boundary_conditions() -> None:
    start_position = np.array([0.0, 0.0, -1.0])
    end_position = np.array([0.5, 0.0, -1.0])
    start_velocity = np.zeros(3)
    end_velocity = np.array([0.0, 0.25, 0.0])
    start_acceleration = np.zeros(3)
    end_acceleration = np.array([-0.125, 0.0, 0.0])

    start = quintic_segment(
        0.0,
        3.0,
        start_position,
        end_position,
        start_velocity,
        end_velocity,
        start_acceleration,
        end_acceleration,
    )
    end = quintic_segment(
        3.0,
        3.0,
        start_position,
        end_position,
        start_velocity,
        end_velocity,
        start_acceleration,
        end_acceleration,
    )

    for actual, expected in zip(start, (start_position, start_velocity, start_acceleration)):
        np.testing.assert_allclose(actual, expected)
    for actual, expected in zip(end, (end_position, end_velocity, end_acceleration)):
        np.testing.assert_allclose(actual, expected)
