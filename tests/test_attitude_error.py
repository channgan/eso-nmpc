import numpy as np

from nmpc.model.quadrotor import quaternion_attitude_error


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(0.5 * angle), axis * np.sin(0.5 * angle)]


def test_attitude_error_is_axis_angle_rotation_vector() -> None:
    angle = np.deg2rad(60.0)
    error = quaternion_attitude_error(
        _axis_angle(np.array([0.0, 1.0, 0.0]), angle),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(error, [0.0, angle, 0.0], atol=1.0e-12)


def test_attitude_error_is_invariant_to_quaternion_sign() -> None:
    actual = _axis_angle(np.array([1.0, 2.0, -3.0]), np.deg2rad(120.0))
    reference = _axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(20.0))
    expected = quaternion_attitude_error(actual, reference)
    np.testing.assert_allclose(quaternion_attitude_error(-actual, reference), expected)
    np.testing.assert_allclose(quaternion_attitude_error(actual, -reference), expected)


def test_attitude_error_uses_shortest_rotation() -> None:
    error = quaternion_attitude_error(
        _axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(350.0)),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(error, [0.0, 0.0, np.deg2rad(-10.0)], atol=1.0e-12)
