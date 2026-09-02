import numpy as np

from ros1.eso_nmpc_ros1.src.eso_nmpc_ros1.frames import (
    enu_quaternion_to_ned,
    enu_to_ned,
    flu_to_frd,
    mavros_odometry_to_state,
    rotate_vector,
)


def test_world_and_body_frame_conversions() -> None:
    np.testing.assert_allclose(enu_to_ned(np.array([1.0, 2.0, 3.0])), [2.0, 1.0, -3.0])
    np.testing.assert_allclose(flu_to_frd(np.array([1.0, 2.0, 3.0])), [1.0, -2.0, -3.0])


def test_identity_enu_flu_quaternion_maps_frd_to_ned() -> None:
    converted = enu_quaternion_to_ned(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(converted, [-np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)])


def test_quaternion_rotates_body_velocity_into_world_frame() -> None:
    yaw_90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(
        rotate_vector(yaw_90, np.array([1.0, 0.0, 0.0])),
        [0.0, 1.0, 0.0],
        atol=1.0e-12,
    )


def test_mavros_odometry_state_has_ned_frd_layout() -> None:
    state = mavros_odometry_to_state(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),  # Odometry twist is in child/body FLU.
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([7.0, 8.0, 9.0]),
    )
    assert state.shape == (13,)
    np.testing.assert_allclose(state[:6], [2.0, 1.0, -3.0, 5.0, 4.0, -6.0])
    np.testing.assert_allclose(
        state[6:10], [-np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]
    )
    np.testing.assert_allclose(state[10:], [7.0, -8.0, -9.0])


def test_mavros_body_velocity_is_rotated_before_ned_conversion() -> None:
    yaw_90_enu = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    state = mavros_odometry_to_state(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        yaw_90_enu,
        np.zeros(3),
    )
    # Body-forward points north in this pose after ENU -> NED axis exchange.
    np.testing.assert_allclose(state[3:6], [1.0, 0.0, 0.0], atol=1.0e-12)
