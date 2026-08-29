import numpy as np

from nmpc.model.quadrotor import QuadrotorModel, quaternion_to_rotation
from nmpc.reference import (
    align_reference_quaternions,
    circular_reference,
    flatness_attitude_and_thrust,
    stationary_reference,
)


def test_stationary_reference_has_n_plus_one_states() -> None:
    reference = stationary_reference(np.array([1.0, 2.0, -3.0]), 30, 14.7, yaw=0.4)
    reference.validate(30)
    np.testing.assert_allclose(
        reference.states[:, :3], np.repeat([[1.0, 2.0, -3.0]], 31, axis=0)
    )
    np.testing.assert_allclose(reference.controls[:, 0], 14.7)
    np.testing.assert_allclose(np.linalg.norm(reference.states[:, 6:10], axis=1), 1.0)


def test_flatness_mapping_reproduces_requested_acceleration() -> None:
    mass = 1.5
    gravity = 9.80665
    acceleration = np.array([1.2, -0.7, 0.3])
    disturbance = np.array([0.1, 0.2, -0.1])
    quaternion, thrust = flatness_attitude_and_thrust(
        acceleration, yaw=0.6, mass=mass, gravity=gravity, disturbance=disturbance
    )
    state = np.r_[np.zeros(6), quaternion, np.zeros(3)]
    control = np.r_[thrust, np.zeros(3)]
    derivative = QuadrotorModel(mass, gravity).continuous_dynamics(
        state, control, disturbance
    )
    np.testing.assert_allclose(derivative[3:6], acceleration, atol=1.0e-12)
    rotation = quaternion_to_rotation(quaternion)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)


def test_reference_quaternion_signs_are_continuous() -> None:
    reference = stationary_reference(np.zeros(3), 3, 10.0)
    reference.states[1, 6:10] *= -1.0
    reference.states[3, 6:10] *= -1.0
    aligned = align_reference_quaternions(reference, np.array([1.0, 0.0, 0.0, 0.0]))
    dots = np.sum(aligned.states[1:, 6:10] * aligned.states[:-1, 6:10], axis=1)
    assert np.all(dots >= 0.0)


def test_circular_reference_is_dynamically_consistent() -> None:
    mass = 1.5
    gravity = 9.80665
    sample_time = 0.01
    radius = 1.0
    speed = 0.5
    disturbance = np.array([0.2, -0.1, 0.0])
    reference = circular_reference(
        time=0.7,
        horizon_steps=30,
        sample_time=sample_time,
        center=np.array([0.0, 0.0, -1.0]),
        radius=radius,
        speed=speed,
        mass=mass,
        gravity=gravity,
        disturbance=disturbance,
    )
    reference.validate(30)

    radii = np.linalg.norm(reference.states[:, :2], axis=1)
    speeds = np.linalg.norm(reference.states[:, 3:5], axis=1)
    np.testing.assert_allclose(radii, radius, atol=1.0e-12)
    np.testing.assert_allclose(speeds, speed, atol=1.0e-12)
    np.testing.assert_allclose(reference.states[:, 2], -1.0)
    np.testing.assert_allclose(
        np.linalg.norm(reference.states[:, 6:10], axis=1), 1.0, atol=1.0e-12
    )

    model = QuadrotorModel(mass, gravity)
    expected_acceleration = -(speed**2 / radius**2) * reference.states[0, :3]
    expected_acceleration[2] = 0.0
    derivative = model.continuous_dynamics(
        reference.states[0], reference.controls[0], disturbance
    )
    np.testing.assert_allclose(derivative[:3], reference.states[0, 3:6], atol=1.0e-12)
    np.testing.assert_allclose(derivative[3:6], expected_acceleration, atol=1.0e-12)
