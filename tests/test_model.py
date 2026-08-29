import numpy as np

from nmpc.model.quadrotor import QuadrotorModel, quaternion_to_rotation


def identity_state(position=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.r_[position, np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]


def test_hover_is_equilibrium() -> None:
    model = QuadrotorModel(mass=1.5)
    state = identity_state()
    control = np.array([model.mass * model.gravity, 0.0, 0.0, 0.0])
    derivative = model.continuous_dynamics(state, control, np.zeros(3))
    np.testing.assert_allclose(derivative, 0.0, atol=1.0e-12)


def test_disturbance_enters_as_world_acceleration() -> None:
    model = QuadrotorModel(mass=1.5)
    state = identity_state()
    control = np.array([model.mass * model.gravity, 0.0, 0.0, 0.0])
    disturbance = np.array([0.5, -0.2, 0.1])
    derivative = model.continuous_dynamics(state, control, disturbance)
    np.testing.assert_allclose(derivative[3:6], disturbance, atol=1.0e-12)


def test_positive_yaw_rate_uses_scalar_first_hamilton_product() -> None:
    model = QuadrotorModel(mass=1.5)
    state = identity_state()
    state[10:13] = [0.0, 0.0, 2.0]  # actual rate already at the commanded value
    control = np.array([model.mass * model.gravity, 0.0, 0.0, 2.0])
    derivative = model.continuous_dynamics(state, control)
    np.testing.assert_allclose(derivative[6:10], [0.0, 0.0, 0.0, 1.0])


def test_rate_lag_drives_actual_rate_toward_command() -> None:
    model = QuadrotorModel(mass=1.5, rate_tau=0.2)
    state = identity_state()
    state[10:13] = [1.0, -0.5, 0.25]
    control = np.array([model.mass * model.gravity, 0.0, 0.0, 0.5])
    derivative = model.continuous_dynamics(state, control)
    np.testing.assert_allclose(
        derivative[10:13], (control[1:4] - state[10:13]) / 0.2, atol=1.0e-12
    )


def test_rk4_preserves_unit_quaternion() -> None:
    model = QuadrotorModel(mass=1.5)
    state = identity_state()
    control = np.array([model.mass * model.gravity, 1.0, -0.5, 0.2])
    for _ in range(1000):
        state = model.step_rk4(state, control, 0.01)
    np.testing.assert_allclose(np.linalg.norm(state[6:10]), 1.0, atol=1.0e-12)
    rotation = quaternion_to_rotation(state[6:10])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1.0e-12)

