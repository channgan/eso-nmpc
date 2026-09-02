import numpy as np
import pytest

from nmpc.observer import VelocityLESO


def test_velocity_leso_converges_to_constant_disturbance() -> None:
    observer = VelocityLESO(3.0, 2.0, 1.0)
    velocity = np.zeros(3)
    model_acceleration = np.array([0.0, 0.0, 9.80665])
    disturbance = np.array([0.25, -0.15, 0.4])
    for _ in range(2500):
        velocity += 0.01 * (model_acceleration + disturbance)
        estimate = observer.update(velocity, model_acceleration, 0.01)
    np.testing.assert_allclose(estimate, disturbance, atol=2.0e-2)


def test_velocity_leso_clamps_disturbance_and_holds_velocity() -> None:
    observer = VelocityLESO(4.0, 0.5, 0.2)
    observer.reset(np.zeros(3))
    for _ in range(100):
        estimate = observer.update(np.array([10.0, 0.0, 0.0]), np.zeros(3), 0.01)
    assert np.max(np.abs(estimate)) <= 0.5 + 1.0e-12
    held = observer.hold(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(held, estimate)
    np.testing.assert_allclose(observer.velocity_hat, [1.0, 2.0, 3.0])


def test_velocity_leso_ignores_invalid_step() -> None:
    observer = VelocityLESO(3.0, 1.0, 1.0)
    observer.reset(np.zeros(3))
    before = observer.disturbance_hat.copy()
    np.testing.assert_allclose(observer.update(np.ones(3), np.zeros(3), 0.0), before)
    np.testing.assert_allclose(observer.update(np.ones(3), np.zeros(3), np.nan), before)


def test_velocity_leso_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        VelocityLESO(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        VelocityLESO(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        VelocityLESO(1.0, 1.0, 0.0)
