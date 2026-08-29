"""Reference horizon construction and differential-flatness helpers."""

from __future__ import annotations

import numpy as np

from .model.quadrotor import align_quaternion, normalize_quaternion
from .types import Reference


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a scalar-first unit quaternion."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_quaternion(quaternion)


def inverse_dynamics_attitude_and_thrust(
    acceleration: np.ndarray,
    yaw: float,
    mass: float,
    gravity: float,
    disturbance: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Solve translational inverse dynamics for attitude and total thrust.

    The model is a = g*e3 - T/m*R*e3 + d.  FRD body z therefore
    points along g*e3 + d - a at the requested acceleration.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    disturbance = np.zeros(3) if disturbance is None else np.asarray(disturbance, dtype=float)
    if acceleration.shape != (3,) or disturbance.shape != (3,):
        raise ValueError("acceleration and disturbance must have shape (3,)")
    specific_thrust = np.array([0.0, 0.0, gravity]) + disturbance - acceleration
    magnitude = np.linalg.norm(specific_thrust)
    if magnitude < 1.0e-8:
        raise ValueError("desired acceleration implies zero thrust and undefined attitude")
    body_z = specific_thrust / magnitude
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    body_y_unnormalized = np.cross(body_z, heading)
    if np.linalg.norm(body_y_unnormalized) < 1.0e-8:
        # Heading is parallel to body z; choose a deterministic horizontal axis.
        heading = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        body_y_unnormalized = np.cross(body_z, heading)
    body_y = body_y_unnormalized / np.linalg.norm(body_y_unnormalized)
    body_x = np.cross(body_y, body_z)
    rotation = np.column_stack((body_x, body_y, body_z))
    return rotation_to_quaternion(rotation), mass * magnitude


# Compatibility alias for older callers. The operation is inverse dynamics;
# differential flatness is how a sufficiently smooth trajectory supplies the
# acceleration and yaw inputs used by it.
flatness_attitude_and_thrust = inverse_dynamics_attitude_and_thrust


def stationary_reference(
    position: np.ndarray,
    horizon_steps: int,
    hover_thrust: float,
    yaw: float = 0.0,
) -> Reference:
    """Create a constant pose horizon with zero velocity and body rate."""
    position = np.asarray(position, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    state = np.r_[position, np.zeros(3), quaternion]
    control = np.array([hover_thrust, 0.0, 0.0, 0.0])
    return Reference(
        states=np.repeat(state[None, :], horizon_steps + 1, axis=0),
        controls=np.repeat(control[None, :], horizon_steps, axis=0),
    )


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def average_body_rate(
    quaternion: np.ndarray, next_quaternion: np.ndarray, sample_time: float
) -> np.ndarray:
    """Return the constant body rate rotating quaternion to next_quaternion."""
    next_quaternion = align_quaternion(next_quaternion, quaternion)
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    delta = normalize_quaternion(_quaternion_product(conjugate, next_quaternion))
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = np.linalg.norm(delta[1:])
    if vector_norm < 1.0e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[0], -1.0, 1.0))
    return angle / sample_time * delta[1:] / vector_norm


def circular_reference(
    time: float,
    horizon_steps: int,
    sample_time: float,
    center: np.ndarray,
    radius: float,
    speed: float,
    mass: float,
    gravity: float,
    yaw: float = 0.0,
    disturbance: np.ndarray | None = None,
) -> Reference:
    """Create a dynamically consistent horizontal circular reference horizon.

    ``center`` and the generated trajectory use NED coordinates.  Positive
    ``speed`` traverses the circle from north toward east.
    """
    center = np.asarray(center, dtype=float)
    disturbance = (
        np.zeros(3) if disturbance is None else np.asarray(disturbance, dtype=float)
    )
    scalar_values = np.array(
        [time, sample_time, radius, speed, mass, gravity, yaw], dtype=float
    )
    if center.shape != (3,) or disturbance.shape != (3,):
        raise ValueError("center and disturbance must have shape (3,)")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(disturbance)):
        raise ValueError("center or disturbance contains a non-finite value")
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("circular reference input contains a non-finite value")
    if horizon_steps < 1 or sample_time <= 0.0:
        raise ValueError("horizon_steps and sample_time must be positive")
    if radius <= 0.0 or speed <= 0.0 or mass <= 0.0 or gravity <= 0.0:
        raise ValueError("radius, speed, mass, and gravity must be positive")

    angular_rate = speed / radius
    states = np.empty((horizon_steps + 1, 10))
    thrust = np.empty(horizon_steps + 1)
    previous_quaternion: np.ndarray | None = None
    for stage in range(horizon_steps + 1):
        stage_time = time + stage * sample_time
        angle = angular_rate * stage_time
        cosine = np.cos(angle)
        sine = np.sin(angle)
        position = center + np.array([radius * cosine, radius * sine, 0.0])
        velocity = np.array([-speed * sine, speed * cosine, 0.0])
        acceleration = np.array(
            [
                -speed * angular_rate * cosine,
                -speed * angular_rate * sine,
                0.0,
            ]
        )
        quaternion, thrust[stage] = inverse_dynamics_attitude_and_thrust(
            acceleration,
            yaw=yaw,
            mass=mass,
            gravity=gravity,
            disturbance=disturbance,
        )
        if previous_quaternion is not None:
            quaternion = align_quaternion(quaternion, previous_quaternion)
        states[stage] = np.r_[position, velocity, quaternion]
        previous_quaternion = quaternion

    controls = np.empty((horizon_steps, 4))
    controls[:, 0] = thrust[:-1]
    for stage in range(horizon_steps):
        controls[stage, 1:4] = average_body_rate(
            states[stage, 6:10], states[stage + 1, 6:10], sample_time
        )
    return Reference(states=states, controls=controls)


def align_reference_quaternions(reference: Reference, anchor: np.ndarray) -> Reference:
    """Remove quaternion double-cover sign jumps across a reference horizon."""
    states = np.asarray(reference.states, dtype=float).copy()
    previous = normalize_quaternion(anchor)
    for state in states:
        state[6:10] = align_quaternion(state[6:10], previous)
        previous = state[6:10]
    return Reference(states=states, controls=np.asarray(reference.controls, dtype=float).copy())
