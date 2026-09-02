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


def inverse_dynamics_attitude_and_thrust_batch(
    acceleration: np.ndarray,
    yaw: np.ndarray,
    mass: float,
    gravity: float,
    disturbance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized inverse dynamics for a complete reference horizon.

    The translational inverse-dynamics equations are independent at each
    horizon stage, so all vector operations are evaluated on ``(N, 3)``
    arrays.  Quaternion conversion is batched as well; only the later
    hemisphere alignment remains sequential because it depends on the
    previous stage quaternion.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    disturbance = np.zeros(3) if disturbance is None else np.asarray(disturbance, dtype=float)
    if acceleration.ndim != 2 or acceleration.shape[1] != 3:
        raise ValueError("acceleration must have shape (N, 3)")
    if yaw.shape != (acceleration.shape[0],) or disturbance.shape != (3,):
        raise ValueError("yaw must have shape (N,) and disturbance shape (3,)")
    if not np.all(np.isfinite(acceleration)) or not np.all(np.isfinite(yaw)):
        raise ValueError("acceleration and yaw must be finite")
    if not np.all(np.isfinite(disturbance)):
        raise ValueError("disturbance must be finite")
    specific_thrust = np.array([0.0, 0.0, gravity]) + disturbance - acceleration
    magnitude = np.linalg.norm(specific_thrust, axis=1)
    if np.any(magnitude < 1.0e-8):
        raise ValueError("desired acceleration implies zero thrust and undefined attitude")
    body_z = specific_thrust / magnitude[:, None]
    heading = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
    body_y_raw = np.cross(body_z, heading)
    parallel = np.linalg.norm(body_y_raw, axis=1) < 1.0e-8
    if np.any(parallel):
        fallback_heading = np.column_stack(
            (-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw))
        )
        body_y_raw[parallel] = np.cross(body_z[parallel], fallback_heading[parallel])
    body_y = body_y_raw / np.linalg.norm(body_y_raw, axis=1)[:, None]
    body_x = np.cross(body_y, body_z)
    rotation = np.stack((body_x, body_y, body_z), axis=2)

    # Batched scalar-first rotation-matrix to quaternion conversion.  The
    # branch masks mirror ``rotation_to_quaternion`` and cover all valid
    # rotation matrices without introducing Euler angles.
    count = acceleration.shape[0]
    quaternion = np.empty((count, 4), dtype=float)
    trace = np.trace(rotation, axis1=1, axis2=2)
    positive = trace > 0.0
    if np.any(positive):
        scale = 2.0 * np.sqrt(trace[positive] + 1.0)
        matrix = rotation[positive]
        quaternion[positive, 0] = 0.25 * scale
        quaternion[positive, 1] = (matrix[:, 2, 1] - matrix[:, 1, 2]) / scale
        quaternion[positive, 2] = (matrix[:, 0, 2] - matrix[:, 2, 0]) / scale
        quaternion[positive, 3] = (matrix[:, 1, 0] - matrix[:, 0, 1]) / scale
    remaining = ~positive
    diagonal_index = np.argmax(np.diagonal(rotation, axis1=1, axis2=2), axis=1)
    for index in range(3):
        selected = remaining & (diagonal_index == index)
        if not np.any(selected):
            continue
        matrix = rotation[selected]
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[:, 0, 0] - matrix[:, 1, 1] - matrix[:, 2, 2])
            quaternion[selected, 0] = (matrix[:, 2, 1] - matrix[:, 1, 2]) / scale
            quaternion[selected, 1] = 0.25 * scale
            quaternion[selected, 2] = (matrix[:, 0, 1] + matrix[:, 1, 0]) / scale
            quaternion[selected, 3] = (matrix[:, 0, 2] + matrix[:, 2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[:, 1, 1] - matrix[:, 0, 0] - matrix[:, 2, 2])
            quaternion[selected, 0] = (matrix[:, 0, 2] - matrix[:, 2, 0]) / scale
            quaternion[selected, 1] = (matrix[:, 0, 1] + matrix[:, 1, 0]) / scale
            quaternion[selected, 2] = 0.25 * scale
            quaternion[selected, 3] = (matrix[:, 1, 2] + matrix[:, 2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[:, 2, 2] - matrix[:, 0, 0] - matrix[:, 1, 1])
            quaternion[selected, 0] = (matrix[:, 1, 0] - matrix[:, 0, 1]) / scale
            quaternion[selected, 1] = (matrix[:, 0, 2] + matrix[:, 2, 0]) / scale
            quaternion[selected, 2] = (matrix[:, 1, 2] + matrix[:, 2, 1]) / scale
            quaternion[selected, 3] = 0.25 * scale
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    return quaternion, mass * magnitude


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
    reference = Reference(
        states=np.repeat(state[None, :], horizon_steps + 1, axis=0),
        controls=np.repeat(control[None, :], horizon_steps, axis=0),
    )
    return attach_feedforward_rate_states(reference)


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


def average_body_rate_batch(
    quaternion: np.ndarray, next_quaternion: np.ndarray, sample_time: float
) -> np.ndarray:
    """Vectorized quaternion-difference body-rate calculation."""
    quaternion = np.asarray(quaternion, dtype=float)
    next_quaternion = np.asarray(next_quaternion, dtype=float)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4 or next_quaternion.shape != quaternion.shape:
        raise ValueError("quaternion arrays must both have shape (N, 4)")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    sign = np.where(np.sum(quaternion * next_quaternion, axis=1) < 0.0, -1.0, 1.0)
    next_quaternion = next_quaternion * sign[:, None]
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    lw, lx, ly, lz = conjugate.T
    rw, rx, ry, rz = next_quaternion.T
    delta = np.column_stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
    delta /= np.linalg.norm(delta, axis=1)[:, None]
    negative = delta[:, 0] < 0.0
    delta[negative] *= -1.0
    vector_norm = np.linalg.norm(delta[:, 1:], axis=1)
    rates = np.zeros((delta.shape[0], 3), dtype=float)
    moving = vector_norm >= 1.0e-12
    angle = 2.0 * np.arctan2(
        vector_norm[moving], np.clip(delta[moving, 0], -1.0, 1.0)
    )
    rates[moving] = (
        angle[:, None] / sample_time * delta[moving, 1:] / vector_norm[moving, None]
    )
    return rates


def align_quaternion_sequence(
    quaternions: np.ndarray, anchor: np.ndarray
) -> np.ndarray:
    """Normalize and align a quaternion sequence without a Python stage loop.

    The sign of the first quaternion is chosen relative to ``anchor``.  Each
    subsequent sign is chosen relative to the preceding *raw* quaternion;
    taking the cumulative product reproduces the usual sequential hemisphere
    alignment exactly, while evaluating all dot products in one NumPy pass.
    """
    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4)")
    if values.shape[0] == 0:
        return values.copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("quaternions must be finite")
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (4,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor must be a finite quaternion")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < 1.0e-12):
        raise ValueError("cannot normalize a zero quaternion")
    result = values / norms[:, None]
    anchor = anchor / np.linalg.norm(anchor)
    signs = np.ones(result.shape[0], dtype=float)
    signs[0] = -1.0 if float(np.dot(result[0], anchor)) < 0.0 else 1.0
    if result.shape[0] > 1:
        adjacent = np.sum(result[1:] * result[:-1], axis=1)
        signs[1:] = np.where(adjacent < 0.0, -1.0, 1.0)
    return result * np.cumprod(signs)[:, None]


def inverse_dynamics_body_rate(
    acceleration: np.ndarray,
    jerk: np.ndarray,
    yaw: float,
    yaw_rate: float,
    sample_time: float,
    mass: float,
    gravity: float,
    disturbance: np.ndarray | None = None,
) -> np.ndarray:
    """Compute discrete body-rate feed-forward from jerk and yaw rate.

    Jerk advances the desired acceleration over one controller interval.  The
    resulting change of inverse-dynamics attitude, together with yaw rate, is
    converted to the constant body rate used by the discrete NMPC model.
    """
    jerk = np.asarray(jerk, dtype=float)
    if jerk.shape != (3,) or not np.all(np.isfinite(jerk)):
        raise ValueError("jerk must be a finite three-vector")
    if not np.isfinite(yaw_rate) or sample_time <= 0.0:
        raise ValueError("yaw_rate must be finite and sample_time positive")
    quaternion, _ = inverse_dynamics_attitude_and_thrust(
        acceleration, yaw, mass, gravity, disturbance
    )
    next_quaternion, _ = inverse_dynamics_attitude_and_thrust(
        np.asarray(acceleration, dtype=float) + jerk * sample_time,
        yaw + yaw_rate * sample_time,
        mass,
        gravity,
        disturbance,
    )
    return average_body_rate(quaternion, next_quaternion, sample_time)


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
    states = np.empty((horizon_steps + 1, 13))
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
        states[stage, :10] = np.r_[position, velocity, quaternion]
        previous_quaternion = quaternion

    controls = np.empty((horizon_steps, 4))
    controls[:, 0] = thrust[:-1]
    for stage in range(horizon_steps):
        controls[stage, 1:4] = average_body_rate(
            states[stage, 6:10], states[stage + 1, 6:10], sample_time
        )
    return attach_feedforward_rate_states(
        Reference(states=states[:, :10], controls=controls)
    )


def attach_feedforward_rate_states(reference: Reference) -> Reference:
    """Extend a 10-column state reference with the feedforward body rates.

    The model's last three states are the actual body rates, driven by the
    rate commands through a first-order lag.  Their reference trajectory is
    the inverse-dynamics feedforward rate, so the linearization point is a
    consistent trajectory of the lagged dynamics.
    """
    states = np.empty((reference.states.shape[0], 13), dtype=float)
    states[:, :10] = reference.states
    states[:-1, 10:13] = reference.controls[:, 1:4]
    states[-1, 10:13] = reference.controls[-1, 1:4]
    return Reference(states=states, controls=reference.controls)


def align_reference_quaternions(reference: Reference, anchor: np.ndarray) -> Reference:
    """Remove quaternion double-cover sign jumps across a reference horizon."""
    states = np.asarray(reference.states, dtype=float).copy()
    states[:, 6:10] = align_quaternion_sequence(states[:, 6:10], anchor)
    return Reference(states=states, controls=np.asarray(reference.controls, dtype=float).copy())
