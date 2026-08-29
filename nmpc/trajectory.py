"""Small trajectory primitives shared by simulations and integrations."""

from __future__ import annotations

import numpy as np


def smooth_profile(time_s: float, duration: float) -> tuple[float, float, float]:
    """Quintic position fraction and its first two time derivatives."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if time_s <= 0.0:
        return 0.0, 0.0, 0.0
    if time_s >= duration:
        return 1.0, 0.0, 0.0
    ratio = time_s / duration
    fraction = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5
    velocity = (30.0 * ratio**2 - 60.0 * ratio**3 + 30.0 * ratio**4) / duration
    acceleration = (60.0 * ratio - 180.0 * ratio**2 + 120.0 * ratio**3) / duration**2
    return fraction, velocity, acceleration


def quintic_segment(
    time_s: float,
    duration: float,
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_velocity: np.ndarray | None = None,
    end_velocity: np.ndarray | None = None,
    start_acceleration: np.ndarray | None = None,
    end_acceleration: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a vector trajectory with position, velocity and acceleration bounds."""
    start_position = np.asarray(start_position, dtype=float)
    end_position = np.asarray(end_position, dtype=float)
    if start_position.shape != end_position.shape:
        raise ValueError("start_position and end_position must have the same shape")
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    zeros = np.zeros_like(start_position)
    start_velocity = zeros if start_velocity is None else np.asarray(start_velocity, dtype=float)
    end_velocity = zeros if end_velocity is None else np.asarray(end_velocity, dtype=float)
    start_acceleration = (
        zeros if start_acceleration is None else np.asarray(start_acceleration, dtype=float)
    )
    end_acceleration = (
        zeros if end_acceleration is None else np.asarray(end_acceleration, dtype=float)
    )
    boundaries = (start_velocity, end_velocity, start_acceleration, end_acceleration)
    if any(value.shape != start_position.shape for value in boundaries):
        raise ValueError("all boundary vectors must have the same shape")

    if time_s <= 0.0:
        return start_position.copy(), start_velocity.copy(), start_acceleration.copy()
    if time_s >= duration:
        return end_position.copy(), end_velocity.copy(), end_acceleration.copy()

    delta = end_position - start_position
    duration_squared = duration**2
    c0 = start_position
    c1 = start_velocity
    c2 = 0.5 * start_acceleration
    c3 = (
        20.0 * delta
        - (12.0 * start_velocity + 8.0 * end_velocity) * duration
        - (3.0 * start_acceleration - end_acceleration) * duration_squared
    ) / (2.0 * duration**3)
    c4 = (
        -30.0 * delta
        + (16.0 * start_velocity + 14.0 * end_velocity) * duration
        + (3.0 * start_acceleration - 2.0 * end_acceleration) * duration_squared
    ) / (2.0 * duration**4)
    c5 = (
        12.0 * delta
        - 6.0 * (start_velocity + end_velocity) * duration
        - (start_acceleration - end_acceleration) * duration_squared
    ) / (2.0 * duration**5)

    t = float(time_s)
    position = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    velocity = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
    acceleration = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2 + 20.0 * c5 * t**3
    return position, velocity, acceleration
