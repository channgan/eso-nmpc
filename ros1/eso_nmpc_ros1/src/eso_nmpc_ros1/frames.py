"""Coordinate conversions at the ROS 1 MAVROS boundary.

The NMPC core uses PX4 NED/FRD coordinates. MAVROS local odometry uses
ROS ENU/FLU coordinates, so conversion is kept in this dependency-free
module and can be tested without a ROS installation.
"""

from __future__ import annotations

import numpy as np


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def enu_to_ned(value: np.ndarray) -> np.ndarray:
    """Convert an ENU world vector to NED."""
    x, y, z = _vector(value, "value")
    return np.array([y, x, -z])


def flu_to_frd(value: np.ndarray) -> np.ndarray:
    """Convert a body vector from ROS FLU to PX4 FRD."""
    x, y, z = _vector(value, "value")
    return np.array([x, -y, -z])


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply scalar-first quaternions."""
    lw, lx, ly, lz = np.asarray(left, dtype=float)
    rw, rx, ry, rz = np.asarray(right, dtype=float)
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be a finite four-vector")
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-12:
        raise ValueError("quaternion cannot be zero")
    return quaternion / norm


def enu_quaternion_to_ned(value: np.ndarray) -> np.ndarray:
    """Convert a MAVROS body-to-ENU quaternion to body-to-NED."""
    # ENU -> NED is a 180 degree rotation around the [1, 1, 0] axis.
    enu_to_ned_quaternion = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])
    return normalize_quaternion(
        quaternion_multiply(enu_to_ned_quaternion, normalize_quaternion(value))
    )


def mavros_odometry_to_state(
    position_enu: np.ndarray,
    velocity_enu: np.ndarray,
    orientation_wxyz_enu: np.ndarray,
    angular_velocity_flu: np.ndarray,
) -> np.ndarray:
    """Build the 13-state NED/FRD vector consumed by ``AcadosNmpc``."""
    return np.r_[
        enu_to_ned(position_enu),
        enu_to_ned(velocity_enu),
        enu_quaternion_to_ned(orientation_wxyz_enu),
        flu_to_frd(angular_velocity_flu),
    ]
