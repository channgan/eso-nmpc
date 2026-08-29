"""10-state quadrotor model in PX4-compatible NED/FRD coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NX = 10
NU = 4
NP = 3
ACADOS_NP = NP + 4  # translational disturbance + reference quaternion


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        raise ValueError("cannot normalize a zero quaternion")
    return q / norm


def align_quaternion(reference: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Pick the quaternion sign closest to anchor (q and -q are equivalent)."""
    reference = normalize_quaternion(reference)
    anchor = normalize_quaternion(anchor)
    return -reference if np.dot(reference, anchor) < 0.0 else reference


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Return the body-to-world rotation matrix for scalar-first quaternion q."""
    qw, qx, qy, qz = normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def quaternion_derivative(q: np.ndarray, body_rate: np.ndarray) -> np.ndarray:
    """q_dot = 0.5 * q (x) [0, omega], scalar-first convention."""
    qw, qx, qy, qz = q
    wx, wy, wz = body_rate
    return 0.5 * np.array(
        [
            -qx * wx - qy * wy - qz * wz,
            qw * wx + qy * wz - qz * wy,
            qw * wy + qz * wx - qx * wz,
            qw * wz + qx * wy - qy * wx,
        ]
    )


def quaternion_attitude_error(q: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) rotation vector from reference to attitude.

    Both inputs use scalar-first convention.  The result is invariant to either
    quaternion's sign and has magnitude equal to the principal rotation angle.
    """
    q = normalize_quaternion(q)
    q_reference = normalize_quaternion(q_reference)
    wr, xr, yr, zr = q_reference
    w, x, y, z = q
    # conjugate(q_reference) (x) q
    error = np.array(
        [
            wr * w + xr * x + yr * y + zr * z,
            wr * x - xr * w - yr * z + zr * y,
            wr * y + xr * z - yr * w - zr * x,
            wr * z - xr * y + yr * x - zr * w,
        ]
    )
    if error[0] < 0.0:
        error = -error
    vector_norm = np.linalg.norm(error[1:4])
    if vector_norm < 1.0e-10:
        return 2.0 * error[1:4]
    angle = 2.0 * np.arctan2(vector_norm, error[0])
    return angle * error[1:4] / vector_norm


@dataclass(frozen=True)
class QuadrotorModel:
    mass: float
    gravity: float = 9.80665

    def __post_init__(self) -> None:
        if self.mass <= 0.0 or self.gravity <= 0.0:
            raise ValueError("mass and gravity must be positive")

    def continuous_dynamics(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray | None = None
    ) -> np.ndarray:
        state, control, disturbance = self._validate(state, control, disturbance)
        velocity = state[3:6]
        rotation = quaternion_to_rotation(state[6:10])
        acceleration = np.array([0.0, 0.0, self.gravity])
        acceleration -= control[0] / self.mass * rotation[:, 2]
        acceleration += disturbance
        q_dot = quaternion_derivative(state[6:10], control[1:4])
        return np.r_[velocity, acceleration, q_dot]

    def step_rk4(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
        disturbance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate one sample and explicitly restore the unit quaternion invariant."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        state, control, disturbance = self._validate(state, control, disturbance)
        f = self.continuous_dynamics
        k1 = f(state, control, disturbance)
        k2 = f(state + 0.5 * dt * k1, control, disturbance)
        k3 = f(state + 0.5 * dt * k2, control, disturbance)
        k4 = f(state + dt * k3, control, disturbance)
        result = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        result[6:10] = normalize_quaternion(result[6:10])
        return result

    @staticmethod
    def _validate(
        state: np.ndarray, control: np.ndarray, disturbance: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        control = np.asarray(control, dtype=float)
        disturbance = np.zeros(NP) if disturbance is None else np.asarray(disturbance, dtype=float)
        if state.shape != (NX,):
            raise ValueError(f"state must have shape ({NX},), got {state.shape}")
        if control.shape != (NU,):
            raise ValueError(f"control must have shape ({NU},), got {control.shape}")
        if disturbance.shape != (NP,):
            raise ValueError(f"disturbance must have shape ({NP},), got {disturbance.shape}")
        if not all(np.all(np.isfinite(v)) for v in (state, control, disturbance)):
            raise ValueError("model input contains a non-finite value")
        return state, control, disturbance

    def export_acados_model(self, name: str = "quadrotor_nmpc") -> Any:
        """Build an AcadosModel lazily so NumPy-only tests need no acados install."""
        try:
            import casadi as ca
            from acados_template import AcadosModel
        except ImportError as error:
            raise RuntimeError(
                "CasADi/acados_template is unavailable. Install acados and export "
                "ACADOS_SOURCE_DIR before generating the solver."
            ) from error

        x = ca.SX.sym("x", NX)
        x_dot = ca.SX.sym("x_dot", NX)
        u = ca.SX.sym("u", NU)
        parameters = ca.SX.sym("p", ACADOS_NP)
        d = parameters[:NP]

        qw, qx, qy, qz = (x[i] for i in range(6, 10))
        # Third column of R_body_to_world. Quaternion norm is preserved by q_dot.
        body_z_world = ca.vertcat(
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        acceleration = ca.vertcat(0.0, 0.0, self.gravity)
        acceleration -= u[0] / self.mass * body_z_world
        acceleration += d
        omega_quaternion = ca.vertcat(0.0, u[1], u[2], u[3])
        q = x[6:10]
        q_dot = 0.5 * ca.vertcat(
            q[0] * omega_quaternion[0] - ca.dot(q[1:4], omega_quaternion[1:4]),
            q[0] * omega_quaternion[1] + q[1] * omega_quaternion[0] + q[2] * omega_quaternion[3] - q[3] * omega_quaternion[2],
            q[0] * omega_quaternion[2] + q[2] * omega_quaternion[0] + q[3] * omega_quaternion[1] - q[1] * omega_quaternion[3],
            q[0] * omega_quaternion[3] + q[3] * omega_quaternion[0] + q[1] * omega_quaternion[2] - q[2] * omega_quaternion[1],
        )
        f_expl = ca.vertcat(x[3:6], acceleration, q_dot)

        model = AcadosModel()
        model.name = name
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.p = parameters
        model.f_expl_expr = f_expl
        model.f_impl_expr = x_dot - f_expl
        return model
