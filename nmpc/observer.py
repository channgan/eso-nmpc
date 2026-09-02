"""Disturbance observers used by the NMPC integration."""

from __future__ import annotations

import numpy as np


class VelocityLESO:
    """Three-axis second-order linear ESO for a velocity measurement.

    The plant channel is ``v_dot = a_model + d`` with slowly varying lumped
    acceleration disturbance ``d``. Observer poles are placed at
    ``-bandwidth``; hence the gains are ``2*w`` and ``w**2``.
    """

    def __init__(
        self,
        bandwidth_rad_s: float,
        disturbance_clamp_m_s2: float,
        innovation_limit_m_s: float,
    ) -> None:
        if not np.isfinite(bandwidth_rad_s) or bandwidth_rad_s <= 0.0:
            raise ValueError("bandwidth_rad_s must be finite and positive")
        if not np.isfinite(disturbance_clamp_m_s2) or disturbance_clamp_m_s2 <= 0.0:
            raise ValueError("disturbance_clamp_m_s2 must be finite and positive")
        if not np.isfinite(innovation_limit_m_s) or innovation_limit_m_s <= 0.0:
            raise ValueError("innovation_limit_m_s must be finite and positive")
        self.bandwidth = float(bandwidth_rad_s)
        self.beta1 = 2.0 * self.bandwidth
        self.beta2 = self.bandwidth * self.bandwidth
        self.disturbance_clamp = float(disturbance_clamp_m_s2)
        self.innovation_limit = float(innovation_limit_m_s)
        self.velocity_hat = np.zeros(3)
        self.disturbance_hat = np.zeros(3)
        self.initialized = False

    def reset(
        self,
        velocity: np.ndarray | None = None,
        disturbance: np.ndarray | None = None,
    ) -> None:
        velocity_value = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)
        disturbance_value = (
            np.zeros(3) if disturbance is None else np.asarray(disturbance, dtype=float)
        )
        if velocity_value.shape != (3,) or disturbance_value.shape != (3,):
            raise ValueError("velocity and disturbance must have shape (3,)")
        if not np.all(np.isfinite(velocity_value)) or not np.all(np.isfinite(disturbance_value)):
            raise ValueError("velocity and disturbance must be finite")
        self.velocity_hat = velocity_value.copy()
        self.disturbance_hat = np.clip(
            disturbance_value, -self.disturbance_clamp, self.disturbance_clamp
        )
        self.initialized = True

    def hold(self, velocity: np.ndarray) -> np.ndarray:
        """Freeze disturbance adaptation while synchronizing velocity state."""
        value = np.asarray(velocity, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("velocity must be a finite three-vector")
        self.velocity_hat = value.copy()
        self.initialized = True
        return self.disturbance_hat.copy()

    def update(
        self,
        velocity: np.ndarray,
        model_acceleration: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance the observer and return the current disturbance estimate."""
        measured = np.asarray(velocity, dtype=float)
        model = np.asarray(model_acceleration, dtype=float)
        if measured.shape != (3,) or model.shape != (3,):
            raise ValueError("velocity and model_acceleration must have shape (3,)")
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(model)):
            raise ValueError("velocity and model_acceleration must be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            return self.disturbance_hat.copy()
        if not self.initialized:
            self.reset(measured)
            return self.disturbance_hat.copy()
        step = min(float(dt), 0.1)
        innovation = np.clip(
            measured - self.velocity_hat,
            -self.innovation_limit,
            self.innovation_limit,
        )
        self.velocity_hat += step * (
            model + self.disturbance_hat + self.beta1 * innovation
        )
        self.disturbance_hat += step * (self.beta2 * innovation)
        np.clip(
            self.disturbance_hat,
            -self.disturbance_clamp,
            self.disturbance_clamp,
            out=self.disturbance_hat,
        )
        return self.disturbance_hat.copy()
