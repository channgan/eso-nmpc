"""Slow direct-shooting NMPC used only for desktop mathematical validation.

The deployment backend is acados. This implementation deliberately shares the
same model, cost, constraints and reference interface so the project remains
executable before acados has been installed.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ..config import NmpcConfig
from ..model.quadrotor import (
    NP,
    NU,
    NX,
    QuadrotorModel,
    normalize_quaternion,
    quaternion_attitude_error,
)
from ..reference import align_reference_quaternions
from ..types import Control, Reference


class ScipyNmpc:
    def __init__(self, config: NmpcConfig, max_iterations: int = 30) -> None:
        try:
            from scipy.optimize import minimize
        except ImportError as error:
            raise RuntimeError("SciPy is required for the validation backend") from error
        self._minimize = minimize
        self.config = config
        self.max_iterations = max_iterations
        self.model = QuadrotorModel(config.model.mass, config.model.gravity, config.model.rate_tau)
        self._guess: np.ndarray | None = None
        self.last_solve_time = 0.0
        self.last_status = 0
        self.last_message = ""
        self._predicted_states = np.empty((0, NX))
        self._predicted_controls = np.empty((0, NU))

    def _rollout(self, state: np.ndarray, controls: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        dt = self.config.controller.sample_time
        states = np.empty((controls.shape[0] + 1, NX))
        states[0] = state
        for stage, control in enumerate(controls):
            states[stage + 1] = self.model.step_rk4(states[stage], control, dt, disturbance)
        return states

    def _cost(
        self,
        flat_controls: np.ndarray,
        state: np.ndarray,
        reference: Reference,
        disturbance: np.ndarray,
    ) -> float:
        n = self.config.controller.horizon_steps
        controls = flat_controls.reshape(n, NU)
        states = self._rollout(state, controls, disturbance)
        state_weights = self.config.cost_scales.state_weights
        q = state_weights[:6]
        attitude_weights = state_weights[6:9]
        r = self.config.cost_scales.control_weights
        state_error = states[:-1, :6] - reference.states[:-1, :6]
        attitude_error = np.vstack(
            [
                quaternion_attitude_error(actual[6:10], desired[6:10])
                for actual, desired in zip(states[:-1], reference.states[:-1])
            ]
        )
        # Feedback correction around the inverse-dynamics nominal control.
        control_error = controls - reference.feedforward_controls
        stage_cost = np.sum(state_error * state_error * q)
        stage_cost += np.sum(attitude_error * attitude_error * attitude_weights)
        stage_cost += np.sum(control_error * control_error * r)
        terminal_error = states[-1, :6] - reference.states[-1, :6]
        terminal_attitude_error = quaternion_attitude_error(
            states[-1, 6:10], reference.states[-1, 6:10]
        )
        terminal_cost = np.sum(terminal_error * terminal_error * q) + np.sum(
            terminal_attitude_error * terminal_attitude_error * attitude_weights
        )
        return float(self.config.controller.sample_time * stage_cost + terminal_cost)

    def solve(
        self,
        state: np.ndarray,
        reference: Reference,
        disturbance: np.ndarray | None = None,
    ) -> Control:
        state = np.asarray(state, dtype=float).copy()
        disturbance = np.zeros(NP) if disturbance is None else np.asarray(disturbance, dtype=float)
        if state.shape != (NX,) or disturbance.shape != (NP,):
            raise ValueError(f"state must have shape ({NX},) and disturbance shape ({NP},)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(disturbance)):
            raise ValueError("state or disturbance contains a non-finite value")
        state[6:10] = normalize_quaternion(state[6:10])
        n = self.config.controller.horizon_steps
        reference.validate(n)
        reference = align_reference_quaternions(reference, state[6:10])

        if self._guess is None:
            initial = reference.feedforward_controls.copy()
        else:
            initial = np.vstack((self._guess[1:], self._guess[-1]))
        initial[:, 0] = np.clip(
            initial[:, 0], self.config.limits.thrust_min, self.config.limits.thrust_max
        )
        initial[:, 1:4] = np.clip(
            initial[:, 1:4],
            -self.config.limits.body_rate_max,
            self.config.limits.body_rate_max,
        )
        one_stage_bounds = [
            (self.config.limits.thrust_min, self.config.limits.thrust_max),
            *[(-bound, bound) for bound in self.config.limits.body_rate_max],
        ]

        started = perf_counter()
        result = self._minimize(
            self._cost,
            initial.ravel(),
            args=(state, reference, disturbance),
            method="L-BFGS-B",
            bounds=one_stage_bounds * n,
            options={"maxiter": self.max_iterations, "ftol": 1.0e-8, "maxls": 20},
        )
        self.last_solve_time = perf_counter() - started
        self.last_status = int(result.status)
        self.last_message = str(result.message)
        # Reaching the desktop iteration limit still produces a feasible control;
        # only reject non-finite optimization results.
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"SciPy NMPC failed: {result.message}")
        self._guess = result.x.reshape(n, NU)
        self._predicted_controls = self._guess.copy()
        self._predicted_states = self._rollout(state, self._guess, disturbance)
        return Control.from_array(self._guess[0])

    def predicted_states(self) -> np.ndarray:
        return self._predicted_states.copy()

    def predicted_controls(self) -> np.ndarray:
        return self._predicted_controls.copy()
