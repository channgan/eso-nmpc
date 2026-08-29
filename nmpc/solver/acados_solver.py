"""acados OCP definition and runtime wrapper."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..config import NmpcConfig
from ..model.quadrotor import ACADOS_NP, NP, NU, NX, QuadrotorModel, normalize_quaternion
from ..reference import align_reference_quaternions
from ..types import Control, Reference


def _require_acados() -> tuple[Any, Any]:
    try:
        from acados_template import AcadosOcp, AcadosOcpSolver
    except ImportError as error:
        raise RuntimeError(
            "acados_template is unavailable. Follow the acados installation steps in "
            "README.md and export ACADOS_SOURCE_DIR."
        ) from error
    return AcadosOcp, AcadosOcpSolver


def build_ocp(config: NmpcConfig) -> Any:
    """Construct the acados OCP without generating or compiling code."""
    AcadosOcp, _ = _require_acados()
    model = QuadrotorModel(config.model.mass, config.model.gravity, config.model.rate_tau)
    ocp = AcadosOcp()
    ocp.model = model.export_acados_model(config.code_generation.model_name)
    n = config.controller.horizon_steps
    ocp.solver_options.N_horizon = n

    # Nonlinear least-squares output uses the SO(3) logarithm of the relative
    # quaternion. The 10-state quaternion model is retained, while its cost is
    # a geometrically meaningful three-dimensional attitude error.
    import casadi as ca

    x = ocp.model.x
    u = ocp.model.u
    q = x[6:10]
    q_reference = ocp.model.p[NP:ACADOS_NP]
    wr, xr, yr, zr = (q_reference[i] for i in range(4))
    w, qx, qy, qz = (q[i] for i in range(4))
    error_scalar = wr * w + xr * qx + yr * qy + zr * qz
    error_vector = ca.vertcat(
        wr * qx - xr * w - yr * qz + zr * qy,
        wr * qy + xr * qz - yr * w - zr * qx,
        wr * qz - xr * qy + yr * qx - zr * w,
    )
    # Choose the quaternion hemisphere giving the principal rotation (<= pi).
    signed_vector = ca.if_else(error_scalar >= 0.0, error_vector, -error_vector)
    positive_scalar = ca.fabs(error_scalar)
    vector_norm = ca.norm_2(signed_vector)
    rotation_vector = ca.if_else(
        vector_norm > 1.0e-8,
        2.0 * ca.atan2(vector_norm, positive_scalar) * signed_vector / vector_norm,
        2.0 * signed_vector,
    )
    stage_output = ca.vertcat(x[:6], rotation_vector, u)
    terminal_output = ca.vertcat(x[:6], rotation_vector)
    state_weights = np.r_[
        config.weights.position,
        config.weights.velocity,
        config.weights.attitude,
    ]
    # R penalizes the feedback correction around inverse-dynamics feed-forward:
    # (u - u_ff)' R (u - u_ff). It does not penalize absolute thrust/rate.
    control_deviation_weights = np.r_[config.weights.thrust, config.weights.body_rate]
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(np.r_[state_weights, control_deviation_weights])
    ocp.cost.W_e = config.weights.terminal_factor * np.diag(state_weights)
    ocp.model.cost_y_expr = stage_output
    ocp.model.cost_y_expr_e = terminal_output
    hover = np.array([config.hover_thrust, 0.0, 0.0, 0.0])
    ocp.cost.yref = np.r_[np.zeros(9), hover]
    ocp.cost.yref_e = np.zeros(9)

    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.r_[config.limits.thrust_min, -config.limits.body_rate_max]
    ocp.constraints.ubu = np.r_[config.limits.thrust_max, config.limits.body_rate_max]

    # PX4-compatible state envelope. NED vertical velocity is negative upward
    # and positive downward. Horizontal speed is constrained as a norm so the
    # limit is independent of direction. R_33 = 1 - 2(qx^2 + qy^2) is the
    # cosine of tilt for an upright body-to-NED quaternion. Quaternion unit
    # length is enforced by normalized initial/reference states and the
    # norm-preserving quaternion dynamics; adding q.T@q == 1 here would be a
    # redundant equality that makes the SQP-RTI QP rank-deficient.
    ocp.constraints.idxbx = np.array([5])
    ocp.constraints.lbx = np.array([-config.limits.vertical_speed_max_up])
    ocp.constraints.ubx = np.array([config.limits.vertical_speed_max_down])
    ocp.constraints.idxbx_e = np.array([5])
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()

    horizontal_speed_squared = x[3] ** 2 + x[4] ** 2
    upright_cosine = 1.0 - 2.0 * (x[7] ** 2 + x[8] ** 2)
    path_constraints = ca.vertcat(
        horizontal_speed_squared,
        upright_cosine,
    )
    ocp.model.con_h_expr = path_constraints
    ocp.model.con_h_expr_e = path_constraints
    nonlinear_lower = np.array(
        [
            -1.0e9,
            np.cos(np.deg2rad(config.limits.tilt_max_deg)),
        ]
    )
    nonlinear_upper = np.array(
        [
            config.limits.horizontal_speed_max**2,
            1.0e9,
        ]
    )
    ocp.constraints.lh = nonlinear_lower
    ocp.constraints.uh = nonlinear_upper
    ocp.constraints.lh_e = nonlinear_lower.copy()
    ocp.constraints.uh_e = nonlinear_upper.copy()
    ocp.constraints.x0 = np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
    ocp.parameter_values = np.r_[np.zeros(NP), [1.0, 0.0, 0.0, 0.0]]

    options = ocp.solver_options
    options.tf = n * config.controller.sample_time
    options.nlp_solver_type = config.controller.nlp_solver_type
    options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    options.qp_solver_cond_N = config.controller.qp_solver_cond_N
    options.integrator_type = config.controller.integrator_type
    options.hessian_approx = "GAUSS_NEWTON"
    options.levenberg_marquardt = config.controller.levenberg_marquardt
    options.print_level = 0
    ocp.code_gen_options.code_export_directory = config.code_generation.code_export_directory
    ocp.code_gen_options.json_file = config.code_generation.json_file
    return ocp


def generate_solver(config: NmpcConfig, build: bool = True) -> Any:
    """Generate (and normally compile) the native solver."""
    _, AcadosOcpSolver = _require_acados()
    json_path = Path(config.code_generation.json_file)
    if json_path.name != str(json_path):
        raise ValueError("acados json_file must be a basename without directory components")
    Path(config.code_generation.code_export_directory).parent.mkdir(parents=True, exist_ok=True)
    ocp = build_ocp(config)
    if not build:
        # Calling the constructor with build=False can intentionally switch build
        # back on when no reusable artifacts exist. The static generator is the
        # unambiguous generate-only API.
        AcadosOcpSolver.generate(ocp, json_file=str(json_path))
        return None
    return AcadosOcpSolver(
        ocp,
        generate=True,
        build=True,
    )


def load_or_generate_solver(config: NmpcConfig) -> Any:
    """Reuse matching generated artifacts, generating them only when necessary."""
    _, AcadosOcpSolver = _require_acados()
    json_file = config.code_generation.json_file
    if Path(json_file).name != json_file:
        raise ValueError("acados json_file must be a basename without directory components")
    # Current acados checks its formulation hash. If files are absent or stale,
    # it automatically changes these flags back to True and rebuilds safely.
    return AcadosOcpSolver(
        build_ocp(config),
        generate=False,
        build=False,
    )


class AcadosNmpc:
    """Stateful SQP-RTI controller with warm-start retained by acados."""

    def __init__(self, config: NmpcConfig, solver: Any | None = None) -> None:
        self.config = config
        self._solver = solver if solver is not None else load_or_generate_solver(config)
        self.last_solve_time = 0.0
        self.last_status = 0
        self._last_states: np.ndarray | None = None
        self._last_controls: np.ndarray | None = None

    def solve(
        self,
        state: np.ndarray,
        reference: Reference,
        disturbance: np.ndarray | None = None,
    ) -> Control:
        state = np.asarray(state, dtype=float).copy()
        disturbance = np.zeros(NP) if disturbance is None else np.asarray(disturbance, dtype=float)
        if state.shape != (NX,):
            raise ValueError(f"state must have shape ({NX},), got {state.shape}")
        if disturbance.shape != (NP,):
            raise ValueError(f"disturbance must have shape ({NP},), got {disturbance.shape}")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(disturbance)):
            raise ValueError("state or disturbance contains a non-finite value")
        state[6:10] = normalize_quaternion(state[6:10])
        n = self.config.controller.horizon_steps
        reference.validate(n)
        reference = align_reference_quaternions(reference, state[6:10])

        # SQP-RTI needs a meaningful linearization trajectory.  Linearize
        # around the reference trajectory itself rather than continuing the
        # previous solution: with a single QP iteration, a shifted previous
        # plan carries its (already rotated) quaternion states into the new
        # linearization, and the QP then commands large rates to correct its
        # own guess's divergence -- sustaining an attitude limit cycle against
        # the ~0.2 s PX4 rate-loop delay.  Seeding from the reference keeps
        # every stage's linearized error small, so u0 stays proportional to
        # the true stage-0 error.
        state_guess = reference.states.copy()
        control_guess = reference.feedforward_controls.copy()
        state_guess[0] = state
        for stage in range(n):
            self._solver.set(stage, "x", state_guess[stage])
            self._solver.set(stage, "u", control_guess[stage])
        self._solver.set(n, "x", state_guess[n])

        self._solver.set(0, "lbx", state)
        self._solver.set(0, "ubx", state)
        for stage in range(n):
            self._solver.set(
                stage, "p", np.r_[disturbance, reference.states[stage, 6:10]]
            )
            self._solver.set(
                stage,
                "yref",
                np.r_[reference.states[stage, :6], np.zeros(3), reference.feedforward_controls[stage]],
            )
        self._solver.set(n, "p", np.r_[disturbance, reference.states[n, 6:10]])
        self._solver.set(n, "yref", np.r_[reference.states[n, :6], np.zeros(3)])

        started = perf_counter()
        self.last_status = int(self._solver.solve())
        wall_solve_time = perf_counter() - started
        self.last_solve_time = wall_solve_time
        if hasattr(self._solver, "get_stats"):
            # Some acados builds occasionally expose a negative or otherwise
            # invalid time_tot value.  A duration cannot be negative, so retain
            # the monotonic wall-clock measurement in that case instead of
            # contaminating benchmark percentiles.
            try:
                reported_solve_time = float(self._solver.get_stats("time_tot"))
            except (TypeError, ValueError):
                reported_solve_time = float("nan")
            if np.isfinite(reported_solve_time) and reported_solve_time >= 0.0:
                self.last_solve_time = reported_solve_time
        if self.last_status != 0:
            raise RuntimeError(f"acados solve failed with status {self.last_status}")
        self._last_states = np.vstack(
            [np.asarray(self._solver.get(i, "x"), dtype=float).reshape(NX) for i in range(n + 1)]
        )
        self._last_controls = np.vstack(
            [np.asarray(self._solver.get(i, "u"), dtype=float).reshape(NU) for i in range(n)]
        )
        return Control.from_array(self._last_controls[0])

    def predicted_states(self) -> np.ndarray:
        if self._last_states is None:
            return np.empty((0, NX))
        return self._last_states.copy()

    def predicted_controls(self) -> np.ndarray:
        if self._last_controls is None:
            return np.empty((0, NU))
        return self._last_controls.copy()
