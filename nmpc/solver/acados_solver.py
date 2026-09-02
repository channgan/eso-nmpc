"""acados OCP definition and runtime wrapper."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..config import NmpcConfig
from ..model.quadrotor import (
    ACADOS_NP,
    NP,
    NU,
    NX,
    QuadrotorModel,
    normalize_quaternion,
)
from ..reference import align_reference_quaternions
from ..reference import align_quaternion_sequence
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
    # Each physical residual is normalized by its configured engineering
    # scale.  In physical coordinates this is W_i = weight_factor / scale_i^2;
    # no additional state/control priority multipliers are applied.
    state_weights = config.cost_scales.state_weights
    # R penalizes the feedback correction around inverse-dynamics feed-forward:
    # (u - u_ff)' R (u - u_ff). It does not penalize absolute thrust/rate.
    control_deviation_weights = config.cost_scales.control_weights
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(np.r_[state_weights, control_deviation_weights])
    ocp.cost.W_e = np.diag(state_weights)
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
    """Stateful SQP-RTI controller with shifted primal/dual warm start."""

    def __init__(self, config: NmpcConfig, solver: Any | None = None) -> None:
        self.config = config
        self._solver = solver if solver is not None else load_or_generate_solver(config)
        self.last_solve_time = 0.0
        self.last_status = 0
        self._last_states: np.ndarray | None = None
        self._last_controls: np.ndarray | None = None
        self._last_dynamics_multipliers: np.ndarray | None = None
        self.warm_start_used = False
        self.last_timing: dict[str, float] = {}
        self.last_solver_stats: dict[str, float] = {}

    def reset_warm_start(self) -> None:
        """Discard the previous SQP-RTI iterate before a discontinuity."""
        self._last_states = None
        self._last_controls = None
        self._last_dynamics_multipliers = None
        self.warm_start_used = False

    def solve(
        self,
        state: np.ndarray,
        reference: Reference,
        disturbance: np.ndarray | None = None,
    ) -> Control:
        wrapper_started = perf_counter()
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
        input_validation_finished = perf_counter()

        # Standard receding-horizon initialization: shift the previous optimal
        # state, control, and dynamics-multiplier trajectories forward by one
        # shooting interval, and repeat their terminal values.  The measured
        # state below always replaces x[0].  A cold start uses the current
        # reference and zero multipliers.
        state_guess = reference.states.copy()
        control_guess = reference.feedforward_controls.copy()
        dynamics_multiplier_guess = np.zeros((n, NX))
        self.warm_start_used = False
        if (
            self.config.controller.warm_start
            and self._last_states is not None
            and self._last_controls is not None
            and self._last_dynamics_multipliers is not None
            and self._last_states.shape == (n + 1, NX)
            and self._last_controls.shape == (n, NU)
            and self._last_dynamics_multipliers.shape == (n, NX)
        ):
            state_guess = np.vstack((self._last_states[1:], self._last_states[-1]))
            control_guess = np.vstack((self._last_controls[1:], self._last_controls[-1]))
            dynamics_multiplier_guess = np.vstack(
                (self._last_dynamics_multipliers[1:], self._last_dynamics_multipliers[-1])
            )

            # Quaternion signs are not physical states.  Keep the shifted
            # trajectory normalized and in one hemisphere before acados
            # linearizes around it.
            state_guess[:, 6:10] = align_quaternion_sequence(
                state_guess[:, 6:10], state[6:10]
            )

            control_guess[:, 0] = np.clip(
                control_guess[:, 0], self.config.limits.thrust_min, self.config.limits.thrust_max
            )
            control_guess[:, 1:4] = np.clip(
                control_guess[:, 1:4],
                -self.config.limits.body_rate_max,
                self.config.limits.body_rate_max,
            )
            self.warm_start_used = True
        warm_start_finished = perf_counter()
        state_guess[0] = state
        for stage in range(n):
            self._solver.set(stage, "x", state_guess[stage])
            self._solver.set(stage, "u", control_guess[stage])
            self._solver.set(stage, "pi", dynamics_multiplier_guess[stage])
        self._solver.set(n, "x", state_guess[n])
        initial_guess_finished = perf_counter()

        bounds_started = perf_counter()
        self._solver.set(0, "lbx", state)
        self._solver.set(0, "ubx", state)
        bounds_finished = perf_counter()
        parameters_started = perf_counter()
        for stage in range(n):
            self._solver.set(
                stage, "p", np.r_[disturbance, reference.states[stage, 6:10]]
            )
        self._solver.set(n, "p", np.r_[disturbance, reference.states[n, 6:10]])
        parameters_finished = perf_counter()
        yref_started = perf_counter()
        for stage in range(n):
            self._solver.set(
                stage,
                "yref",
                np.r_[reference.states[stage, :6], np.zeros(3), reference.feedforward_controls[stage]],
            )
        self._solver.set(n, "yref", np.r_[reference.states[n, :6], np.zeros(3)])
        yref_finished = perf_counter()
        set_finished = perf_counter()

        started = set_finished
        self.last_status = int(self._solver.solve())
        solve_finished = perf_counter()
        wall_solve_time = solve_finished - started
        self.last_solve_time = wall_solve_time
        stats_started = perf_counter()
        self.last_solver_stats = {}
        if hasattr(self._solver, "get_stats"):
            # Some acados builds occasionally expose a negative or otherwise
            # invalid time_tot value.  A duration cannot be negative, so retain
            # the monotonic wall-clock measurement in that case instead of
            # contaminating benchmark percentiles.
            reported_solve_time = float("nan")
            for field in (
                "time_tot",
                "time_qp",
                "time_qp_xcond",
                "time_qp_solver_call",
                "time_qpscaling",
                "time_lin",
                "time_sim",
                "time_reg",
                "time_preparation",
            ):
                try:
                    value = float(self._solver.get_stats(field))
                except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
                    value = float("nan")
                if np.isfinite(value) and value >= 0.0:
                    self.last_solver_stats[field] = value
                if field == "time_tot":
                    reported_solve_time = value
            # Keep the host monotonic measurement as the authoritative pure
            # solve duration.  Native ``time_tot`` is retained in
            # ``last_solver_stats`` for diagnosis; some builds report it in a
            # different unit or occasionally return an implausible outlier.
        stats_finished = perf_counter()
        if self.last_status != 0:
            # Do not carry a failed primal guess into the next SQP-RTI call.
            self._last_states = None
            self._last_controls = None
            self._last_dynamics_multipliers = None
            raise RuntimeError(f"acados solve failed with status {self.last_status}")
        extraction_started = perf_counter()
        self._last_states = np.vstack(
            [np.asarray(self._solver.get(i, "x"), dtype=float).reshape(NX) for i in range(n + 1)]
        )
        self._last_controls = np.vstack(
            [np.asarray(self._solver.get(i, "u"), dtype=float).reshape(NU) for i in range(n)]
        )
        self._last_dynamics_multipliers = np.vstack(
            [np.asarray(self._solver.get(i, "pi"), dtype=float).reshape(NX) for i in range(n)]
        )
        extraction_finished = perf_counter()
        self.last_timing = {
            "wrapper_total_ms": 1.0e3 * (extraction_finished - wrapper_started),
            "input_validation_ms": 1.0e3 * (input_validation_finished - wrapper_started),
            "warm_start_initialization_ms": 1.0e3 * (warm_start_finished - input_validation_finished),
            "initial_guess_set_ms": 1.0e3 * (initial_guess_finished - warm_start_finished),
            "parameters_yref_bounds_set_ms": 1.0e3 * (set_finished - initial_guess_finished),
            "state_bounds_set_ms": 1.0e3 * (bounds_finished - bounds_started),
            "stage_parameters_set_ms": 1.0e3 * (parameters_finished - parameters_started),
            "stage_yref_set_ms": 1.0e3 * (yref_finished - yref_started),
            "pure_solver_wall_ms": 1.0e3 * (solve_finished - started),
            "solver_stats_query_ms": 1.0e3 * (stats_finished - solve_finished),
            "trajectory_extraction_ms": 1.0e3 * (extraction_finished - extraction_started),
            "t_set_steady_s": set_finished,
            "t_solve_0_steady_s": started,
            "t_solve_1_steady_s": solve_finished,
        }
        return Control.from_array(self._last_controls[0])

    def predicted_states(self) -> np.ndarray:
        if self._last_states is None:
            return np.empty((0, NX))
        return self._last_states.copy()

    def predicted_controls(self) -> np.ndarray:
        if self._last_controls is None:
            return np.empty((0, NU))
        return self._last_controls.copy()
