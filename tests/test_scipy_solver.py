from dataclasses import replace
import warnings

import numpy as np
import pytest

from nmpc.config import load_config
from nmpc.reference import stationary_reference


def test_hover_solution_is_hover_control() -> None:
    # Import lazily because some base images emit a SciPy/NumPy compatibility warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nmpc.solver.scipy_solver import ScipyNmpc

    config = load_config()
    config = replace(config, controller=replace(config.controller, horizon_steps=3, qp_solver_cond_N=3))
    try:
        controller = ScipyNmpc(config, max_iterations=2)
    except RuntimeError as error:
        pytest.skip(f"compatible SciPy is not installed: {error}")
    state = np.r_[np.array([0.0, 0.0, -1.0]), np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
    reference = stationary_reference(
        state[:3], config.controller.horizon_steps, config.hover_thrust
    )
    control = controller.solve(state, reference).as_array()
    np.testing.assert_allclose(control, [config.hover_thrust, 0.0, 0.0, 0.0], atol=1.0e-8)
