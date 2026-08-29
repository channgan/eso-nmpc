"""Optional native integration test; enabled when ACADOS_SOURCE_DIR is set."""

import importlib.util
import os

import numpy as np
import pytest

from nmpc.config import load_config
from nmpc.reference import stationary_reference


HAS_ACADOS = bool(os.environ.get("ACADOS_SOURCE_DIR")) and importlib.util.find_spec(
    "acados_template"
) is not None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ACADOS, reason="acados native toolchain is not configured")
def test_native_hover_and_disturbance_parameter() -> None:
    from nmpc.solver.acados_solver import AcadosNmpc

    config = load_config()
    controller = AcadosNmpc(config)
    state = np.r_[[0.0, 0.0, -1.0], np.zeros(3), [1.0, 0.0, 0.0, 0.0]]
    reference = stationary_reference(
        state[:3], config.controller.horizon_steps, config.hover_thrust
    )
    hover = controller.solve(state, reference, np.zeros(3)).as_array()
    np.testing.assert_allclose(hover, [config.hover_thrust, 0.0, 0.0, 0.0], atol=1.0e-7)

    disturbed = hover
    for _ in range(3):
        disturbed = controller.solve(state, reference, np.array([0.5, 0.0, 0.0])).as_array()
    assert abs(disturbed[2]) > 0.1
    assert config.limits.thrust_min <= disturbed[0] <= config.limits.thrust_max
