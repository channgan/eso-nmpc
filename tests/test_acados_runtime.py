from dataclasses import replace

import numpy as np

from nmpc.config import load_config
from nmpc.reference import stationary_reference
from nmpc.types import Reference
from nmpc.solver.acados_solver import AcadosNmpc


class FakeAcadosSolver:
    def __init__(self, control: np.ndarray) -> None:
        self.control = control
        self.values: dict[tuple[int, str], np.ndarray] = {}

    def set(self, stage: int, field: str, value: np.ndarray) -> None:
        self.values[(stage, field)] = np.asarray(value).copy()

    def solve(self) -> int:
        return 0

    def get(self, stage: int, field: str) -> np.ndarray:
        if field == "u":
            return self.control.copy()
        if field == "x":
            return np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
        if field == "pi":
            return np.zeros(13)
        raise KeyError(field)


class FakeAcadosSolverWithInvalidTiming(FakeAcadosSolver):
    def get_stats(self, field: str) -> float:
        assert field == "time_tot"
        return -1.0


def test_runtime_sets_state_reference_and_disturbance_at_every_stage() -> None:
    config = load_config()
    expected_control = np.array([config.hover_thrust, 0.1, -0.2, 0.3])
    fake = FakeAcadosSolver(expected_control)
    controller = AcadosNmpc(config, solver=fake)
    state = np.r_[[0.2, 0.1, -0.3], np.zeros(3), [2.0, 0.0, 0.0, 0.0], np.zeros(3)]
    reference = stationary_reference(
        np.array([0.0, 0.0, -1.0]), config.controller.horizon_steps, config.hover_thrust
    )
    disturbance = np.array([0.5, -0.1, 0.2])
    result = controller.solve(state, reference, disturbance)

    np.testing.assert_allclose(result.as_array(), expected_control)
    np.testing.assert_allclose(fake.values[(0, "lbx")][6:10], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fake.values[(0, "ubx")], fake.values[(0, "lbx")])
    for stage in range(config.controller.horizon_steps + 1):
        np.testing.assert_allclose(fake.values[(stage, "p")][:3], disturbance)
        np.testing.assert_allclose(
            fake.values[(stage, "p")][3:7], reference.states[stage, 6:10]
        )
    assert fake.values[(0, "yref")].shape == (13,)
    np.testing.assert_allclose(
        fake.values[(0, "yref")][9:13], reference.feedforward_controls[0]
    )
    assert fake.values[(config.controller.horizon_steps, "yref")].shape == (9,)


def test_runtime_rejects_negative_native_solve_time() -> None:
    config = load_config()
    fake = FakeAcadosSolverWithInvalidTiming(
        np.array([config.hover_thrust, 0.0, 0.0, 0.0])
    )
    controller = AcadosNmpc(config, solver=fake)
    state = np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
    reference = stationary_reference(
        np.zeros(3), config.controller.horizon_steps, config.hover_thrust
    )

    controller.solve(state, reference)

    assert np.isfinite(controller.last_solve_time)
    assert controller.last_solve_time >= 0.0


class FakeAcadosSolverWithTrajectory(FakeAcadosSolver):
    def get(self, stage: int, field: str) -> np.ndarray:
        if field == "u":
            return self.control + np.array([0.1 * stage, 0.01 * stage, 0.0, 0.0])
        if field == "x":
            return np.r_[[float(stage), 0.0, -1.0], np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
        if field == "pi":
            return np.full(13, float(stage + 1))
        raise KeyError(field)


def test_runtime_warm_starts_shifted_state_control_and_multiplier() -> None:
    # The repository default is the cold-start deployment baseline, but keep
    # this unit test focused on the warm-start implementation itself.
    loaded = load_config()
    config = replace(loaded, controller=replace(loaded.controller, warm_start=True))
    previous_control = np.array([config.hover_thrust, 0.1, -0.2, 0.3])
    fake = FakeAcadosSolverWithTrajectory(previous_control)
    controller = AcadosNmpc(config, solver=fake)
    state = np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
    first = stationary_reference(
        np.array([0.0, 0.0, -1.0]), config.controller.horizon_steps, config.hover_thrust
    )
    controller.solve(state, first)

    shifted_feedforward = first.feedforward_controls.copy()
    shifted_feedforward[:, 0] += 1.0
    second = Reference(states=first.states.copy(), controls=shifted_feedforward)
    controller.solve(state, second)

    assert controller.warm_start_used
    np.testing.assert_allclose(
        fake.values[(0, "u")], previous_control + np.array([0.1, 0.01, 0.0, 0.0])
    )
    np.testing.assert_allclose(fake.values[(0, "x")], state)
    np.testing.assert_allclose(fake.values[(1, "x")][0], 2.0)
    np.testing.assert_allclose(fake.values[(0, "pi")], np.full(13, 2.0))
    np.testing.assert_allclose(
        fake.values[(config.controller.horizon_steps - 1, "pi")],
        np.full(13, float(config.controller.horizon_steps)),
    )
