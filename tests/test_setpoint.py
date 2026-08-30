import numpy as np
import pytest

from nmpc.config import load_config
from nmpc.model.quadrotor import QuadrotorModel
from nmpc.setpoint import (
    KinematicTrajectory,
    KinematicSetpoint,
    PresetTrajectory,
    PresetTrajectoryParameters,
    RcVelocityReference,
    build_reference_from_trajectory,
    apply_deadzone,
    build_reference_horizon,
)


def test_complete_kinematic_trajectory_converts_to_reference() -> None:
    points = 4
    trajectory = KinematicTrajectory(
        position=np.zeros((points, 3)),
        velocity=np.zeros((points, 3)),
        acceleration=np.zeros((points, 3)),
        yaw=np.zeros(points),
        sample_time=0.01,
        jerk=np.zeros((points, 3)),
    )
    reference = build_reference_from_trajectory(
        trajectory,
        horizon_steps=3,
        sample_time=0.01,
        mass=2.0,
        gravity=9.81,
        thrust_min=0.0,
        thrust_max=30.0,
        body_rate_max=np.ones(3),
        quaternion_anchor=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert reference.states.shape == (4, 13)
    assert reference.controls.shape == (3, 4)
    np.testing.assert_allclose(reference.controls[:, 0], 19.62)


def test_complete_trajectory_rejects_wrong_runtime_horizon() -> None:
    trajectory = KinematicTrajectory(
        position=np.zeros((4, 3)),
        velocity=np.zeros((4, 3)),
        acceleration=np.zeros((4, 3)),
        yaw=np.zeros(4),
        sample_time=0.02,
    )
    with np.testing.assert_raises_regex(ValueError, "sample_time"):
        trajectory.validate(horizon_steps=3, expected_sample_time=0.01)


def test_complete_trajectory_rejects_motion_limit_violation() -> None:
    trajectory = KinematicTrajectory(
        position=np.zeros((4, 3)),
        velocity=np.tile([2.1, 0.0, 0.0], (4, 1)),
        acceleration=np.zeros((4, 3)),
        yaw=np.zeros(4),
        sample_time=0.01,
        jerk=np.zeros((4, 3)),
    )
    with np.testing.assert_raises_regex(ValueError, "horizontal speed"):
        trajectory.validate_motion_limits(
            horizontal_speed_max=2.0,
            vertical_speed_max_up=3.0,
            vertical_speed_max_down=1.5,
            horizontal_acceleration_max=2.0,
            vertical_acceleration_max_up=4.0,
            vertical_acceleration_max_down=3.0,
            jerk_max=4.0,
        )


def test_complete_trajectory_does_not_apply_an_extra_jerk_constraint() -> None:
    acceleration = np.zeros((4, 3))
    acceleration[:, 0] = np.arange(4) * 0.01
    trajectory = KinematicTrajectory(
        position=np.zeros((4, 3)),
        velocity=np.zeros((4, 3)),
        acceleration=acceleration,
        jerk=np.zeros((4, 3)),
        yaw=np.zeros(4),
        sample_time=0.01,
    )
    trajectory.validate_motion_limits(
        horizontal_speed_max=2.0,
        vertical_speed_max_up=3.0,
        vertical_speed_max_down=1.5,
        horizontal_acceleration_max=2.0,
        vertical_acceleration_max_up=4.0,
        vertical_acceleration_max_down=3.0,
        jerk_max=4.0,
        jerk_consistency_tolerance=0.1,
    )


def test_jerk_is_ignored_by_nmpc_feedforward_on_test_branch() -> None:
    points = 4
    sample_time = 0.01
    jerk = np.tile([1.0, 0.0, 0.0], (points, 1))
    acceleration = np.arange(points)[:, None] * sample_time * jerk
    trajectory = KinematicTrajectory(
        position=np.zeros((points, 3)),
        velocity=np.zeros((points, 3)),
        acceleration=acceleration,
        jerk=jerk,
        yaw=np.zeros(points),
        yaw_rate=np.zeros(points),
        sample_time=sample_time,
    )
    reference = build_reference_from_trajectory(
        trajectory,
        horizon_steps=points - 1,
        sample_time=sample_time,
        mass=2.0,
        gravity=9.81,
        thrust_min=0.0,
        thrust_max=30.0,
        body_rate_max=np.ones(3),
        quaternion_anchor=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    no_jerk_reference = build_reference_from_trajectory(
        KinematicTrajectory(
            position=trajectory.position,
            velocity=trajectory.velocity,
            acceleration=trajectory.acceleration,
            jerk=np.zeros_like(jerk),
            yaw=trajectory.yaw,
            yaw_rate=trajectory.yaw_rate,
            sample_time=sample_time,
        ),
        horizon_steps=points - 1,
        sample_time=sample_time,
        mass=2.0,
        gravity=9.81,
        thrust_min=0.0,
        thrust_max=30.0,
        body_rate_max=np.ones(3),
        quaternion_anchor=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(reference.controls, no_jerk_reference.controls)


def _circle_parameters() -> PresetTrajectoryParameters:
    return PresetTrajectoryParameters(
        mode="circle",
        altitude=1.0,
        ascent=4.0,
        hold=6.0,
        transition=3.0,
        descent=4.0,
        settle=1.5,
        radius=0.5,
        speed=0.25,
    )


def _parameters(mode: str) -> PresetTrajectoryParameters:
    return PresetTrajectoryParameters(
        mode=mode, altitude=1.0, ascent=4.0, hold=2.0, transition=3.0,
        descent=4.0, settle=1.5, radius=0.5, speed=0.25,
    )


def test_deadzone_is_continuous_and_rescaled() -> None:
    assert apply_deadzone(0.05, 0.08) == 0.0
    assert apply_deadzone(-0.08, 0.08) == 0.0
    assert apply_deadzone(1.0, 0.08) == pytest.approx(1.0)
    assert apply_deadzone(-1.0, 0.08) == pytest.approx(-1.0)
    assert apply_deadzone(0.54, 0.08) == pytest.approx(0.5)


def test_circle_preset_is_continuous_at_every_segment_boundary() -> None:
    source = PresetTrajectory(np.zeros(3), 0.3, _circle_parameters())
    p = source.parameters
    boundaries = [
        p.ascent,
        p.ascent + p.transition,
        p.ascent + p.transition + source.circle_duration,
        p.ascent + 2.0 * p.transition + source.circle_duration,
    ]
    epsilon = 1.0e-7
    for boundary in boundaries:
        left = source.sample(boundary - epsilon)
        right = source.sample(boundary + epsilon)
        np.testing.assert_allclose(left.position, right.position, atol=2.0e-7)
        np.testing.assert_allclose(left.velocity, right.velocity, atol=2.0e-7)
        np.testing.assert_allclose(left.acceleration, right.acceleration, atol=2.0e-7)


def test_step_preset_visits_all_four_cardinal_directions_and_returns() -> None:
    source = PresetTrajectory(np.zeros(3), 0.0, _parameters("step"))
    samples = [source.sample(4.0 + (index + 0.5) * 2.0) for index in range(9)]
    expected = np.array(
        [[0, 0, -1], [.5, 0, -1], [0, 0, -1], [-.5, 0, -1],
         [0, 0, -1], [0, .5, -1], [0, 0, -1], [0, -.5, -1], [0, 0, -1]]
    )
    np.testing.assert_allclose([sample.position for sample in samples], expected)


def test_figure8_crosses_center_and_is_periodic() -> None:
    source = PresetTrajectory(np.zeros(3), 0.0, _parameters("figure8"))
    start = source.parameters.ascent + source.parameters.transition
    first = source.sample(start)
    middle = source.sample(start + 0.5 * source.circle_duration)
    end = source.sample(start + source.circle_duration)
    np.testing.assert_allclose(first.position, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(middle.position, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(end.position, [0.0, 0.0, -1.0], atol=1e-12)


def test_rc_forward_command_uses_heading_frame_and_acceleration_limit() -> None:
    config = load_config().manual_control
    source = RcVelocityReference(np.array([0.0, 0.0, -1.0]), 0.0, config, -2.0, -0.2)
    source.set_sticks(roll=0.0, pitch=1.0, yaw=0.0, throttle=0.0)
    source.step(0.1, np.array([0.0, 0.0, -1.0]))
    expected_speed = config.max_horizontal_acceleration * 0.1
    np.testing.assert_allclose(source.velocity, [expected_speed, 0.0, 0.0], atol=1.0e-12)

    east_facing = RcVelocityReference(
        np.array([0.0, 0.0, -1.0]), np.pi / 2.0, config, -2.0, -0.2
    )
    east_facing.set_sticks(roll=0.0, pitch=1.0, yaw=0.0, throttle=0.0)
    east_facing.step(0.1, np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(east_facing.velocity, [0.0, expected_speed, 0.0], atol=1.0e-12)


def test_rc_positive_throttle_commands_up_in_ned() -> None:
    config = load_config().manual_control
    source = RcVelocityReference(np.array([0.0, 0.0, -1.0]), 0.0, config, -2.0, -0.2)
    source.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=1.0)
    source.step(0.1, np.array([0.0, 0.0, -1.0]))
    assert source.velocity[2] == pytest.approx(-0.06)


def test_rc_reference_lead_is_bounded() -> None:
    config = load_config().manual_control
    measured = np.array([0.0, 0.0, -1.0])
    source = RcVelocityReference(measured, 0.0, config, -2.0, -0.2)
    source.set_sticks(roll=0.0, pitch=1.0, yaw=0.0, throttle=0.0)
    for _ in range(100):
        source.step(0.1, measured)
    assert np.linalg.norm(source.position[:2] - measured[:2]) == pytest.approx(
        config.max_horizontal_position_lead
    )


def test_generic_horizon_contains_rc_yaw_rate_feedforward() -> None:
    config = load_config()
    source = RcVelocityReference(
        np.array([0.0, 0.0, -1.0]), 0.0, config.manual_control, -2.0, -0.2
    )
    source.set_sticks(roll=0.0, pitch=0.0, yaw=1.0, throttle=0.0)
    source.step(1.0, np.array([0.0, 0.0, -1.0]))
    reference = build_reference_horizon(
        source.sample,
        start_time=0.0,
        horizon_steps=config.controller.horizon_steps,
        sample_time=config.controller.sample_time,
        mass=config.model.mass,
        gravity=config.model.gravity,
        thrust_min=config.limits.thrust_min,
        thrust_max=config.limits.thrust_max,
        body_rate_max=config.limits.body_rate_max,
        quaternion_anchor=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    reference.validate(config.controller.horizon_steps)
    assert reference.controls[0, 3] == pytest.approx(config.manual_control.max_yaw_rate)


def test_inverse_dynamics_horizon_reproduces_requested_acceleration() -> None:
    config = load_config()
    requested_acceleration = np.array([0.8, -0.4, 0.2])
    disturbance = np.array([0.1, 0.2, -0.05])

    def sample(time_s: float) -> KinematicSetpoint:
        return KinematicSetpoint(
            0.5 * requested_acceleration * time_s**2,
            requested_acceleration * time_s,
            requested_acceleration,
            0.3,
            "constant_acceleration",
        )

    reference = build_reference_horizon(
        sample,
        start_time=0.0,
        horizon_steps=config.controller.horizon_steps,
        sample_time=config.controller.sample_time,
        mass=config.model.mass,
        gravity=config.model.gravity,
        thrust_min=config.limits.thrust_min,
        thrust_max=config.limits.thrust_max,
        body_rate_max=config.limits.body_rate_max,
        quaternion_anchor=np.array([1.0, 0.0, 0.0, 0.0]),
        disturbance=disturbance,
    )
    model = QuadrotorModel(config.model.mass, config.model.gravity)
    for stage in range(config.controller.horizon_steps):
        derivative = model.continuous_dynamics(
            reference.states[stage],
            reference.feedforward_controls[stage],
            disturbance,
        )
        np.testing.assert_allclose(derivative[3:6], requested_acceleration, atol=1e-12)


def test_infeasible_inverse_dynamics_feedforward_is_not_silently_clipped() -> None:
    config = load_config()

    def sample(_: float) -> KinematicSetpoint:
        return KinematicSetpoint(
            np.zeros(3), np.zeros(3), np.array([0.0, 0.0, -20.0]), 0.0, "unsafe"
        )

    with pytest.raises(ValueError, match="thrust feed-forward"):
        build_reference_horizon(
            sample,
            start_time=0.0,
            horizon_steps=config.controller.horizon_steps,
            sample_time=config.controller.sample_time,
            mass=config.model.mass,
            gravity=config.model.gravity,
            thrust_min=config.limits.thrust_min,
            thrust_max=config.limits.thrust_max,
            body_rate_max=config.limits.body_rate_max,
            quaternion_anchor=np.array([1.0, 0.0, 0.0, 0.0]),
        )
