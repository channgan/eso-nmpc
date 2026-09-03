"""Kinematic reference sources shared by integrations and future planners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import ManualControlConfig
from .model.quadrotor import align_quaternion
from .reference import (
    attach_feedforward_rate_states,
    average_body_rate_batch,
    align_quaternion_sequence,
    inverse_dynamics_attitude_and_thrust_batch,
)
from .trajectory import quintic_segment, smooth_profile
from .types import Reference


@dataclass(frozen=True)
class KinematicSetpoint:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    segment: str
    jerk: np.ndarray | None = None
    yaw_rate: float | None = None


@dataclass(frozen=True)
class KinematicTrajectory:
    """A complete external or PX4-generated NMPC kinematic horizon."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: np.ndarray
    sample_time: float
    jerk: np.ndarray | None = None
    yaw_rate: np.ndarray | None = None

    def validate(self, horizon_steps: int, expected_sample_time: float) -> None:
        points = horizon_steps + 1
        vectors = {
            "position": np.asarray(self.position),
            "velocity": np.asarray(self.velocity),
            "acceleration": np.asarray(self.acceleration),
        }
        if self.jerk is not None:
            vectors["jerk"] = np.asarray(self.jerk)
        for name, values in vectors.items():
            if values.shape != (points, 3):
                raise ValueError(f"{name} must have shape ({points}, 3), got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains a non-finite value")
        yaw = np.asarray(self.yaw)
        if yaw.shape != (points,) or not np.all(np.isfinite(yaw)):
            raise ValueError(f"yaw must be a finite array with shape ({points},)")
        if self.yaw_rate is not None:
            yaw_rate = np.asarray(self.yaw_rate)
            if yaw_rate.shape != (points,) or not np.all(np.isfinite(yaw_rate)):
                raise ValueError(f"yaw_rate must be a finite array with shape ({points},)")
        if not np.isfinite(self.sample_time) or self.sample_time <= 0.0:
            raise ValueError("trajectory sample_time must be finite and positive")
        tolerance = max(1.0e-7, 1.0e-4 * expected_sample_time)
        if abs(self.sample_time - expected_sample_time) > tolerance:
            raise ValueError(
                "trajectory sample_time does not match the NMPC controller sample_time"
            )

    def validate_motion_limits(
        self,
        horizontal_speed_max: float,
        vertical_speed_max_up: float,
        vertical_speed_max_down: float,
        horizontal_acceleration_max: float,
        vertical_acceleration_max_up: float,
        vertical_acceleration_max_down: float,
        jerk_max: float,
        jerk_consistency_tolerance: float | None = None,
    ) -> None:
        """Reject a trajectory that bypasses the configured deployment envelope."""
        velocity = np.asarray(self.velocity, dtype=float)
        acceleration = np.asarray(self.acceleration, dtype=float)
        tolerance = 1.0e-5
        if np.any(np.linalg.norm(velocity[:, :2], axis=1) > horizontal_speed_max + tolerance):
            raise ValueError("trajectory exceeds horizontal speed limit")
        if np.any(velocity[:, 2] < -vertical_speed_max_up - tolerance) or np.any(
            velocity[:, 2] > vertical_speed_max_down + tolerance
        ):
            raise ValueError("trajectory exceeds vertical speed limit")
        # The deployment envelope is defined per horizontal axis from the PX4
        # vehicle limits; a 3D norm would be stricter on turns than the stated
        # per-axis contract.
        if np.any(
            np.abs(acceleration[:, :2]) > horizontal_acceleration_max + tolerance
        ):
            raise ValueError("trajectory exceeds horizontal acceleration limit")
        if np.any(acceleration[:, 2] < -vertical_acceleration_max_up - tolerance) or np.any(
            acceleration[:, 2] > vertical_acceleration_max_down + tolerance
        ):
            raise ValueError("trajectory exceeds vertical acceleration limit")
        # Jerk is deliberately not used as an additional NMPC constraint.  The
        # producer of a direct trajectory is responsible for time-parameterizing
        # it smoothly; the complete jerk samples are still carried in the
        # interface for logging and future constraint support.


@dataclass(frozen=True)
class PresetTrajectoryParameters:
    mode: str
    altitude: float
    ascent: float
    hold: float
    transition: float
    descent: float
    settle: float
    radius: float
    speed: float


class PresetTrajectory:
    """Takeoff/landing trajectory for the repeatable SITL regression maneuvers."""

    def __init__(
        self,
        start_position: np.ndarray,
        yaw: float,
        parameters: PresetTrajectoryParameters,
    ) -> None:
        self.start_position = np.asarray(start_position, dtype=float).copy()
        if self.start_position.shape != (3,) or not np.all(np.isfinite(self.start_position)):
            raise ValueError("start_position must be a finite three-vector")
        if parameters.mode not in ("hover", "step", "circle", "figure8"):
            raise ValueError("preset mode must be hover, step, circle or figure8")
        positive = np.array(
            [
                parameters.altitude,
                parameters.ascent,
                parameters.hold,
                parameters.transition,
                parameters.descent,
                parameters.radius,
                parameters.speed,
            ]
        )
        if np.any(positive <= 0.0) or parameters.settle < 0.0:
            raise ValueError("preset trajectory dimensions and durations must be positive")
        self.yaw = float(yaw)
        self.parameters = parameters

    @property
    def circle_duration(self) -> float:
        return 2.0 * np.pi * self.parameters.radius / self.parameters.speed

    @property
    def step_positions(self) -> tuple[np.ndarray, ...]:
        center = self.start_position + np.array([0.0, 0.0, -self.parameters.altitude])
        distance = self.parameters.radius
        offsets = (
            (0.0, 0.0, 0.0),
            (distance, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-distance, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, distance, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -distance, 0.0),
            (0.0, 0.0, 0.0),
        )
        return tuple(center + np.asarray(offset) for offset in offsets)

    @property
    def duration(self) -> float:
        p = self.parameters
        if p.mode == "hover":
            motion = p.ascent + p.hold + p.descent
        elif p.mode == "step":
            motion = p.ascent + len(self.step_positions) * p.hold + p.descent
        else:
            motion = p.ascent + 2.0 * p.transition + self.circle_duration + p.descent
        return motion + p.settle

    @property
    def direct_duration(self) -> float:
        """Duration of the complete, smooth point-sequence trajectory.

        The legacy ``sample`` method represents the step test as held points.
        The direct NMPC interface instead needs a continuous point-to-point
        trajectory, so each move uses a quintic transition followed by the
        configured dwell time.
        """
        p = self.parameters
        if p.mode != "step":
            return self.duration
        sequence = p.hold + (len(self.step_positions) - 1) * (p.transition + p.hold)
        return p.ascent + sequence + p.descent + p.settle

    def sample_direct(self, time_s: float) -> KinematicSetpoint:
        """Sample the trajectory intended for the direct NMPC interface."""
        p = self.parameters
        if p.mode != "step":
            return self.sample(time_s)
        if time_s < p.ascent:
            return self.sample(time_s)

        sequence_time = time_s - p.ascent
        positions = self.step_positions
        sequence_duration = p.hold + (len(positions) - 1) * (p.transition + p.hold)
        if sequence_time < sequence_duration:
            if sequence_time < p.hold:
                return KinematicSetpoint(
                    positions[0], np.zeros(3), np.zeros(3), self.yaw, "step_center_0"
                )
            move_time = sequence_time - p.hold
            block = p.transition + p.hold
            index = min(
                int(move_time // block) + 1,
                len(positions) - 1,
            )
            local_time = move_time - (index - 1) * block
            if local_time < p.transition:
                position, velocity, acceleration = quintic_segment(
                    local_time,
                    p.transition,
                    positions[index - 1],
                    positions[index],
                )
                labels = (
                    "north", "center_1", "south", "center_2",
                    "east", "center_3", "west", "center_4",
                )
                return KinematicSetpoint(
                    position,
                    velocity,
                    acceleration,
                    self.yaw,
                    "step_" + labels[index - 1],
                )
            labels = (
                "north", "center_1", "south", "center_2",
                "east", "center_3", "west", "center_4",
            )
            return KinematicSetpoint(
                positions[index], np.zeros(3), np.zeros(3), self.yaw,
                "step_" + labels[index - 1],
            )

        descent_time = sequence_time - sequence_duration
        fraction, velocity_fraction, acceleration_fraction = smooth_profile(
            descent_time, p.descent
        )
        hover = self.start_position + np.array([0.0, 0.0, -p.altitude])
        delta = self.start_position - hover
        return KinematicSetpoint(
            hover + delta * fraction,
            delta * velocity_fraction,
            delta * acceleration_fraction,
            self.yaw,
            "descent",
        )

    def sample(self, time_s: float) -> KinematicSetpoint:
        p = self.parameters
        center = self.start_position
        hover = center + np.array([0.0, 0.0, -p.altitude])

        if time_s < p.ascent:
            fraction, velocity_fraction, acceleration_fraction = smooth_profile(
                time_s, p.ascent
            )
            delta = hover - center
            return KinematicSetpoint(
                center + delta * fraction,
                delta * velocity_fraction,
                delta * acceleration_fraction,
                self.yaw,
                "ascent",
            )

        if p.mode == "hover":
            hold_end = p.ascent + p.hold
            if time_s < hold_end:
                return KinematicSetpoint(
                    hover, np.zeros(3), np.zeros(3), self.yaw, "hold"
                )
            descent_time = time_s - hold_end
            fraction, velocity_fraction, acceleration_fraction = smooth_profile(
                descent_time, p.descent
            )
            delta = center - hover
            return KinematicSetpoint(
                hover + delta * fraction,
                delta * velocity_fraction,
                delta * acceleration_fraction,
                self.yaw,
                "descent",
            )

        if p.mode == "step":
            sequence_time = time_s - p.ascent
            sequence_duration = len(self.step_positions) * p.hold
            if sequence_time < sequence_duration:
                index = min(int(max(sequence_time, 0.0) // p.hold), len(self.step_positions) - 1)
                labels = ("center_0", "north", "center_1", "south", "center_2",
                          "east", "center_3", "west", "center_4")
                return KinematicSetpoint(
                    self.step_positions[index], np.zeros(3), np.zeros(3), self.yaw,
                    "step_" + labels[index],
                )
            descent_time = sequence_time - sequence_duration
            fraction, velocity_fraction, acceleration_fraction = smooth_profile(
                descent_time, p.descent
            )
            delta = center - hover
            return KinematicSetpoint(
                hover + delta * fraction,
                delta * velocity_fraction,
                delta * acceleration_fraction,
                self.yaw,
                "descent",
            )

        angular_rate = p.speed / p.radius
        is_figure8 = p.mode == "figure8"
        circle_start = hover if is_figure8 else hover + np.array([p.radius, 0.0, 0.0])
        tangent_velocity = (
            np.array([p.speed, p.speed, 0.0])
            if is_figure8
            else np.array([0.0, p.speed, 0.0])
        )
        centripetal_acceleration = (
            np.zeros(3)
            if is_figure8
            else np.array([-p.speed * angular_rate, 0.0, 0.0])
        )
        circle_start_time = p.ascent + p.transition
        circle_end_time = circle_start_time + self.circle_duration
        inbound_end = circle_end_time + p.transition

        if time_s < circle_start_time:
            position, velocity, acceleration = quintic_segment(
                time_s - p.ascent,
                p.transition,
                hover,
                circle_start,
                end_velocity=tangent_velocity,
                end_acceleration=centripetal_acceleration,
            )
            return KinematicSetpoint(position, velocity, acceleration, self.yaw, "outbound")
        if time_s < circle_end_time:
            angle = angular_rate * (time_s - circle_start_time)
            cosine = np.cos(angle)
            sine = np.sin(angle)
            if is_figure8:
                position = hover + np.array(
                    [p.radius * sine, p.radius * sine * cosine, 0.0]
                )
                velocity = np.array(
                    [p.speed * cosine, p.speed * np.cos(2.0 * angle), 0.0]
                )
                acceleration = np.array(
                    [
                        -p.speed * angular_rate * sine,
                        -2.0 * p.speed * angular_rate * np.sin(2.0 * angle),
                        0.0,
                    ]
                )
                return KinematicSetpoint(
                    position, velocity, acceleration, self.yaw, "figure8"
                )
            position = hover + np.array([p.radius * cosine, p.radius * sine, 0.0])
            velocity = np.array([-p.speed * sine, p.speed * cosine, 0.0])
            acceleration = np.array(
                [-p.speed * angular_rate * cosine, -p.speed * angular_rate * sine, 0.0]
            )
            return KinematicSetpoint(position, velocity, acceleration, self.yaw, "circle")
        if time_s < inbound_end:
            position, velocity, acceleration = quintic_segment(
                time_s - circle_end_time,
                p.transition,
                circle_start,
                hover,
                start_velocity=tangent_velocity,
                start_acceleration=centripetal_acceleration,
            )
            return KinematicSetpoint(position, velocity, acceleration, self.yaw, "inbound")

        descent_time = time_s - inbound_end
        fraction, velocity_fraction, acceleration_fraction = smooth_profile(
            descent_time, p.descent
        )
        delta = center - hover
        return KinematicSetpoint(
            hover + delta * fraction,
            delta * velocity_fraction,
            delta * acceleration_fraction,
            self.yaw,
            "descent",
        )


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply a continuous deadzone and rescale the remaining stick range."""
    if not np.isfinite(value):
        raise ValueError("stick value must be finite")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must lie in [0, 1)")
    value = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return float(np.sign(value) * (magnitude - deadzone) / (1.0 - deadzone))


def _approach_vector(current: np.ndarray, target: np.ndarray, maximum_delta: float) -> np.ndarray:
    delta = target - current
    norm = np.linalg.norm(delta)
    if norm <= maximum_delta or norm < 1.0e-12:
        return target.copy()
    return current + maximum_delta * delta / norm


def _approach_scalar(current: float, target: float, maximum_delta: float) -> float:
    return float(current + np.clip(target - current, -maximum_delta, maximum_delta))


class RcVelocityReference:
    """Integrate acceleration-limited RC velocity commands into a position reference."""

    def __init__(
        self,
        position: np.ndarray,
        yaw: float,
        config: ManualControlConfig,
        minimum_z: float,
        maximum_z: float,
    ) -> None:
        self.position = np.asarray(position, dtype=float).copy()
        if self.position.shape != (3,) or not np.all(np.isfinite(self.position)):
            raise ValueError("position must be a finite three-vector")
        if minimum_z > maximum_z:
            raise ValueError("minimum_z cannot exceed maximum_z")
        self.yaw = float(yaw)
        self.config = config
        self.minimum_z = float(minimum_z)
        self.maximum_z = float(maximum_z)
        self.velocity = np.zeros(3)
        self.yaw_rate = 0.0
        self._sticks = np.zeros(4)

    def set_sticks(self, roll: float, pitch: float, yaw: float, throttle: float) -> None:
        values = np.array([roll, pitch, yaw, throttle], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("manual-control sticks must be finite")
        self._sticks = np.array(
            [apply_deadzone(value, self.config.deadzone) for value in values]
        )

    @property
    def sticks_neutral(self) -> bool:
        return bool(np.allclose(self._sticks, 0.0))

    def step(self, time_step: float, measured_position: np.ndarray) -> None:
        if time_step <= 0.0:
            return
        measured_position = np.asarray(measured_position, dtype=float)
        if measured_position.shape != (3,) or not np.all(np.isfinite(measured_position)):
            raise ValueError("measured_position must be a finite three-vector")

        roll, pitch, yaw_stick, throttle = self._sticks
        heading_velocity = np.array([pitch, roll], dtype=float)
        heading_norm = np.linalg.norm(heading_velocity)
        if heading_norm > 1.0:
            heading_velocity /= heading_norm
        cosine = np.cos(self.yaw)
        sine = np.sin(self.yaw)
        target_horizontal = self.config.max_horizontal_speed * np.array(
            [
                cosine * heading_velocity[0] - sine * heading_velocity[1],
                sine * heading_velocity[0] + cosine * heading_velocity[1],
            ]
        )
        target_vertical = -self.config.max_vertical_speed * throttle
        target_yaw_rate = self.config.max_yaw_rate * yaw_stick

        self.velocity[:2] = _approach_vector(
            self.velocity[:2],
            target_horizontal,
            self.config.max_horizontal_acceleration * time_step,
        )
        self.velocity[2] = _approach_scalar(
            self.velocity[2],
            target_vertical,
            self.config.max_vertical_acceleration * time_step,
        )
        self.yaw_rate = _approach_scalar(
            self.yaw_rate,
            target_yaw_rate,
            self.config.max_yaw_acceleration * time_step,
        )

        self.position += self.velocity * time_step
        horizontal_lead = self.position[:2] - measured_position[:2]
        horizontal_lead_norm = np.linalg.norm(horizontal_lead)
        if horizontal_lead_norm > self.config.max_horizontal_position_lead:
            self.position[:2] = measured_position[:2] + (
                self.config.max_horizontal_position_lead
                * horizontal_lead
                / horizontal_lead_norm
            )
        self.position[2] = np.clip(
            self.position[2],
            measured_position[2] - self.config.max_vertical_position_lead,
            measured_position[2] + self.config.max_vertical_position_lead,
        )
        self.position[2] = np.clip(self.position[2], self.minimum_z, self.maximum_z)
        self.yaw = float(np.arctan2(np.sin(self.yaw + self.yaw_rate * time_step),
                                    np.cos(self.yaw + self.yaw_rate * time_step)))

    def sample(self, future_time: float) -> KinematicSetpoint:
        future_time = max(0.0, float(future_time))
        position = self.position + self.velocity * future_time
        position[2] = np.clip(position[2], self.minimum_z, self.maximum_z)
        yaw = self.yaw + self.yaw_rate * future_time
        return KinematicSetpoint(
            position,
            self.velocity.copy(),
            np.zeros(3),
            yaw,
            "rc",
        )


def build_kinematic_horizon(
    sample: Callable[[float], KinematicSetpoint],
    start_time: float,
    horizon_steps: int,
    sample_time: float,
) -> KinematicTrajectory:
    """Materialize the direct horizon shared by ROS1 and ROS2 adapters."""
    if horizon_steps < 1 or not np.isfinite(sample_time) or sample_time <= 0.0:
        raise ValueError("horizon_steps and sample_time must be positive")
    points = horizon_steps + 1
    samples = [sample(start_time + stage * sample_time) for stage in range(points)]
    acceleration = np.asarray([item.acceleration for item in samples], dtype=float)
    yaw = np.asarray([item.yaw for item in samples], dtype=float)
    return KinematicTrajectory(
        position=np.asarray([item.position for item in samples], dtype=float),
        velocity=np.asarray([item.velocity for item in samples], dtype=float),
        acceleration=acceleration,
        jerk=np.gradient(acceleration, sample_time, axis=0, edge_order=1),
        yaw=yaw,
        yaw_rate=np.gradient(np.unwrap(yaw), sample_time, edge_order=1),
        sample_time=sample_time,
    )


def build_reference_horizon(
    sample: Callable[[float], KinematicSetpoint],
    start_time: float,
    horizon_steps: int,
    sample_time: float,
    mass: float,
    gravity: float,
    thrust_min: float,
    thrust_max: float,
    body_rate_max: np.ndarray,
    quaternion_anchor: np.ndarray,
    disturbance: np.ndarray | None = None,
    timing: dict[str, float] | None = None,
) -> Reference:
    """Build state references and inverse-dynamics feed-forward controls."""
    disturbance = (
        np.zeros(3) if disturbance is None else np.asarray(disturbance, dtype=float)
    )
    if disturbance.shape != (3,) or not np.all(np.isfinite(disturbance)):
        raise ValueError("disturbance must be a finite three-vector")
    states = np.empty((horizon_steps + 1, 10))
    thrust = np.empty(horizon_steps + 1)
    previous_quaternion = np.asarray(quaternion_anchor, dtype=float)
    sample_time_accum = 0.0
    setpoints: list[KinematicSetpoint] = []
    sample_started_total = perf_counter()
    for stage in range(horizon_steps + 1):
        setpoints.append(sample(start_time + stage * sample_time))
    sample_finished_total = perf_counter()
    sample_time_accum = sample_finished_total - sample_started_total

    inverse_started_total = perf_counter()
    accelerations = np.asarray([item.acceleration for item in setpoints], dtype=float)
    yaws = np.asarray([item.yaw for item in setpoints], dtype=float)
    quaternions, thrust = inverse_dynamics_attitude_and_thrust_batch(
        accelerations,
        yaws,
        mass=mass,
        gravity=gravity,
        disturbance=disturbance,
    )
    inverse_finished_total = perf_counter()
    inverse_dynamics_time_accum = inverse_finished_total - inverse_started_total
    quaternion_alignment_time_accum = 0.0
    for stage, setpoint in enumerate(setpoints):
        alignment_started = perf_counter()
        quaternion = align_quaternion(quaternions[stage], previous_quaternion)
        quaternion_alignment_time_accum += perf_counter() - alignment_started
        states[stage] = np.r_[setpoint.position, setpoint.velocity, quaternion]
        previous_quaternion = quaternion

    feedforward_controls = np.empty((horizon_steps, 4))
    feedforward_controls[:, 0] = thrust[:-1]
    # Jerk is intentionally not an NMPC input on this branch.  The direct
    # producer has already time-parameterized the preview smoothly; NMPC
    # tracks the supplied position/velocity/acceleration samples.
    body_rate_started = perf_counter()
    feedforward_controls[:, 1:4] = average_body_rate_batch(
        states[:-1, 6:10], states[1:, 6:10], sample_time
    )
    body_rate_finished = perf_counter()
    limits_started = perf_counter()
    tolerance = 1.0e-9
    if np.any(feedforward_controls[:, 0] < thrust_min - tolerance) or np.any(
        feedforward_controls[:, 0] > thrust_max + tolerance
    ):
        raise ValueError("inverse-dynamics thrust feed-forward violates NMPC limits")
    if np.any(np.abs(feedforward_controls[:, 1:4]) > body_rate_max + tolerance):
        raise ValueError("inverse-dynamics body-rate feed-forward violates NMPC limits")
    limits_finished = perf_counter()
    attach_started = perf_counter()
    reference = attach_feedforward_rate_states(
        Reference(states=states, controls=feedforward_controls)
    )
    attach_finished = perf_counter()
    if timing is not None:
        timing.update(
            {
                "sample_eval_ms": 1.0e3 * sample_time_accum,
                "inverse_dynamics_ms": 1.0e3 * inverse_dynamics_time_accum,
                "quaternion_alignment_ms": 1.0e3 * quaternion_alignment_time_accum,
                "body_rate_feedforward_ms": 1.0e3 * (body_rate_finished - body_rate_started),
                "limits_check_ms": 1.0e3 * (limits_finished - limits_started),
                "attach_states_ms": 1.0e3 * (attach_finished - attach_started),
            }
        )
    return reference


def build_reference_from_trajectory(
    trajectory: KinematicTrajectory,
    horizon_steps: int,
    sample_time: float,
    mass: float,
    gravity: float,
    thrust_min: float,
    thrust_max: float,
    body_rate_max: np.ndarray,
    quaternion_anchor: np.ndarray,
    disturbance: np.ndarray | None = None,
    timing: dict[str, float] | None = None,
) -> Reference:
    """Convert a complete kinematic horizon into an NMPC reference.

    The upper computer supplies every point of the horizon. PX4 is not
    involved in reference generation; it receives only the final NMPC command.
    """
    validation_started = perf_counter()
    trajectory.validate(horizon_steps, sample_time)
    validation_finished = perf_counter()

    # A complete trajectory already contains the whole horizon.  Do not turn
    # it into 31 scalar ``KinematicSetpoint`` objects and then immediately
    # unpack them again: that path dominated the direct-interface timing.
    points = horizon_steps + 1
    position = np.asarray(trajectory.position, dtype=float)
    velocity = np.asarray(trajectory.velocity, dtype=float)
    acceleration = np.asarray(trajectory.acceleration, dtype=float)
    yaw = np.asarray(trajectory.yaw, dtype=float)
    inverse_started = perf_counter()
    quaternions, thrust = inverse_dynamics_attitude_and_thrust_batch(
        acceleration,
        yaw,
        mass=mass,
        gravity=gravity,
        disturbance=disturbance,
    )
    inverse_finished = perf_counter()
    alignment_started = perf_counter()
    quaternions = align_quaternion_sequence(quaternions, quaternion_anchor)
    alignment_finished = perf_counter()

    states = np.empty((points, 10), dtype=float)
    states[:, :3] = position
    states[:, 3:6] = velocity
    states[:, 6:10] = quaternions
    controls = np.empty((horizon_steps, 4), dtype=float)
    controls[:, 0] = thrust[:-1]
    body_rate_started = perf_counter()
    controls[:, 1:4] = average_body_rate_batch(
        quaternions[:-1], quaternions[1:], sample_time
    )
    body_rate_finished = perf_counter()
    limits_started = perf_counter()
    tolerance = 1.0e-9
    if np.any(controls[:, 0] < thrust_min - tolerance) or np.any(
        controls[:, 0] > thrust_max + tolerance
    ):
        raise ValueError("inverse-dynamics thrust feed-forward violates NMPC limits")
    if np.any(np.abs(controls[:, 1:4]) > body_rate_max + tolerance):
        raise ValueError("inverse-dynamics body-rate feed-forward violates NMPC limits")
    limits_finished = perf_counter()
    reference = attach_feedforward_rate_states(
        Reference(states=states, controls=controls)
    )
    if timing is not None:
        timing.update(
            {
                "sample_eval_ms": 0.0,
                "inverse_dynamics_ms": 1.0e3 * (inverse_finished - inverse_started),
                "quaternion_alignment_ms": 1.0e3 * (alignment_finished - alignment_started),
                "body_rate_feedforward_ms": 1.0e3 * (body_rate_finished - body_rate_started),
                "limits_check_ms": 1.0e3 * (limits_finished - limits_started),
                "attach_states_ms": 0.0,
            }
        )
    if timing is not None:
        timing["trajectory_validation_ms"] = 1.0e3 * (
            validation_finished - validation_started
        )
    return reference
