#!/usr/bin/env python3
"""Run guarded PX4 SITL trajectory regression tests with the NMPC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from px4_msgs.msg import (
    ManualControlSetpoint,
    NmpcReferenceSetpoint,
    NmpcTrajectory,
    NmpcTrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nmpc.config import NmpcConfig, load_config
from nmpc.model.quadrotor import (
    QuadrotorModel,
    quaternion_attitude_error,
    quaternion_to_rotation,
)
from nmpc.px4 import thrust_newton_to_px4
from nmpc.setpoint import (
    KinematicTrajectory,
    KinematicSetpoint,
    PresetTrajectory,
    PresetTrajectoryParameters,
    RcVelocityReference,
    apply_deadzone,
    build_reference_from_trajectory,
)
from nmpc.solver.acados_solver import AcadosNmpc
from nmpc.types import Control, Reference
from nmpc.validation import ModelValidationRecorder


@dataclass
class TestOptions:
    trajectory: str = "hover"
    reference_source: str = "px4-smoothed"
    altitude: float = 1.0
    radius: float = 0.5
    speed: float = 0.25
    rc_duration: float = 60.0
    rc_geofence_radius: float = 5.0
    validate_model: bool = False
    log_directory: Path | None = None
    prestream: float = 1.5
    ascent: float = 4.0
    hold: float = 6.0
    transition: float = 3.0
    descent: float = 4.0
    settle: float = 1.5
    takeoff_extra: float = 3.5
    takeoff_ramp: float = 0.8
    takeoff_height: float = 0.15
    takeoff_brake_extra: float = 1.5
    takeoff_settle_speed: float = 0.05
    takeoff_brake_timeout: float = 1.5
    takeoff_converge_distance: float = 0.05
    takeoff_converge_speed: float = 0.15
    takeoff_converge_accel: float = 1.0
    # The convergence gap at subphase entry is ~0.3-0.45 m (odom vs PX4 local
    # frame offset at the takeoff hold) and closes at ~0.04 m/s; a first
    # flight after a fresh PX4 boot starts near the top of that range and
    # needs ~8-10 s to close it.
    takeoff_converge_timeout: float = 12.0
    takeoff_hold_kp: float = 8.0
    takeoff_hold_kd: float = 4.0
    takeoff_hold_max_delta: float = 5.0
    takeoff_timeout: float = 20.0

    @property
    def circle_duration(self) -> float:
        return 2.0 * np.pi * self.radius / self.speed


def _yaw_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class Px4NmpcHover(Node):
    def __init__(self, config: NmpcConfig, options: TestOptions) -> None:
        super().__init__("px4_sitl_nmpc_trajectory")
        self.config = config
        self.options = options
        if config.controller.horizon_steps + 1 > NmpcTrajectory.MAX_POINTS:
            raise ValueError(
                "configured NMPC horizon exceeds the NmpcTrajectory protocol capacity"
            )
        self.controller = AcadosNmpc(config)

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odometry, output_qos
        )
        self.create_subscription(
            ManualControlSetpoint,
            "/fmu/out/manual_control_setpoint",
            self._on_manual_control,
            output_qos,
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, output_qos
        )
        self.create_subscription(
            VehicleCommandAck, "/fmu/out/vehicle_command_ack", self._on_ack, output_qos
        )
        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._on_land_detected,
            output_qos,
        )
        self.create_subscription(
            NmpcTrajectory,
            "/fmu/out/nmpc_trajectory",
            self._on_nmpc_trajectory,
            output_qos,
        )
        self.create_subscription(
            NmpcTrajectorySetpoint,
            "/nmpc/in/trajectory_setpoint",
            self._on_direct_trajectory,
            input_qos,
        )
        self.mode_publisher = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", input_qos
        )
        self.rates_publisher = self.create_publisher(
            VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", input_qos
        )
        self.command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", input_qos
        )
        self.reference_publisher = self.create_publisher(
            NmpcReferenceSetpoint, "/fmu/in/nmpc_reference_setpoint", input_qos
        )
        self.direct_trajectory_publisher = self.create_publisher(
            NmpcTrajectorySetpoint, "/nmpc/in/trajectory_setpoint", input_qos
        )

        self.phase = "WAIT_ODOMETRY"
        self.phase_started = monotonic()
        self.takeoff_subphase = ""
        self.takeoff_subphase_started = 0.0
        self.takeoff_hold_z = 0.0
        self.takeoff_last_converge_log = 0.0
        self.last_odometry_time = 0.0
        self.last_command_time = 0.0
        self.last_log_time = 0.0
        self.estimator_stable_since = 0.0
        self.last_reset_counter: int | None = None
        self.odometry: VehicleOdometry | None = None
        self.status: VehicleStatus | None = None
        self.land_detected: VehicleLandDetected | None = None
        self.initial_position: np.ndarray | None = None
        self.reference_quaternion: np.ndarray | None = None
        self.reference_yaw = 0.0
        self.preset_source: PresetTrajectory | None = None
        self.rc_source: RcVelocityReference | None = None
        self.manual_control: ManualControlSetpoint | None = None
        self.nmpc_trajectory: NmpcTrajectory | None = None
        self.last_nmpc_trajectory_time = 0.0
        self.direct_trajectory: NmpcTrajectorySetpoint | None = None
        self.last_direct_trajectory_time = 0.0
        self.direct_trajectory_sequence = 0
        self.last_manual_control_time = 0.0
        self.flight_started = 0.0
        self.flight_started_px4_us = 0
        self.last_solved_odometry_us = 0
        self.last_trajectory_odometry_us = 0
        self.trajectory_elapsed = 0.0
        self.timestamp_jump_count = 0
        self.finished = False
        self.success = False
        self.finish_reason = "not finished"
        self.safety_abort = False
        self.command_acks: list[dict[str, int]] = []
        self.solve_times: list[float] = []
        self.tracking_errors: list[float] = []
        self.saturation_count = 0
        self.solve_count = 0
        self.max_body_rate_command = np.zeros(3)
        self.thrust_commands: list[float] = []
        self.trajectory_records: list[dict[str, object]] = []
        self.model_validation = (
            ModelValidationRecorder(
                QuadrotorModel(config.model.mass, config.model.gravity, config.model.rate_tau),
                maximum_interval=0.1,
            )
            if options.validate_model
            else None
        )
        # World-frame acceleration residual estimator: the OCP model exposes a
        # constant translational disturbance parameter that absorbs the thrust
        # curve offset, rotor drag, and other slowly varying model error.
        self.disturbance_estimate = np.zeros(3)
        # Slow on purpose: the residual is built from odometry velocity
        # differences, which lag the true motion by ~0.1-0.2 s in SITL.  A
        # fast estimate chases those lag transients and feeds phantom
        # disturbances back into the OCP, which then tilts to counter them.
        self.disturbance_tau = 1.0
        self.disturbance_clamp = 1.0
        self.last_measurement_state: np.ndarray | None = None
        self.last_measurement_dt = 0.0
        self.last_command_thrust = 0.0
        # Keep outbound PX4 timestamps in the synchronised epoch domain, but
        # advance them from CLOCK_MONOTONIC.  WSL can step CLOCK_REALTIME when
        # it resynchronises with the Windows host; publishing that step would
        # make PX4 treat otherwise healthy Offboard heartbeats as stale.
        self.timestamp_epoch_us = self.get_clock().now().nanoseconds // 1000
        self.timestamp_monotonic_origin = monotonic()
        # WSL periodically corrects CLOCK_REALTIME against the Windows host.  A
        # ROS system-clock timer can consequently pause or execute a burst of
        # callbacks, leaving PX4 holding a stale rate command in the meantime.
        # Drive the controller from CLOCK_MONOTONIC instead; message timestamps
        # remain in the PX4-synchronised system-time domain.
        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            config.controller.sample_time,
            self._control_tick,
            clock=self.control_clock,
        )

    def _now_us(self) -> int:
        elapsed_us = round((monotonic() - self.timestamp_monotonic_origin) * 1.0e6)
        return self.timestamp_epoch_us + elapsed_us

    def _on_odometry(self, message: VehicleOdometry) -> None:
        self.odometry = message
        self.last_odometry_time = monotonic()
        # Tie the Offboard heartbeat to the high-rate PX4 stream as well as the
        # wall timer so simulator clock catch-up cannot create a heartbeat gap.
        if self.phase != "DISARMING" and not self.finished:
            self._publish_mode()

    def _on_status(self, message: VehicleStatus) -> None:
        self.status = message

    def _on_manual_control(self, message: ManualControlSetpoint) -> None:
        self.manual_control = message
        self.last_manual_control_time = monotonic()

    def _on_ack(self, message: VehicleCommandAck) -> None:
        self.command_acks.append({"command": int(message.command), "result": int(message.result)})

    def _on_land_detected(self, message: VehicleLandDetected) -> None:
        self.land_detected = message

    def _on_nmpc_trajectory(self, message: NmpcTrajectory) -> None:
        self.nmpc_trajectory = message
        self.last_nmpc_trajectory_time = monotonic()

    def _on_direct_trajectory(self, message: NmpcTrajectorySetpoint) -> None:
        self.direct_trajectory = message
        self.last_direct_trajectory_time = monotonic()

    def _publish_mode(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._now_us()
        message.body_rate = True
        self.mode_publisher.publish(message)

    def _publish_rates(self, thrust_newton: float, body_rate: np.ndarray) -> None:
        message = VehicleRatesSetpoint()
        message.timestamp = self._now_us()
        message.roll = float(body_rate[0])
        message.pitch = float(body_rate[1])
        message.yaw = float(body_rate[2])
        message.thrust_body = [0.0, 0.0, thrust_newton_to_px4(thrust_newton, self.config)]
        self.rates_publisher.publish(message)

    def _publish_position_reference(self, setpoint: KinematicSetpoint) -> None:
        message = NmpcReferenceSetpoint()
        message.timestamp = self._now_us()
        message.mode = NmpcReferenceSetpoint.MODE_POSITION
        message.prediction_steps = self.config.controller.horizon_steps
        message.sample_time = self.config.controller.sample_time
        message.position = np.asarray(setpoint.position, dtype=float).tolist()
        message.velocity = [float("nan")] * 3
        message.acceleration = [float("nan")] * 3
        message.yaw = float(setpoint.yaw)
        message.yaw_rate = float("nan")
        self.reference_publisher.publish(message)

    def _publish_initial_position_reference(self) -> None:
        assert self.initial_position is not None
        self._publish_position_reference(
            KinematicSetpoint(
                position=self.initial_position,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=self.reference_yaw,
                segment="initial_hold",
            )
        )

    def _publish_initial_direct_trajectory(self) -> None:
        assert self.initial_position is not None
        points = self.config.controller.horizon_steps + 1
        self._publish_direct_trajectory(
            KinematicTrajectory(
                position=np.repeat(self.initial_position[None, :], points, axis=0),
                velocity=np.zeros((points, 3)),
                acceleration=np.zeros((points, 3)),
                jerk=np.zeros((points, 3)),
                yaw=np.full(points, self.reference_yaw),
                sample_time=self.config.controller.sample_time,
            )
        )

    def _publish_initial_reference(self) -> None:
        if self.options.reference_source == "px4-smoothed":
            self._publish_initial_position_reference()
        else:
            self._publish_initial_direct_trajectory()

    def _kinematic_horizon(self, elapsed: float) -> KinematicTrajectory:
        sample = self._trajectory_sample
        if self.options.trajectory == "step":
            current = sample(elapsed)
            sample = lambda _time: current
        points = self.config.controller.horizon_steps + 1
        sample_time = self.config.controller.sample_time
        samples = [sample(elapsed + stage * sample_time) for stage in range(points)]
        acceleration = np.asarray([item.acceleration for item in samples], dtype=float)
        jerk = np.gradient(acceleration, sample_time, axis=0, edge_order=1)
        yaw = np.asarray([item.yaw for item in samples], dtype=float)
        return KinematicTrajectory(
            position=np.asarray([item.position for item in samples], dtype=float),
            velocity=np.asarray([item.velocity for item in samples], dtype=float),
            acceleration=acceleration,
            jerk=jerk,
            yaw=yaw,
            yaw_rate=np.gradient(np.unwrap(yaw), sample_time, edge_order=1),
            sample_time=sample_time,
        )

    def _publish_direct_trajectory(self, trajectory: KinematicTrajectory) -> None:
        trajectory.validate(
            self.config.controller.horizon_steps,
            self.config.controller.sample_time,
        )
        message = NmpcTrajectorySetpoint()
        message.timestamp = self._now_us()
        self.direct_trajectory_sequence += 1
        message.sequence = self.direct_trajectory_sequence
        message.points = self.config.controller.horizon_steps + 1
        message.sample_time = trajectory.sample_time
        capacity = NmpcTrajectorySetpoint.MAX_POINTS

        def padded(values: np.ndarray, columns: int) -> list[float]:
            output = np.zeros((capacity, columns), dtype=np.float32)
            output[: message.points] = np.asarray(values, dtype=np.float32)
            return output.reshape(-1).tolist()

        message.position = padded(trajectory.position, 3)
        message.velocity = padded(trajectory.velocity, 3)
        message.acceleration = padded(trajectory.acceleration, 3)
        assert trajectory.jerk is not None
        message.jerk = padded(trajectory.jerk, 3)
        message.yaw = padded(np.asarray(trajectory.yaw)[:, None], 1)
        self.direct_trajectory_publisher.publish(message)

    def _publish_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self._now_us()
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.command = int(command)
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_publisher.publish(message)

    def _request_offboard(self) -> None:
        self._publish_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def _request_arm(self) -> None:
        self._publish_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def _request_disarm(self) -> None:
        self._publish_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)

    def _state(self) -> np.ndarray:
        assert self.odometry is not None
        position = np.asarray(self.odometry.position, dtype=float)
        velocity = np.asarray(self.odometry.velocity, dtype=float)
        quaternion = np.asarray(self.odometry.q, dtype=float)
        body_rate = np.asarray(self.odometry.angular_velocity, dtype=float)
        return np.r_[position, velocity, quaternion, body_rate]

    def _manual_control_ready(self, require_neutral: bool = False) -> bool:
        message = self.manual_control
        if (
            message is None
            or not message.valid
            or monotonic() - self.last_manual_control_time > self.config.manual_control.timeout
        ):
            return False
        sticks = np.array(
            [message.roll, message.pitch, message.yaw, message.throttle], dtype=float
        )
        if not np.all(np.isfinite(sticks)):
            return False
        if require_neutral:
            return all(
                apply_deadzone(value, self.config.manual_control.deadzone) == 0.0
                for value in sticks
            )
        return True

    def _initialize_reference_source(self, anchor_position: np.ndarray) -> None:
        anchor = np.asarray(anchor_position, dtype=float)
        parameters = PresetTrajectoryParameters(
            mode=self.options.trajectory if self.options.trajectory != "rc" else "hover",
            altitude=self.options.altitude,
            ascent=self.options.ascent,
            hold=self.options.hold if self.options.trajectory != "rc" else 1.0e6,
            transition=self.options.transition,
            descent=self.options.descent,
            settle=self.options.settle,
            radius=self.options.radius,
            speed=self.options.speed,
        )
        self.preset_source = PresetTrajectory(
            anchor, self.reference_yaw, parameters
        )
        if self.options.trajectory == "rc":
            hover_position = anchor + np.array(
                [0.0, 0.0, -self.options.altitude]
            )
            self.rc_source = RcVelocityReference(
                hover_position,
                self.reference_yaw,
                self.config.manual_control,
                minimum_z=float(anchor[2] - 1.8),
                maximum_z=float(anchor[2] - 0.3),
            )

    def _trajectory_sample(self, time_s: float) -> KinematicSetpoint:
        assert self.preset_source is not None
        if self.options.trajectory != "rc" or time_s < self.options.ascent:
            return self.preset_source.sample(time_s)
        assert self.rc_source is not None
        return self.rc_source.sample(time_s - self.trajectory_elapsed)

    def _total_duration(self) -> float:
        if self.options.trajectory == "rc":
            return self.options.ascent + self.options.rc_duration
        assert self.preset_source is not None
        return self.preset_source.duration

    def _estimate_disturbance(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Low-pass the measured-vs-modelled acceleration residual (world frame).

        The OCP model exposes a constant translational disturbance parameter on
        every stage; a slow estimate absorbs the thrust-curve offset, rotor
        drag, and other slowly varying model error so the MPC does not have to
        chase them with the terminal cost alone.
        """
        if self.last_measurement_state is None:
            self.last_measurement_state = state.copy()
            self.last_measurement_dt = dt
            return self.disturbance_estimate.copy()
        step = dt if dt > 0.0 and np.isfinite(dt) else self.last_measurement_dt
        step = max(step, 1.0e-3)
        measured_acceleration = (state[3:6] - self.last_measurement_state[3:6]) / step
        self.last_measurement_state = state.copy()
        self.last_measurement_dt = step
        if self.initial_position is not None and state[2] >= self.initial_position[2] - 0.2:
            # Near or on the ground the contact reaction dominates the residual;
            # hold the last in-flight estimate instead of winding it up.
            return self.disturbance_estimate.copy()
        rotation = quaternion_to_rotation(state[6:10])
        modelled_acceleration = np.array([0.0, 0.0, self.config.model.gravity])
        modelled_acceleration -= self.last_command_thrust / self.config.model.mass * rotation[:, 2]
        residual = measured_acceleration - modelled_acceleration
        alpha = min(1.0, step / self.disturbance_tau)
        self.disturbance_estimate += alpha * (residual - self.disturbance_estimate)
        np.clip(
            self.disturbance_estimate,
            -self.disturbance_clamp,
            self.disturbance_clamp,
            out=self.disturbance_estimate,
        )
        return self.disturbance_estimate.copy()

    def _reference_from_trajectory(self, trajectory: KinematicTrajectory) -> Reference:
        assert self.reference_quaternion is not None
        limits = self.config.limits
        trajectory.validate_motion_limits(
            horizontal_speed_max=limits.horizontal_speed_max,
            vertical_speed_max_up=limits.vertical_speed_max_up,
            vertical_speed_max_down=limits.vertical_speed_max_down,
            horizontal_acceleration_max=limits.horizontal_acceleration_max,
            vertical_acceleration_max_up=limits.vertical_acceleration_max_up,
            vertical_acceleration_max_down=limits.vertical_acceleration_max_down,
            jerk_max=limits.jerk_max,
        )
        return build_reference_from_trajectory(
            trajectory,
            horizon_steps=self.config.controller.horizon_steps,
            sample_time=self.config.controller.sample_time,
            mass=self.config.model.mass,
            gravity=self.config.model.gravity,
            thrust_min=limits.thrust_min,
            thrust_max=limits.thrust_max,
            body_rate_max=limits.body_rate_max,
            quaternion_anchor=self.reference_quaternion,
        )

    def _direct_reference(self) -> Reference:
        message = self.direct_trajectory
        if message is None:
            raise RuntimeError("complete NMPC trajectory has not been received")
        timeout = self.config.controller.reference_timeout
        if monotonic() - self.last_direct_trajectory_time > timeout:
            raise RuntimeError("complete NMPC trajectory is stale")
        timestamp_age = (self._now_us() - int(message.timestamp)) * 1.0e-6
        if abs(timestamp_age) > timeout:
            raise RuntimeError("complete NMPC trajectory timestamp is outside the allowed age")
        points = int(message.points)
        expected_points = self.config.controller.horizon_steps + 1
        if points != expected_points:
            raise RuntimeError(
                f"complete trajectory has {points} points, expected {expected_points}"
            )
        trajectory = KinematicTrajectory(
            position=np.asarray(message.position, dtype=float).reshape(-1, 3)[:points],
            velocity=np.asarray(message.velocity, dtype=float).reshape(-1, 3)[:points],
            acceleration=np.asarray(message.acceleration, dtype=float).reshape(-1, 3)[:points],
            jerk=np.asarray(message.jerk, dtype=float).reshape(-1, 3)[:points],
            yaw=np.asarray(message.yaw, dtype=float)[:points],
            sample_time=float(message.sample_time),
        )
        return self._reference_from_trajectory(trajectory)

    def _trajectory_point0(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the first point (position, velocity, acceleration) of a fresh PX4 trajectory."""
        message = self.nmpc_trajectory
        if message is None or monotonic() - self.last_nmpc_trajectory_time > 0.3:
            return None
        points = int(message.points)
        if points < 1:
            return None
        position = np.asarray(message.position, dtype=float).reshape(-1, 3)[0]
        velocity = np.asarray(message.velocity, dtype=float).reshape(-1, 3)[0]
        acceleration = np.asarray(message.acceleration, dtype=float).reshape(-1, 3)[0]
        return position, velocity, acceleration

    def _px4_smoothed_reference(self) -> Reference:
        assert self.reference_quaternion is not None
        message = self.nmpc_trajectory
        if message is None:
            raise RuntimeError("PX4 NMPC trajectory has not been received")
        maximum_age = self.config.controller.reference_timeout
        if monotonic() - self.last_nmpc_trajectory_time > maximum_age:
            raise RuntimeError("PX4 NMPC trajectory is stale")
        if message.input_timed_out:
            raise RuntimeError("PX4 reports that the NMPC reference input timed out")
        points = int(message.points)
        expected_points = self.config.controller.horizon_steps + 1
        if points != expected_points:
            raise RuntimeError(
                f"PX4 trajectory has {points} points, expected {expected_points}"
            )
        trajectory = KinematicTrajectory(
            position=np.asarray(message.position, dtype=float).reshape(-1, 3)[:points],
            velocity=np.asarray(message.velocity, dtype=float).reshape(-1, 3)[:points],
            acceleration=np.asarray(message.acceleration, dtype=float).reshape(-1, 3)[:points],
            jerk=np.asarray(message.jerk, dtype=float).reshape(-1, 3)[:points],
            yaw=np.asarray(message.yaw, dtype=float)[:points],
            yaw_rate=np.asarray(message.yaw_rate, dtype=float)[:points],
            sample_time=float(message.sample_time),
        )
        return self._reference_from_trajectory(trajectory)

    def _reference(self, elapsed: float) -> Reference:
        if self.options.reference_source == "direct":
            return self._direct_reference()
        return self._px4_smoothed_reference()

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = monotonic()
        self.get_logger().info(f"phase={phase}")

    def _start_disarming(self, reason: str, safety_abort: bool = False) -> None:
        if self.phase == "DISARMING":
            return
        self.finish_reason = reason
        self.safety_abort = safety_abort
        self._set_phase("DISARMING")
        self.last_command_time = 0.0

    def _collect_tracking(self, target: KinematicSetpoint, elapsed: float) -> bool:
        collect = target.segment in ("circle", "figure8", "rc") or target.segment.startswith(
            "step_"
        )
        if self.options.trajectory == "hover":
            hold_start = self.options.ascent + 1.0
            hold_end = self.options.ascent + self.options.hold
            collect = hold_start <= elapsed <= hold_end
        return collect

    def _record_solve(
        self,
        odometry_us: int,
        elapsed: float,
        target: KinematicSetpoint,
        state: np.ndarray,
        reference: Reference,
        command: Control,
        saturated: bool,
        disturbance: np.ndarray | None = None,
    ) -> None:
        reference_state = reference.states[0]
        feedforward = reference.feedforward_controls[0]
        actual_rate = np.asarray(self.odometry.angular_velocity, dtype=float)
        position_error = state[:3] - reference_state[:3]
        velocity_error = state[3:6] - reference_state[3:6]
        attitude_error = quaternion_attitude_error(state[6:10], reference_state[6:10])
        rate_error = actual_rate - command.body_rate
        message_age = (
            monotonic() - self.last_nmpc_trajectory_time
            if self.options.reference_source == "px4-smoothed"
            else monotonic() - self.last_direct_trajectory_time
        )
        record: dict[str, object] = {
            "timestamp_us": odometry_us,
            "trajectory_time_s": elapsed,
            "segment": target.segment,
            "reference_source": self.options.reference_source,
            "metric_enabled": int(self._collect_tracking(target, elapsed)),
            "solve_time_ms": 1000.0 * self.controller.last_solve_time,
            "reference_age_ms": 1000.0 * message_age,
            "saturated": int(saturated),
            "thrust_command_N": command.thrust,
            "thrust_feedforward_N": feedforward[0],
            "position_error_norm_m": np.linalg.norm(position_error),
            "velocity_error_norm_m_s": np.linalg.norm(velocity_error),
            "attitude_error_norm_rad": np.linalg.norm(attitude_error),
            "body_rate_error_norm_rad_s": np.linalg.norm(rate_error),
        }
        for prefix, values in (
            ("position", state[:3]),
            ("position_reference", reference_state[:3]),
            ("velocity", state[3:6]),
            ("velocity_reference", reference_state[3:6]),
            ("quaternion", state[6:10]),
            ("quaternion_reference", reference_state[6:10]),
            ("body_rate", actual_rate),
            ("body_rate_command", command.body_rate),
            ("body_rate_feedforward", feedforward[1:4]),
            ("position_error", position_error),
            ("velocity_error", velocity_error),
            ("attitude_error", attitude_error),
            ("body_rate_error", rate_error),
        ):
            labels = ("w", "x", "y", "z") if "quaternion" in prefix else ("x", "y", "z")
            for label, value in zip(labels, values):
                record[f"{prefix}_{label}"] = float(value)
        if disturbance is not None:
            for label, value in zip(("x", "y", "z"), disturbance):
                record[f"disturbance_{label}"] = float(value)
        self.trajectory_records.append(record)

    @staticmethod
    def _rmse(records: list[dict[str, object]], key: str) -> float:
        values = np.asarray([row[key] for row in records], dtype=float)
        return float(np.sqrt(np.mean(values * values))) if values.size else float("inf")

    def _write_run_artifacts(self, summary: dict[str, object]) -> None:
        directory = self.options.log_directory
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        trajectory_path = directory / "trajectory.csv"
        if self.trajectory_records:
            with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.trajectory_records[0]))
                writer.writeheader()
                writer.writerows(self.trajectory_records)
        else:
            trajectory_path.write_text("", encoding="utf-8")
        summary["trajectory_log"] = str(trajectory_path.resolve())
        summary_path = directory / "summary.json"
        summary["summary_log"] = str(summary_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def _finish(self) -> None:
        tracking = np.asarray(self.tracking_errors, dtype=float)
        solves = 1000.0 * np.asarray(self.solve_times, dtype=float)
        tracking_rmse = (
            float(np.sqrt(np.mean(tracking**2))) if tracking.size else float("inf")
        )
        tracking_max = float(np.max(tracking)) if tracking.size else float("inf")
        p99_ms = float(np.percentile(solves, 99.0)) if solves.size else float("inf")
        max_ms = float(np.max(solves)) if solves.size else float("inf")
        landed = self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
        saturation_fraction = self.saturation_count / max(self.solve_count, 1)
        metric_records = [
            row for row in self.trajectory_records if bool(row["metric_enabled"])
        ]
        self.success = bool(
            landed
            and not self.safety_abort
            and self.solve_count > 100
            and tracking.size > 100
            and tracking_rmse < 0.25
            and tracking_max < 0.60
            and saturation_fraction < 0.10
            and p99_ms < 10.0
        )
        summary = {
            "success": self.success,
            "reason": self.finish_reason,
            "trajectory": self.options.trajectory,
            "radius_or_step_distance_m": (
                self.options.radius if self.options.trajectory in ("step", "circle", "figure8") else None
            ),
            "trajectory_speed_m_s": (
                self.options.speed if self.options.trajectory in ("circle", "figure8") else None
            ),
            "landed_disarmed": landed,
            "solve_count": self.solve_count,
            "solve_p99_ms": p99_ms,
            "solve_max_ms": max_ms,
            "odometry_timestamp_jump_count": self.timestamp_jump_count,
            "tracking_position_rmse_m": tracking_rmse,
            "tracking_position_max_m": tracking_max,
            "input_saturation_fraction": saturation_fraction,
            "reference_source": self.options.reference_source,
            "metric_sample_count": len(metric_records),
            "velocity_rmse_m_s": self._rmse(metric_records, "velocity_error_norm_m_s"),
            "attitude_rmse_rad": self._rmse(metric_records, "attitude_error_norm_rad"),
            "body_rate_tracking_rmse_rad_s": self._rmse(
                metric_records, "body_rate_error_norm_rad_s"
            ),
            "thrust_range_N": [
                float(np.min(self.thrust_commands)) if self.thrust_commands else float("nan"),
                float(np.max(self.thrust_commands)) if self.thrust_commands else float("nan"),
            ],
            "max_abs_body_rate_command_rad_s": self.max_body_rate_command.tolist(),
            "command_acks": self.command_acks,
        }
        if self.model_validation is not None and len(self.model_validation.samples) >= 2:
            summary["model_validation"] = self.model_validation.summary()
        self._write_run_artifacts(summary)
        print("NMPC_SITL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
        self.finished = True

    def _control_tick(self) -> None:
        now = monotonic()
        if self.finished:
            return
        if self.odometry is None or self.status is None:
            if now - self.phase_started > 8.0:
                self.finish_reason = "PX4 odometry/status timeout"
                self.safety_abort = True
                self._finish()
            return
        if now - self.last_odometry_time > 1.0:
            self._start_disarming("odometry became stale", safety_abort=True)

        if self.phase != "DISARMING":
            self._publish_mode()
        if self.phase == "WAIT_ODOMETRY":
            state = self._state()
            if not np.all(np.isfinite(state)):
                return
            if self.odometry.pose_frame != VehicleOdometry.POSE_FRAME_NED:
                self.finish_reason = f"unexpected pose frame {self.odometry.pose_frame}"
                self.safety_abort = True
                self._finish()
                return
            if self.odometry.velocity_frame != VehicleOdometry.VELOCITY_FRAME_NED:
                self.finish_reason = f"unexpected velocity frame {self.odometry.velocity_frame}"
                self.safety_abort = True
                self._finish()
                return
            reset_counter = int(self.odometry.reset_counter)
            if self.last_reset_counter != reset_counter:
                self.last_reset_counter = reset_counter
                self.estimator_stable_since = now
                return
            if not self.status.pre_flight_checks_pass or now - self.estimator_stable_since < 1.0:
                return
            self.initial_position = state[:3].copy()
            self.reference_quaternion = _yaw_quaternion(state[6:10])
            self.reference_yaw = _quaternion_yaw(self.reference_quaternion)
            self._set_phase("PRESTREAM")

        if self.phase == "PRESTREAM":
            self._publish_initial_reference()
            self._publish_rates(self.config.hover_thrust, np.zeros(3))
            if now - self.phase_started >= self.options.prestream:
                self._set_phase("ENTER_OFFBOARD")
                self.last_command_time = 0.0
            return

        if self.phase == "ENTER_OFFBOARD":
            self._publish_initial_reference()
            self._publish_rates(self.config.hover_thrust, np.zeros(3))
            if now - self.last_command_time >= 0.5:
                self._request_offboard()
                self.last_command_time = now
            if self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self._set_phase("ARMING")
                self.last_command_time = 0.0
            elif now - self.phase_started > 6.0:
                self._start_disarming("offboard timeout", safety_abort=True)
            return

        if self.phase == "ARMING":
            self._publish_initial_reference()
            self._publish_rates(self.config.hover_thrust, np.zeros(3))
            if self.options.trajectory == "rc" and not self._manual_control_ready(
                require_neutral=True
            ):
                if now - self.phase_started > 15.0:
                    self._start_disarming(
                        "RC input missing, invalid, stale, or not centered",
                        safety_abort=True,
                    )
                return
            if now - self.last_command_time >= 0.5:
                self._request_arm()
                self.last_command_time = now
            if self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                state = self._state()
                self.initial_position = state[:3].copy()
                self.reference_quaternion = _yaw_quaternion(state[6:10])
                self.reference_yaw = _quaternion_yaw(self.reference_quaternion)
                self.takeoff_subphase = "accel"
                self.takeoff_subphase_started = now
                self._set_phase("TAKEOFF")
            elif now - self.phase_started > 6.0:
                self._start_disarming("arming timeout", safety_abort=True)
            return

        if self.phase == "TAKEOFF":
            # Lift off with a thrust profile before handing authority to the
            # NMPC.  The controller model has no ground model, so solving from
            # ground contact feeds it a state its prediction cannot explain.
            # The takeoff ends settled near hover: the flight reference is
            # anchored at the takeoff-end state, so the first OCP solve sees
            # ~zero tracking error instead of a climb it must violently brake.
            state = self._state()
            if not np.all(np.isfinite(state)):
                return
            # Keep the PX4 reference pinned to the vehicle so the smoothed
            # trajectory follows the takeoff instead of lagging behind it.
            self._publish_position_reference(
                KinematicSetpoint(
                    position=state[:3],
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    yaw=self.reference_yaw,
                    segment="takeoff_hold",
                )
            )
            if self.options.reference_source == "direct":
                # The direct-trajectory path rejects messages older than
                # reference_timeout (0.2 s).  Keep publishing throughout the
                # takeoff so flight entry always finds a fresh horizon.  The
                # preset trajectory does not exist yet (it is built when the
                # flight reference is anchored), so publish the same stationary
                # hold as _publish_initial_direct_trajectory.
                points = self.config.controller.horizon_steps + 1
                self._publish_direct_trajectory(
                    KinematicTrajectory(
                        position=np.repeat(state[:3][None, :], points, axis=0),
                        velocity=np.zeros((points, 3)),
                        acceleration=np.zeros((points, 3)),
                        jerk=np.zeros((points, 3)),
                        yaw=np.full(points, self.reference_yaw),
                        sample_time=self.config.controller.sample_time,
                    )
                )
            if self.takeoff_subphase == "accel":
                ramp = min(1.0, (now - self.phase_started) / self.options.takeoff_ramp)
                thrust = self.config.hover_thrust + self.options.takeoff_extra * ramp
                if state[2] < self.initial_position[2] - self.options.takeoff_height:
                    self.takeoff_subphase = "brake"
                    self.takeoff_subphase_started = now
            elif self.takeoff_subphase == "brake":
                # Brake the climb back to a near-hover.
                thrust = self.config.hover_thrust - self.options.takeoff_brake_extra
                if (
                    abs(state[5]) < self.options.takeoff_settle_speed
                    or now - self.takeoff_subphase_started > self.options.takeoff_brake_timeout
                ):
                    self.takeoff_subphase = "converge"
                    self.takeoff_subphase_started = now
                    self.takeoff_hold_z = state[2]
            else:
                # Hold the vehicle with a gentle vertical P-D loop.  An
                # open-loop hover thrust leaves a residual acceleration that
                # slowly drags the vehicle away, and the smoothed trajectory's
                # approach speed vanishes near the target (sqrt(2 a d)), so it
                # can never close the last decimeters against a drifting
                # vehicle.  A truly held vehicle gives it a static target to
                # settle onto.
                thrust = self.config.hover_thrust + (
                    self.options.takeoff_hold_kp * (state[2] - self.takeoff_hold_z)
                    + self.options.takeoff_hold_kd * state[5]
                )
                thrust = float(
                    np.clip(
                        thrust,
                        self.config.hover_thrust - self.options.takeoff_hold_max_delta,
                        self.config.hover_thrust + self.options.takeoff_hold_max_delta,
                    )
                )
            self._publish_rates(thrust, np.zeros(3))
            self.last_command_thrust = thrust
            body_z_down = 1.0 - 2.0 * (state[7] ** 2 + state[8] ** 2)
            tilt = float(np.arccos(np.clip(body_z_down, -1.0, 1.0)))
            if (
                tilt > np.deg2rad(25.0)
                or float(np.linalg.norm(state[:2] - self.initial_position[:2])) > 0.5
                or now - self.phase_started > self.options.takeoff_timeout
            ):
                self._start_disarming("takeoff failed", safety_abort=True)
                return
            if self.takeoff_subphase == "converge":
                trajectory_point0 = self._trajectory_point0()
                if now - self.takeoff_last_converge_log > 0.5:
                    self.takeoff_last_converge_log = now
                    if trajectory_point0 is not None:
                        dpos = float(np.linalg.norm(trajectory_point0[0] - state[:3]))
                        dvel = float(np.linalg.norm(trajectory_point0[1] - state[3:6]))
                        dacc = float(np.linalg.norm(trajectory_point0[2]))
                        self.get_logger().info(
                            f"converge dpos={dpos:.3f} dvel={dvel:.3f} dacc={dacc:.3f} "
                            f"veh vz={state[5]:+.3f} z={state[2]:+.3f} "
                            f"p0 z={trajectory_point0[0][2]:+.3f} vz={trajectory_point0[1][2]:+.3f}"
                        )
                    else:
                        self.get_logger().info("converge trajectory unavailable")
                # The trajectory must match the vehicle in position, velocity
                # AND acceleration before the flight starts: the OCP trusts
                # point0's acceleration for its thrust feedforward, so entering
                # while the smoother is still braking (or chasing) makes the
                # first solve command a wildly wrong thrust.
                converged = trajectory_point0 is not None and (
                    float(np.linalg.norm(trajectory_point0[0] - state[:3]))
                    < self.options.takeoff_converge_distance
                    and float(np.linalg.norm(trajectory_point0[1] - state[3:6]))
                    < self.options.takeoff_converge_speed
                    and float(np.linalg.norm(trajectory_point0[2]))
                    < self.options.takeoff_converge_accel
                )
                if converged:
                    anchor = state[:3].copy()
                    self.flight_started = now
                    self.flight_started_px4_us = int(self.odometry.timestamp)
                    self.last_solved_odometry_us = 0
                    self.last_trajectory_odometry_us = int(self.odometry.timestamp)
                    self.trajectory_elapsed = 0.0
                    self._initialize_reference_source(anchor)
                    self._set_phase("FLIGHT")
                elif (
                    now - self.takeoff_subphase_started
                    > self.options.takeoff_converge_timeout
                ):
                    self._start_disarming(
                        "takeoff failed: trajectory did not converge", safety_abort=True
                    )
                    return
            return

        if self.phase == "FLIGHT":
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self._start_disarming("offboard mode lost during flight", safety_abort=True)
                return
            if self.options.trajectory == "rc" and not self._manual_control_ready():
                self._start_disarming("manual-control input lost or invalid", safety_abort=True)
                return
            odometry_us = int(self.odometry.timestamp)
            if odometry_us == self.last_solved_odometry_us:
                return
            self.last_solved_odometry_us = odometry_us
            raw_step = (odometry_us - self.last_trajectory_odometry_us) * 1.0e-6
            nominal_step = self.config.controller.sample_time
            maximum_step = max(0.1, 10.0 * nominal_step)
            if raw_step < 0.0 or raw_step > maximum_step:
                self.timestamp_jump_count += 1
                self.get_logger().warn(
                    "odometry timestamp step %.3fs replaced by nominal %.3fs"
                    % (raw_step, nominal_step)
                )
                trajectory_step = nominal_step
            else:
                trajectory_step = raw_step
            self.trajectory_elapsed += trajectory_step
            self.last_trajectory_odometry_us = odometry_us
            elapsed = self.trajectory_elapsed
            state = self._state()
            if self.options.trajectory == "rc" and elapsed >= self.options.ascent:
                assert self.rc_source is not None
                assert self.manual_control is not None
                self.rc_source.set_sticks(
                    self.manual_control.roll,
                    self.manual_control.pitch,
                    self.manual_control.yaw,
                    self.manual_control.throttle,
                )
                self.rc_source.step(trajectory_step, state[:3])
            target = self._trajectory_sample(elapsed)
            if self.options.reference_source == "px4-smoothed":
                self._publish_position_reference(target)
            else:
                self._publish_direct_trajectory(self._kinematic_horizon(elapsed))
            assert self.initial_position is not None
            horizontal_error = float(np.linalg.norm(state[:2] - self.initial_position[:2]))
            horizontal_limit = (
                self.options.rc_geofence_radius
                if self.options.trajectory == "rc"
                else max(1.0, self.options.radius + 0.75)
            )
            body_z_down = 1.0 - 2.0 * (state[7] ** 2 + state[8] ** 2)
            tilt = float(np.arccos(np.clip(body_z_down, -1.0, 1.0)))
            if (
                horizontal_error > horizontal_limit
                or state[2] < self.initial_position[2] - 2.0
                or state[2] > self.initial_position[2] + 0.5
                or tilt > np.deg2rad(35.0)
            ):
                self._start_disarming("position safety bound exceeded", safety_abort=True)
                return
            try:
                reference = self._reference(elapsed)
                disturbance = self._estimate_disturbance(state, trajectory_step)
                command = self.controller.solve(state, reference, disturbance)
            except Exception as error:
                self._start_disarming(f"solver failure: {error}", safety_abort=True)
                return
            self.solve_count += 1
            self.solve_times.append(self.controller.last_solve_time)
            self.thrust_commands.append(command.thrust)
            self.max_body_rate_command = np.maximum(
                self.max_body_rate_command, np.abs(command.body_rate)
            )
            saturated = bool(
                command.thrust <= self.config.limits.thrust_min + 1.0e-4
                or command.thrust >= self.config.limits.thrust_max - 1.0e-4
                or np.any(np.abs(command.body_rate) >= self.config.limits.body_rate_max - 1.0e-4)
            )
            if saturated:
                self.saturation_count += 1
            self._record_solve(
                odometry_us, elapsed, target, state, reference, command, saturated, disturbance
            )
            self._publish_rates(command.thrust, command.body_rate)
            self.last_command_thrust = command.thrust
            tracked_position = reference.states[0, :3]
            if (
                self.model_validation is not None
                and self.initial_position is not None
                and state[2] < self.initial_position[2] - 0.2
            ):
                self.model_validation.add(
                    odometry_us,
                    state,
                    np.asarray(self.odometry.angular_velocity, dtype=float),
                    command.as_array(),
                    target.segment,
                )
            if now - self.last_log_time >= 0.5:
                self.last_log_time = now
                self.get_logger().info(
                    "flight t=%.2f segment=%s p=[%.2f %.2f %.2f] "
                    "pref=[%.2f %.2f %.2f] v=[%.2f %.2f %.2f] "
                    "tilt=%.1fdeg T=%.2f rate=[%.2f %.2f %.2f]"
                    % (
                        elapsed,
                        target.segment,
                        *state[:3],
                        *target.position,
                        *state[3:6],
                        np.degrees(tilt),
                        command.thrust,
                        *command.body_rate,
                    )
                )
                self.get_logger().info(
                    "attitude q=[%.3f %.3f %.3f %.3f] qref=[%.3f %.3f %.3f %.3f]"
                    % (*state[6:10], *self.reference_quaternion)
                )

            if self._collect_tracking(target, elapsed):
                self.tracking_errors.append(float(np.linalg.norm(state[:3] - tracked_position)))

            if elapsed >= self._total_duration():
                self._start_disarming("trajectory completed")
            return

        if self.phase == "DISARMING":
            if now - self.last_command_time >= 0.5:
                if self.land_detected is not None and self.land_detected.landed:
                    self._request_disarm()
                else:
                    self._publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_command_time = now
            if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self._finish()
            elif now - self.phase_started > 12.0:
                self.finish_reason += "; disarm confirmation timeout"
                self.safety_abort = True
                self._finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nmpc.yaml"))
    parser.add_argument(
        "--trajectory", choices=("hover", "step", "circle", "figure8", "rc"), default="hover"
    )
    parser.add_argument(
        "--reference-source",
        choices=("px4-smoothed", "direct"),
        default="px4-smoothed",
        help="PX4-smoothed point commands or a complete directly supplied trajectory",
    )
    parser.add_argument(
        "--step-dwell", type=float, default=2.0,
        help="seconds held at each center/cardinal position during the step test",
    )
    parser.add_argument("--altitude", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--rc-duration", type=float, default=60.0)
    parser.add_argument("--rc-geofence-radius", type=float, default=5.0)
    parser.add_argument("--validate-model", action="store_true")
    parser.add_argument(
        "--log-directory",
        type=Path,
        help="directory for per-solve trajectory.csv and summary.json",
    )
    arguments = parser.parse_args()
    positive_arguments = (
        arguments.altitude,
        arguments.radius,
        arguments.speed,
        arguments.rc_duration,
        arguments.rc_geofence_radius,
        arguments.step_dwell,
    )
    if any(value <= 0.0 for value in positive_arguments):
        parser.error("altitude, radius, speed, RC duration and geofence must be positive")
    config = load_config(arguments.config)
    options = TestOptions(
        trajectory=arguments.trajectory,
        reference_source=arguments.reference_source,
        altitude=arguments.altitude,
        radius=arguments.radius,
        speed=arguments.speed,
        rc_duration=arguments.rc_duration,
        rc_geofence_radius=arguments.rc_geofence_radius,
        validate_model=arguments.validate_model,
        log_directory=arguments.log_directory,
        hold=arguments.step_dwell if arguments.trajectory == "step" else TestOptions.hold,
    )
    rclpy.init()
    node = Px4NmpcHover(config, options)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._start_disarming("interrupted", safety_abort=True)
        deadline = monotonic() + 15.0
        while rclpy.ok() and not node.finished and monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        success = node.success
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
