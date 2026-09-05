#!/usr/bin/env python3
"""Guarded PX4 SITL hover smoke for the single-process C++ NMPC node."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from time import monotonic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rclpy
from px4_msgs.msg import (
    NmpcTrajectorySetpoint,
    VehicleCommand,
    VehicleLandDetected,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from integration.mavlink_params import DEFAULT_PARAMETERS, GuardedParameter, ParamGuard
from nmpc.trajectory import quintic_segment, smooth_profile


class CppHoverSupervisor(Node):
    def __init__(self, output_directory: Path, trajectory: str = "hover",
                 radius: float = 2.0, speed: float = 1.0,
                 point_hold_duration: float = 2.0,
                 safety_drift_limit: float | None = None,
                 expect_rc_timeout_fallback: bool = False,
                 reference_sample_time: float = 0.01) -> None:
        super().__init__("cpp_nmpc_hover_supervisor")
        self.output_directory = output_directory
        if trajectory not in ("hover", "point_1m", "circle", "figure8"):
            raise ValueError("trajectory must be hover, point_1m, circle or figure8")
        self.trajectory = trajectory
        self.radius = float(radius)
        self.speed = float(speed)
        self.point_hold_duration = float(point_hold_duration)
        self.safety_drift_limit = safety_drift_limit
        self.expect_rc_timeout_fallback = bool(expect_rc_timeout_fallback)
        self.points = 31
        self.sample_time = float(reference_sample_time)
        self.ascent_duration = 4.0
        self.transition_duration = 3.0
        self.point_hold_duration = 4.0
        self.hold_duration = (2.0 * self.transition_duration + 2.0 * np.pi * self.radius / self.speed
                              if self.trajectory in ("circle", "figure8") else
                              self.point_hold_duration + 8.0 *
                              (self.transition_duration + self.point_hold_duration)
                              if self.trajectory == "point_1m" else 6.0)
        self.altitude = 1.0
        self.phase = "WAIT"
        self.phase_started = monotonic()
        self.flight_started = 0.0
        self.last_command_time = 0.0
        self.odometry: VehicleOdometry | None = None
        self.status: VehicleStatus | None = None
        self.land_detected: VehicleLandDetected | None = None
        self.initial_position: np.ndarray | None = None
        self.initial_yaw = 0.0
        self.finished = False
        self.failure_reason = ""
        self.rc_timeout_seen = False
        self.rc_timeout_position_seen = False
        self.rc_timeout_fallback_seen = False
        self.rc_timeout_hold_started = 0.0
        self.odometry_timestamp_fault_seen = False
        self.odometry_timestamp_fault_fallback_seen = False
        self.odometry_timestamp_fault_hold_started = 0.0
        self.odometry_timestamp_gap_count = 0
        self.supervisor_observer_gap_count = 0
        self.odometry_update_count = 0
        self.odometry_receive_time_s = 0.0
        self.last_flight_odometry_update_count = 0
        self.last_flight_odometry_receive_time_s = 0.0
        self.last_flight_odometry_timestamp_us = 0
        self.control_enable_sent = False
        self.records: list[dict[str, float]] = []
        self._last_reference_velocity = np.zeros(3, dtype=float)
        self.trajectory_sequence = 0
        self.trajectory_publish_count = 0
        self.trajectory_publish_gap_count = 0
        self.trajectory_publish_max_gap_s = 0.0
        self._last_trajectory_publish_time_s = 0.0
        self._trajectory_stop = threading.Event()
        self._management_lock = threading.RLock()
        self._trajectory_state_lock = threading.Lock()
        # The complete reference horizon is published by a dedicated daemon
        # thread below.  Keep state/fault callbacks and the supervisor timer
        # in their own callback groups so safety checks do not depend on the
        # RMW publish path.
        self._state_callback_group = MutuallyExclusiveCallbackGroup()
        self._timer_callback_group = MutuallyExclusiveCallbackGroup()

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
        fault_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        trajectory_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odometry, output_qos,
            callback_group=self._state_callback_group
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, output_qos,
            callback_group=self._state_callback_group
        )
        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._on_land_detected,
            output_qos,
            callback_group=self._state_callback_group,
        )
        self.trajectory_publisher = self.create_publisher(
            NmpcTrajectorySetpoint, "/nmpc/in/trajectory_setpoint", trajectory_qos
        )
        self.enable_publisher = self.create_publisher(
            Bool, "/nmpc/control_enabled", input_qos
        )
        self.create_subscription(
            Bool, "/nmpc/rc_timeout", self._on_rc_timeout, input_qos,
            callback_group=self._state_callback_group
        )
        self.create_subscription(
            Bool, "/nmpc/odometry_timestamp_fault",
            self._on_odometry_timestamp_fault, fault_qos,
            callback_group=self._state_callback_group
        )
        self.command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", input_qos
        )
        self.timer = self.create_timer(
            self.sample_time, self._tick, callback_group=self._timer_callback_group
        )
        # A complete fixed-size horizon may enter the RMW layer.  Keep it out
        # of the state/safety executor so a DDS publication stall cannot stop
        # the supervisor from observing PX4 or closing the case.
        self._trajectory_thread = threading.Thread(
            target=self._trajectory_publish_loop,
            name="nmpc-trajectory-publisher",
            daemon=True,
        )
        self._trajectory_thread.start()

    def _timestamp_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def _on_odometry(self, message: VehicleOdometry) -> None:
        with self._management_lock:
            self.odometry = message
            self.odometry_update_count += 1
            self.odometry_receive_time_s = monotonic()

    def _on_status(self, message: VehicleStatus) -> None:
        with self._management_lock:
            self.status = message
            if self.phase == "RC_TIMEOUT_HOLD" and message.nav_state in (
                VehicleStatus.NAVIGATION_STATE_POSCTL,
                VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
            ):
                self.rc_timeout_fallback_seen = True
                if message.nav_state == VehicleStatus.NAVIGATION_STATE_POSCTL:
                    self.rc_timeout_position_seen = True

    def _on_land_detected(self, message: VehicleLandDetected) -> None:
        with self._management_lock:
            self.land_detected = message

    def _on_rc_timeout(self, message: Bool) -> None:
        with self._management_lock:
            if not message.data or self.phase in ("LANDING", "DONE") or self.rc_timeout_seen:
                return
            self.rc_timeout_seen = True
            self.get_logger().error(
                "RC timeout received; disabling NMPC and requesting PX4 AUTO.LOITER"
            )
            self._enable(False)
            self.last_command_time = 0.0
            if self.phase == "FLIGHT":
                self.rc_timeout_hold_started = monotonic()
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
                self._set_phase("RC_TIMEOUT_HOLD")
            else:
                self.failure_reason = self.failure_reason or "RC timeout before FLIGHT"
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._set_phase("LANDING")

    def _handle_odometry_timestamp_fault(self, reason: str) -> None:
        with self._management_lock:
            if self.odometry_timestamp_fault_seen:
                return
            self.odometry_timestamp_fault_seen = True
            self.failure_reason = reason
            self._enable(False)
            self.last_command_time = 0.0
            if self.phase == "FLIGHT":
                self.odometry_timestamp_fault_hold_started = monotonic()
                self.get_logger().error(
                    "%s; disabling NMPC and requesting PX4 Position/Hold" % reason
                )
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
                self._set_phase("ODOMETRY_FAULT_HOLD")
            elif self.phase not in ("LANDING", "DONE"):
                self.get_logger().error(
                    "%s before FLIGHT; disabling NMPC and aborting" % reason
                )
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._set_phase("LANDING")

    def _on_odometry_timestamp_fault(self, message: Bool) -> None:
        with self._management_lock:
            if not message.data:
                return
            self.odometry_timestamp_gap_count += 1
            self._handle_odometry_timestamp_fault("C++ odometry timestamp gap fault")

    def _enable(self, enabled: bool) -> None:
        message = Bool()
        message.data = enabled
        self.enable_publisher.publish(message)

    def _command(self, command: int, param1: float = 0.0, param2: float = 0.0,
                 param3: float = 0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp_us()
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.param3 = float(param3)
        message.command = int(command)
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_publisher.publish(message)

    def _sample(
        self, time_s: float, initial_position: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchor = self.initial_position if initial_position is None else initial_position
        assert anchor is not None
        if time_s <= 0.0:
            return anchor.copy(), np.zeros(3), np.zeros(3)
        if time_s < self.ascent_duration:
            fraction, velocity_fraction, acceleration_fraction = smooth_profile(
                time_s, self.ascent_duration
            )
            delta = np.array([0.0, 0.0, -self.altitude])
            return (
                anchor + fraction * delta,
                velocity_fraction * delta,
                acceleration_fraction * delta,
            )
        hover = anchor + np.array([0.0, 0.0, -self.altitude])
        if self.trajectory == "hover":
            return hover, np.zeros(3), np.zeros(3)
        if self.trajectory == "point_1m":
            # Keep the historical four-direction point test: center -> +X ->
            # center -> -X -> center -> +Y -> center -> -Y -> center.
            # Every leg is a quintic, acceleration-limited transition followed
            # by the same dwell used by the Python baseline.
            positions = (
                hover,
                hover + np.array([self.radius, 0.0, 0.0]),
                hover,
                hover + np.array([-self.radius, 0.0, 0.0]),
                hover,
                hover + np.array([0.0, self.radius, 0.0]),
                hover,
                hover + np.array([0.0, -self.radius, 0.0]),
                hover,
            )
            sequence_time = time_s - self.ascent_duration
            sequence_duration = self.point_hold_duration + 8.0 * (
                self.transition_duration + self.point_hold_duration
            )
            if sequence_time < self.point_hold_duration:
                return positions[0], np.zeros(3), np.zeros(3)
            move_time = sequence_time - self.point_hold_duration
            block = self.transition_duration + self.point_hold_duration
            index = min(int(move_time // block) + 1, len(positions) - 1)
            local_time = move_time - (index - 1) * block
            if local_time < self.transition_duration:
                return quintic_segment(
                    local_time,
                    self.transition_duration,
                    positions[index - 1],
                    positions[index],
                )
            if sequence_time < sequence_duration:
                return positions[index], np.zeros(3), np.zeros(3)
            return positions[-1], np.zeros(3), np.zeros(3)
        omega = self.speed / self.radius
        circle_start = hover if self.trajectory == "figure8" else hover + np.array([self.radius, 0.0, 0.0])
        tangent_velocity = (
            np.array([self.speed, self.speed, 0.0])
            if self.trajectory == "figure8" else np.array([0.0, self.speed, 0.0])
        )
        centripetal_acceleration = (
            np.zeros(3)
            if self.trajectory == "figure8" else np.array([-self.speed * omega, 0.0, 0.0])
        )
        circle_start_time = self.ascent_duration + self.transition_duration
        circle_duration = 2.0 * np.pi * self.radius / self.speed
        circle_end_time = circle_start_time + circle_duration
        inbound_end_time = circle_end_time + self.transition_duration
        if time_s < circle_start_time:
            return quintic_segment(
                time_s - self.ascent_duration,
                self.transition_duration,
                hover,
                circle_start,
                end_velocity=tangent_velocity,
                end_acceleration=centripetal_acceleration,
            )
        if time_s < circle_end_time:
            theta = omega * (time_s - circle_start_time)
            sine = np.sin(theta)
            cosine = np.cos(theta)
            if self.trajectory == "figure8":
                position = hover + np.array([
                    self.radius * sine, self.radius * sine * cosine, 0.0
                ])
                velocity = np.array([
                    self.speed * cosine, self.speed * np.cos(2.0 * theta), 0.0
                ])
                acceleration = np.array([
                    -self.speed * omega * sine,
                    -2.0 * self.speed * omega * np.sin(2.0 * theta),
                    0.0,
                ])
            else:
                position = hover + np.array([self.radius * cosine, self.radius * sine, 0.0])
                velocity = np.array([-self.speed * sine, self.speed * cosine, 0.0])
                acceleration = np.array([
                    -self.speed * omega * cosine,
                    -self.speed * omega * sine,
                    0.0,
                ])
            return position, velocity, acceleration
        if time_s < inbound_end_time:
            return quintic_segment(
                time_s - circle_end_time,
                self.transition_duration,
                circle_start,
                hover,
                start_velocity=tangent_velocity,
                start_acceleration=centripetal_acceleration,
            )
        return hover, np.zeros(3), np.zeros(3)

    def _publish_horizon(
        self, elapsed: float, initial_position: np.ndarray | None = None,
        initial_yaw: float | None = None,
    ) -> np.ndarray:
        samples = [
            self._sample(elapsed + i * self.sample_time, initial_position)
            for i in range(self.points)
        ]
        position = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        velocity = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        acceleration = np.asarray([sample[2] for sample in samples], dtype=np.float32)
        jerk = np.empty_like(acceleration)
        jerk[:-1] = np.diff(acceleration, axis=0) / self.sample_time
        jerk[-1] = jerk[-2] if self.points > 1 else 0.0
        message = NmpcTrajectorySetpoint()
        message.timestamp = self._timestamp_us()
        with self._trajectory_state_lock:
            self.trajectory_sequence += 1
            message.sequence = self.trajectory_sequence
            now = monotonic()
            if self._last_trajectory_publish_time_s > 0.0:
                gap = now - self._last_trajectory_publish_time_s
                self.trajectory_publish_max_gap_s = max(self.trajectory_publish_max_gap_s, gap)
                if gap > max(0.1, 2.0 * self.sample_time):
                    self.trajectory_publish_gap_count += 1
            self._last_trajectory_publish_time_s = now
            self.trajectory_publish_count += 1
        message.yaw[: self.points] = [
            self.initial_yaw if initial_yaw is None else initial_yaw
        ] * self.points
        message.points = self.points
        message.sample_time = self.sample_time
        message.position[: 3 * self.points] = position.reshape(-1).tolist()
        message.velocity[: 3 * self.points] = velocity.reshape(-1).tolist()
        message.acceleration[: 3 * self.points] = acceleration.reshape(-1).tolist()
        message.jerk[: 3 * self.points] = jerk.reshape(-1).tolist()
        self.trajectory_publisher.publish(message)
        with self._trajectory_state_lock:
            self._last_reference_velocity = velocity[0].astype(float)
        return position[0].astype(float)

    def _trajectory_publish_loop(self) -> None:
        next_publish = monotonic()
        while not self._trajectory_stop.is_set() and rclpy.ok():
            now = monotonic()
            with self._management_lock:
                phase = self.phase
                initial_position = (
                    None if self.initial_position is None else self.initial_position.copy()
                )
                initial_yaw = self.initial_yaw
                flight_started = self.flight_started
            if phase in ("PRESTREAM", "ENTER_OFFBOARD", "ARMING", "FLIGHT") and initial_position is not None:
                elapsed = now - flight_started if phase == "FLIGHT" else 0.0
                try:
                    self._publish_horizon(elapsed, initial_position, initial_yaw)
                except Exception as error:  # pragma: no cover - defensive during shutdown
                    self.get_logger().error("trajectory publisher failed: %s" % error)
            next_publish += self.sample_time
            now = monotonic()
            if next_publish < now - self.sample_time:
                next_publish = now + self.sample_time
            self._trajectory_stop.wait(max(0.0, next_publish - now))

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = monotonic()
        self.get_logger().info(f"phase={phase}")

    def _abort(self, reason: str) -> None:
        if not self.failure_reason:
            self.failure_reason = reason
            self.get_logger().error(reason)
        self._enable(False)
        self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self._set_phase("LANDING")

    def _tick(self) -> None:
        with self._management_lock:
            self._tick_locked()

    def _tick_locked(self) -> None:
        if self.finished:
            return
        now = monotonic()
        if self.phase == "WAIT":
            self._enable(False)
            if self.odometry is None or self.status is None:
                if now - self.phase_started > 30.0:
                    self.failure_reason = "PX4 DDS odometry/status timeout"
                    self._finish()
                return
            if not self.status.pre_flight_checks_pass:
                return
            state = np.r_[self.odometry.position, self.odometry.velocity, self.odometry.q]
            if not np.all(np.isfinite(state)):
                return
            self.initial_position = np.asarray(self.odometry.position, dtype=float)
            w, x, y, z = np.asarray(self.odometry.q, dtype=float)
            self.initial_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            self._set_phase("PRESTREAM")
            return

        assert self.initial_position is not None
        if self.phase == "PRESTREAM":
            self._enable(True)
            if now - self.phase_started > 1.5:
                self._set_phase("ENTER_OFFBOARD")
            return

        if self.phase == "ENTER_OFFBOARD":
            self._enable(True)
            if now - self.last_command_time > 0.5:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.last_command_time = now
            if self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self._set_phase("ARMING")
                self.last_command_time = 0.0
            elif now - self.phase_started > 8.0:
                self._abort("Offboard entry timeout")
            return

        if self.phase == "ARMING":
            self._enable(True)
            if now - self.last_command_time > 0.5:
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_command_time = now
            if self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.initial_position = np.asarray(self.odometry.position, dtype=float)
                self.flight_started = now
                # The cached odometry may be a pre-arm sample.  Establish the
                # continuity baseline from the first fresh FLIGHT sample
                # instead of measuring the entire pre-arm interval as a gap.
                self.last_flight_odometry_update_count = self.odometry_update_count
                self.last_flight_odometry_receive_time_s = 0.0
                self.last_flight_odometry_timestamp_us = 0
                self.control_enable_sent = False
                self._set_phase("FLIGHT")
            elif now - self.phase_started > 8.0:
                self._abort("arming timeout")
            return

        if self.phase == "FLIGHT":
            # Reference publication is a periodic source in its own right.
            # Do not make it depend on the observer's latest odometry callback:
            # a transient PX4/DDS observer gap must not turn a still-valid
            # trajectory into a stale-reference timeout in the C++ node.
            elapsed = now - self.flight_started
            reference = self._sample(elapsed)[0]
            if self.odometry_update_count == self.last_flight_odometry_update_count:
                if (self.last_flight_odometry_receive_time_s > 0.0 and
                        now - self.last_flight_odometry_receive_time_s > 0.10):
                    # The C++ node is the controller-side safety authority and
                    # publishes its own receive-gap fault.  This Python
                    # supervisor subscription is validation/metrics only; it
                    # can temporarily stop dispatching while the C++ node
                    # continues receiving the same PX4 stream.  Do not turn an
                    # observer scheduling gap into a flight abort.
                    self.supervisor_observer_gap_count += 1
                    self.get_logger().warning(
                        "supervisor odometry observer gap %.3fs; waiting for the "
                        "C++ controller fault topic instead"
                        % (now - self.last_flight_odometry_receive_time_s)
                    )
                    self.last_flight_odometry_receive_time_s = 0.0
                return
            receive_step_s = (
                self.odometry_receive_time_s - self.last_flight_odometry_receive_time_s
                if self.last_flight_odometry_receive_time_s > 0.0 else float("nan")
            )
            self.last_flight_odometry_update_count = self.odometry_update_count
            if (self.last_flight_odometry_receive_time_s > 0.0 and
                    (not np.isfinite(receive_step_s) or receive_step_s <= 0.0 or
                     receive_step_s > 0.10)):
                self.supervisor_observer_gap_count += 1
                self.get_logger().warning(
                    "supervisor odometry observer gap %.3fs; accepting the next "
                    "sample and waiting for the C++ controller fault topic"
                    % receive_step_s
                )
                self.last_flight_odometry_receive_time_s = 0.0
            self.last_flight_odometry_receive_time_s = self.odometry_receive_time_s
            timestamp_us = int(getattr(self.odometry, "timestamp", 0))
            previous_timestamp_us = self.last_flight_odometry_timestamp_us
            if timestamp_us <= 0:
                self.get_logger().warning(
                    "dropping odometry with zero PX4 timestamp; receive clock remains authoritative"
                )
                return
            if previous_timestamp_us > 0 and timestamp_us <= previous_timestamp_us:
                self.get_logger().warning(
                    "dropping out-of-order odometry timestamp %d after %d"
                    % (timestamp_us, previous_timestamp_us)
                )
                return
            self.last_flight_odometry_timestamp_us = timestamp_us
            if not self.control_enable_sent:
                self._enable(True)
                self.control_enable_sent = True
            position = np.asarray(self.odometry.position, dtype=float)
            velocity = np.asarray(self.odometry.velocity, dtype=float)
            with self._trajectory_state_lock:
                reference_velocity = self._last_reference_velocity.copy()
            error = position - reference
            self.records.append(
                {
                    "time_s": elapsed,
                    "position_x": position[0], "position_y": position[1], "position_z": position[2],
                    "reference_x": reference[0], "reference_y": reference[1], "reference_z": reference[2],
                    "velocity_x": velocity[0], "velocity_y": velocity[1], "velocity_z": velocity[2],
                    "reference_velocity_x": reference_velocity[0],
                    "reference_velocity_y": reference_velocity[1],
                    "reference_velocity_z": reference_velocity[2],
                    "position_error_m": float(np.linalg.norm(error)),
                }
            )
            horizontal_drift = float(np.linalg.norm(position[:2] - self.initial_position[:2]))
            drift_limit = self.safety_drift_limit
            if drift_limit is None:
                drift_limit = 5.0 if self.trajectory in ("circle", "figure8") else 2.0
            if horizontal_drift > drift_limit or position[2] < self.initial_position[2] - 1.6:
                self._abort(f"flight safety bound exceeded: drift={horizontal_drift:.2f} z={position[2]:.2f}")
                return
            if elapsed >= self.ascent_duration + self.hold_duration:
                self._enable(False)
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._set_phase("LANDING")
            return

        if self.phase == "RC_TIMEOUT_HOLD":
            self._enable(False)
            if self.status is not None and self.status.nav_state in (
                VehicleStatus.NAVIGATION_STATE_POSCTL,
                VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
            ):
                if not self.rc_timeout_position_seen:
                    self.rc_timeout_position_seen = True
                    self.get_logger().info("RC timeout fallback confirmed: PX4 Position/AUTO.LOITER")
                self.rc_timeout_fallback_seen = True
            elif self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.rc_timeout_fallback_seen = True
                self.get_logger().info("RC timeout fallback confirmed: PX4 Land failsafe")
            elif now - self.last_command_time > 0.5:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
                self.last_command_time = now
            # The test must verify the fallback first, then land explicitly so
            # the SITL case can close cleanly.  Real flight does not use this
            # scripted landing step.
            if now - self.rc_timeout_hold_started >= 3.0:
                if self.expect_rc_timeout_fallback and not self.rc_timeout_fallback_seen:
                    self._abort("RC timeout did not reach PX4 Position or Land fallback")
                else:
                    self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self._set_phase("LANDING")
            return

        if self.phase == "ODOMETRY_FAULT_HOLD":
            self._enable(False)
            if self.status is not None and self.status.nav_state in (
                VehicleStatus.NAVIGATION_STATE_POSCTL,
                VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
            ):
                self.odometry_timestamp_fault_fallback_seen = True
            elif self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.odometry_timestamp_fault_fallback_seen = True
            elif now - self.last_command_time > 0.5:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
                self.last_command_time = now
            # The test confirms the fallback before asking SITL to land.  A
            # real flight manager owns this mode change and recovery policy.
            if now - self.odometry_timestamp_fault_hold_started >= 3.0:
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._set_phase("LANDING")
            return

        if self.phase == "LANDING":
            self._enable(False)
            if now - self.last_command_time > 0.5:
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_command_time = now
            if self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self._finish()
            elif now - self.phase_started > 20.0:
                self.failure_reason = self.failure_reason or "landing/disarm timeout"
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                self._finish()

    def _finish(self) -> None:
        metric = [row for row in self.records if row["time_s"] >= self.ascent_duration + 1.0]
        errors = np.asarray([row["position_error_m"] for row in metric], dtype=float)
        rmse = float(np.sqrt(np.mean(errors * errors))) if errors.size else float("inf")
        maximum = float(np.max(errors)) if errors.size else float("inf")
        landed = self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
        success = bool(landed and not self.failure_reason and errors.size > 100 and rmse < 0.25 and maximum < 0.6)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._trajectory_stop.set()
        csv_path = self.output_directory / "trajectory.csv"
        if self.records:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
        with self._trajectory_state_lock:
            trajectory_publish_count = self.trajectory_publish_count
            trajectory_publish_gap_count = self.trajectory_publish_gap_count
            trajectory_publish_max_gap_s = self.trajectory_publish_max_gap_s
        summary = {
            "success": success,
            "backend": "cpp_single_process",
            "reason": self.failure_reason or "trajectory completed",
            "landed_disarmed": landed,
            "sample_count": len(self.records),
            "metric_sample_count": int(errors.size),
            "tracking_position_rmse_m": rmse,
            "tracking_position_max_m": maximum,
            "trajectory_log": str(csv_path.resolve()),
            "rc_timeout_seen": self.rc_timeout_seen,
            "rc_timeout_position_seen": self.rc_timeout_position_seen,
            "rc_timeout_fallback_seen": self.rc_timeout_fallback_seen,
            "odometry_timestamp_gap_count": self.odometry_timestamp_gap_count,
            "supervisor_observer_gap_count": self.supervisor_observer_gap_count,
            "trajectory_publish_count": trajectory_publish_count,
            "trajectory_publish_gap_count": trajectory_publish_gap_count,
            "trajectory_publish_max_gap_s": trajectory_publish_max_gap_s,
            "odometry_timestamp_fault_seen": self.odometry_timestamp_fault_seen,
            "odometry_timestamp_fault_fallback_seen": (
                self.odometry_timestamp_fault_fallback_seen
            ),
        }
        if self.expect_rc_timeout_fallback and not self.rc_timeout_fallback_seen:
            summary["reason"] = "RC timeout fallback did not reach PX4 Position or Land mode"
        success = bool(
            success and
            (not self.expect_rc_timeout_fallback or
             (self.rc_timeout_seen and self.rc_timeout_fallback_seen))
        )
        summary["success"] = success
        (self.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("CPP_NMPC_SITL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
        self.finished = True

    def stop_trajectory_publisher(self) -> None:
        """Stop and join the publisher before the ROS node is destroyed."""
        self._trajectory_stop.set()
        thread = getattr(self, "_trajectory_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("background/sitl_smoke_cpp/cpp_hover_smoke"),
    )
    parser.add_argument("--trajectory", choices=("hover", "point_1m", "circle", "figure8"), default="hover")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--step-dwell", type=float, default=2.0)
    parser.add_argument(
        "--reference-sample-time", type=float, default=0.01,
        help="reference trajectory sampling step; C++ NMPC control period remains 0.01 s (100 Hz)",
    )
    parser.add_argument("--safety-drift-limit", type=float, default=None)
    parser.add_argument(
        "--expect-rc-timeout-fallback", action="store_true",
        help="RC 超时后必须看到 PX4 Position mode，再结束 SITL 测试",
    )
    parser.add_argument("--position-bias-rw-std-m-sqrt-s", type=float, default=0.0)
    parser.add_argument("--velocity-bias-rw-std-m-s-sqrt-s", type=float, default=0.0)
    parser.add_argument(
        "--skip-params",
        action="store_true",
        help="不在本进程内修改 PX4 参数；由外层回归编排器统一管理参数。",
    )
    args = parser.parse_args()
    if args.safety_drift_limit is not None and args.safety_drift_limit <= 0.0:
        parser.error("safety-drift-limit must be positive")
    if args.reference_sample_time <= 0.0:
        parser.error("reference-sample-time must be positive")
    parameters = DEFAULT_PARAMETERS + (
        GuardedParameter(
            "SIM_GZ_ODOM_RW_P",
            args.position_bias_rw_std_m_sqrt_s,
            "external odometry position random walk",
        ),
        GuardedParameter(
            "SIM_GZ_ODOM_RW_V",
            args.velocity_bias_rw_std_m_s_sqrt_s,
            "external odometry velocity random walk",
        ),
    )
    parameter_context = nullcontext() if args.skip_params else ParamGuard(parameters=parameters)
    with parameter_context:
        rclpy.init()
        node = CppHoverSupervisor(
            args.output_directory, args.trajectory, args.radius, args.speed,
            args.step_dwell, args.safety_drift_limit, args.expect_rc_timeout_fallback,
            args.reference_sample_time,
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            while rclpy.ok() and not node.finished:
                executor.spin_once(timeout_sec=0.1)
        finally:
            node.stop_trajectory_publisher()
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
    return 0 if node.finished and not node.failure_reason else 1


if __name__ == "__main__":
    raise SystemExit(main())
