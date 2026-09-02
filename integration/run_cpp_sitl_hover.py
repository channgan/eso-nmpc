#!/usr/bin/env python3
"""Guarded PX4 SITL hover smoke for the single-process C++ NMPC node."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from integration.mavlink_params import DEFAULT_PARAMETERS, GuardedParameter, ParamGuard
from nmpc.trajectory import quintic_segment, smooth_profile


class CppHoverSupervisor(Node):
    def __init__(self, output_directory: Path, trajectory: str = "hover",
                 radius: float = 2.0, speed: float = 1.0,
                 point_hold_duration: float = 2.0,
                 safety_drift_limit: float | None = None) -> None:
        super().__init__("cpp_nmpc_hover_supervisor")
        self.output_directory = output_directory
        if trajectory not in ("hover", "point_1m", "circle", "figure8"):
            raise ValueError("trajectory must be hover, point_1m, circle or figure8")
        self.trajectory = trajectory
        self.radius = float(radius)
        self.speed = float(speed)
        self.point_hold_duration = float(point_hold_duration)
        self.safety_drift_limit = safety_drift_limit
        self.points = 31
        self.sample_time = 0.02
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
        self.records: list[dict[str, float]] = []
        self.trajectory_sequence = 0

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
        trajectory_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odometry, output_qos
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, output_qos
        )
        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._on_land_detected,
            output_qos,
        )
        self.trajectory_publisher = self.create_publisher(
            NmpcTrajectorySetpoint, "/nmpc/in/trajectory_setpoint", trajectory_qos
        )
        self.enable_publisher = self.create_publisher(
            Bool, "/nmpc/control_enabled", input_qos
        )
        self.command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", input_qos
        )
        self.timer = self.create_timer(self.sample_time, self._tick)

    def _timestamp_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def _on_odometry(self, message: VehicleOdometry) -> None:
        self.odometry = message

    def _on_status(self, message: VehicleStatus) -> None:
        self.status = message

    def _on_land_detected(self, message: VehicleLandDetected) -> None:
        self.land_detected = message

    def _enable(self, enabled: bool) -> None:
        message = Bool()
        message.data = enabled
        self.enable_publisher.publish(message)

    def _command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp_us()
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.command = int(command)
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_publisher.publish(message)

    def _sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.initial_position is not None
        if time_s <= 0.0:
            return self.initial_position.copy(), np.zeros(3), np.zeros(3)
        if time_s < self.ascent_duration:
            fraction, velocity_fraction, acceleration_fraction = smooth_profile(
                time_s, self.ascent_duration
            )
            delta = np.array([0.0, 0.0, -self.altitude])
            return (
                self.initial_position + fraction * delta,
                velocity_fraction * delta,
                acceleration_fraction * delta,
            )
        hover = self.initial_position + np.array([0.0, 0.0, -self.altitude])
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

    def _publish_horizon(self, elapsed: float) -> np.ndarray:
        samples = [self._sample(elapsed + i * self.sample_time) for i in range(self.points)]
        position = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        velocity = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        acceleration = np.asarray([sample[2] for sample in samples], dtype=np.float32)
        jerk = np.empty_like(acceleration)
        jerk[:-1] = np.diff(acceleration, axis=0) / self.sample_time
        jerk[-1] = jerk[-2] if self.points > 1 else 0.0
        message = NmpcTrajectorySetpoint()
        message.timestamp = self._timestamp_us()
        self.trajectory_sequence += 1
        message.sequence = self.trajectory_sequence
        message.points = self.points
        message.sample_time = self.sample_time
        message.position[: 3 * self.points] = position.reshape(-1).tolist()
        message.velocity[: 3 * self.points] = velocity.reshape(-1).tolist()
        message.acceleration[: 3 * self.points] = acceleration.reshape(-1).tolist()
        message.jerk[: 3 * self.points] = jerk.reshape(-1).tolist()
        message.yaw[: self.points] = [self.initial_yaw] * self.points
        self.trajectory_publisher.publish(message)
        return position[0].astype(float)

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
            self._publish_horizon(0.0)
            self._enable(True)
            if now - self.phase_started > 1.5:
                self._set_phase("ENTER_OFFBOARD")
            return

        if self.phase == "ENTER_OFFBOARD":
            self._publish_horizon(0.0)
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
            self._publish_horizon(0.0)
            self._enable(True)
            if now - self.last_command_time > 0.5:
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_command_time = now
            if self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.initial_position = np.asarray(self.odometry.position, dtype=float)
                self.flight_started = now
                self._set_phase("FLIGHT")
            elif now - self.phase_started > 8.0:
                self._abort("arming timeout")
            return

        if self.phase == "FLIGHT":
            elapsed = now - self.flight_started
            reference = self._publish_horizon(elapsed)
            self._enable(True)
            position = np.asarray(self.odometry.position, dtype=float)
            velocity = np.asarray(self.odometry.velocity, dtype=float)
            error = position - reference
            self.records.append(
                {
                    "time_s": elapsed,
                    "position_x": position[0], "position_y": position[1], "position_z": position[2],
                    "reference_x": reference[0], "reference_y": reference[1], "reference_z": reference[2],
                    "velocity_x": velocity[0], "velocity_y": velocity[1], "velocity_z": velocity[2],
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
        csv_path = self.output_directory / "trajectory.csv"
        if self.records:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
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
        }
        (self.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("CPP_NMPC_SITL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
        self.finished = True


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
    parser.add_argument("--safety-drift-limit", type=float, default=None)
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
            args.step_dwell, args.safety_drift_limit
        )
        try:
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    return 0 if node.finished and not node.failure_reason else 1


if __name__ == "__main__":
    raise SystemExit(main())
