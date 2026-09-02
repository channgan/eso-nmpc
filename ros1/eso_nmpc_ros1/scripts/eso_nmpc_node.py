#!/usr/bin/env python3
"""ROS 1/MAVROS hover adapter for the shared ESO-NMPC core.

This is the first ROS 1 vertical slice: it consumes MAVROS local odometry,
solves the existing NMPC problem, and publishes body-rate/thrust attitude
setpoints. The mathematical core remains in ``nmpc/``; ROS 1 message and
frame handling is isolated here.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_SRC = ROOT / "ros1/eso_nmpc_ros1/src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from nmpc.config import load_config
from nmpc.px4 import thrust_newton_to_px4
from nmpc.setpoint import (
    PresetTrajectory,
    PresetTrajectoryParameters,
    build_reference_horizon,
)
from nmpc.solver.acados_solver import AcadosNmpc
from eso_nmpc_ros1.frames import mavros_odometry_to_state


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class Ros1NmpcNode:
    """Guarded ROS 1 offboard trajectory node using MAVROS topics/services."""

    def __init__(self) -> None:
        config_path = Path(rospy.get_param("~config_path", str(ROOT / "config/nmpc.yaml")))
        self.config = load_config(config_path)
        self.controller = AcadosNmpc(self.config)
        self.altitude = float(rospy.get_param("~altitude", 1.0))
        self.prestream_s = float(rospy.get_param("~prestream_s", 1.5))
        self.trajectory_mode = str(rospy.get_param("~trajectory", "hover"))
        self.radius = float(rospy.get_param("~radius", 0.5))
        self.speed = float(rospy.get_param("~speed", 0.25))
        self.ascent_s = float(rospy.get_param("~ascent_s", 4.0))
        self.hold_s = float(rospy.get_param("~hold_s", 8.0))
        self.transition_s = float(rospy.get_param("~transition_s", 2.0))
        self.descent_s = float(rospy.get_param("~descent_s", 4.0))
        self.settle_s = float(rospy.get_param("~settle_s", 1.0))
        self.state_timeout_s = float(rospy.get_param("~state_timeout_s", 1.0))
        self.service_timeout_s = float(rospy.get_param("~service_timeout_s", 8.0))
        self.offboard_mode = str(rospy.get_param("~offboard_mode", "OFFBOARD"))
        log_root = Path(rospy.get_param("~log_root", "/home/ljt/nmpc_log"))
        timing_log = str(rospy.get_param("~timing_log", "")).strip()
        if timing_log:
            self.output_csv = Path(timing_log)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_csv = log_root / timestamp / "nmpc_timing.csv"

        self.odom: Odometry | None = None
        self.mavros_state: State | None = None
        self.initial_position_ned: np.ndarray | None = None
        self.trajectory: PresetTrajectory | None = None
        self.phase = "WAIT"
        self.phase_started = time.monotonic()
        self.last_odom_receive = 0.0
        self.odom_sequence = 0
        self.solved_odom_sequence = -1
        self.last_service_call = 0.0
        self.finished = False
        self.records: list[dict[str, float]] = []

        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=20
        )
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber("/mavros/state", State, self._on_state, queue_size=20)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(self.config.controller.sample_time), self._tick
        )
        rospy.on_shutdown(self._shutdown)

    def _on_odom(self, message: Odometry) -> None:
        self.odom = message
        self.last_odom_receive = time.monotonic()
        self.odom_sequence += 1

    def _on_state(self, message: State) -> None:
        self.mavros_state = message

    def _state(self) -> np.ndarray:
        if self.odom is None:
            raise RuntimeError("odometry is not available")
        position = self.odom.pose.pose.position
        velocity = self.odom.twist.twist.linear
        orientation = self.odom.pose.pose.orientation
        angular = self.odom.twist.twist.angular
        return mavros_odometry_to_state(
            np.array([position.x, position.y, position.z]),
            np.array([velocity.x, velocity.y, velocity.z]),
            np.array([orientation.w, orientation.x, orientation.y, orientation.z]),
            np.array([angular.x, angular.y, angular.z]),
        )

    def _publish(self, command_thrust: float, body_rate: np.ndarray) -> float:
        message = AttitudeTarget()
        message.header.stamp = rospy.Time.now()
        message.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        message.body_rate.x = float(body_rate[0])
        message.body_rate.y = float(body_rate[1])
        message.body_rate.z = float(body_rate[2])
        # MAVROS AttitudeTarget uses positive normalized thrust [0, 1].
        message.thrust = float(-thrust_newton_to_px4(command_thrust, self.config))
        self.setpoint_pub.publish(message)
        return time.monotonic()

    def _reference(self, state: np.ndarray, elapsed: float):
        if self.trajectory is None:
            raise RuntimeError("trajectory is not initialized")
        cfg = self.config
        return build_reference_horizon(
            self.trajectory.sample,
            start_time=elapsed,
            horizon_steps=cfg.controller.horizon_steps,
            sample_time=cfg.controller.sample_time,
            mass=cfg.model.mass,
            gravity=cfg.model.gravity,
            thrust_min=cfg.limits.thrust_min,
            thrust_max=cfg.limits.thrust_max,
            body_rate_max=cfg.limits.body_rate_max,
            quaternion_anchor=state[6:10],
        )

    def _solve_and_publish(self, now: float) -> None:
        if self.trajectory is None or self.odom_sequence == self.solved_odom_sequence:
            return
        state = self._state()
        elapsed = 0.0 if self.phase != "FLIGHT" else now - self.phase_started
        reference = self._reference(state, elapsed)
        received = self.last_odom_receive
        command = self.controller.solve(state, reference)
        published = self._publish(command.thrust, command.body_rate)
        self.solved_odom_sequence = self.odom_sequence
        target = self.trajectory.sample(elapsed)
        self.records.append({
            "t_monotonic": published,
            "rx_to_pub_ms": (published - received) * 1000.0,
            "solve_ms": self.controller.last_solve_time * 1000.0,
            "position_x_ned": state[0],
            "position_y_ned": state[1],
            "position_z_ned": state[2],
            "reference_x_ned": target.position[0],
            "reference_y_ned": target.position[1],
            "reference_z_ned": target.position[2],
            "tracking_error_m": float(np.linalg.norm(state[:3] - target.position)),
        })

    def _change_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        rospy.loginfo("ESO-NMPC ROS1 phase=%s", phase)

    def _call_services(self) -> None:
        now = time.monotonic()
        if now - self.last_service_call < 0.5:
            return
        self.last_service_call = now
        if self.phase == "OFFBOARD":
            try:
                self.set_mode(custom_mode=self.offboard_mode)
            except rospy.ServiceException as error:
                rospy.logwarn_throttle(2.0, "set_mode failed: %s", error)
        elif self.phase == "ARMING":
            try:
                self.arm(value=True)
            except rospy.ServiceException as error:
                rospy.logwarn_throttle(2.0, "arming failed: %s", error)

    def _tick(self, _event: rospy.timer.TimerEvent) -> None:
        if self.finished or self.odom is None or self.mavros_state is None:
            return
        now = time.monotonic()
        if now - self.last_odom_receive > self.state_timeout_s:
            rospy.logerr_throttle(2.0, "MAVROS odometry is stale")
            return
        state = self._state()
        if not np.all(np.isfinite(state)):
            return
        if self.phase == "WAIT":
            self.initial_position_ned = state[:3].copy()
            self.trajectory = PresetTrajectory(
                self.initial_position_ned,
                _quaternion_yaw(state[6:10]),
                PresetTrajectoryParameters(
                    mode=self.trajectory_mode,
                    altitude=self.altitude,
                    ascent=self.ascent_s,
                    hold=self.hold_s,
                    transition=self.transition_s,
                    descent=self.descent_s,
                    settle=self.settle_s,
                    radius=self.radius,
                    speed=self.speed,
                ),
            )
            self._change_phase("PRESTREAM")
        try:
            self._solve_and_publish(now)
        except Exception as error:
            rospy.logerr_throttle(2.0, "NMPC solve failed: %s", error)
            if self.mavros_state.armed:
                try:
                    self.set_mode(custom_mode="AUTO.LAND")
                except rospy.ServiceException:
                    pass
                self._change_phase("LANDING")
            return
        if self.phase == "PRESTREAM" and now - self.phase_started >= self.prestream_s:
            self._change_phase("OFFBOARD")
        elif self.phase == "OFFBOARD":
            self._call_services()
            if self.mavros_state.mode == self.offboard_mode:
                self._change_phase("ARMING")
            elif now - self.phase_started > self.service_timeout_s:
                rospy.logerr("timed out entering %s", self.offboard_mode)
                self._finish()
        elif self.phase == "ARMING":
            self._call_services()
            if self.mavros_state.armed:
                self._change_phase("FLIGHT")
            elif now - self.phase_started > self.service_timeout_s:
                rospy.logerr("timed out arming")
                self._finish()
        elif (
            self.phase == "FLIGHT"
            and self.trajectory is not None
            and now - self.phase_started >= self.trajectory.duration
        ):
            try:
                self.set_mode(custom_mode="AUTO.LAND")
            except rospy.ServiceException as error:
                rospy.logwarn("AUTO.LAND request failed: %s", error)
            self._change_phase("LANDING")
        elif self.phase == "LANDING" and not self.mavros_state.armed:
            self._finish()

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.output_csv:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.output_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.records[0]) if self.records else ["t_monotonic"])
                writer.writeheader()
                writer.writerows(self.records)
        rospy.loginfo("ESO-NMPC ROS1 finished; records=%d", len(self.records))
        rospy.signal_shutdown("trajectory complete")

    def _shutdown(self) -> None:
        self.finished = True


def main() -> None:
    rospy.init_node("eso_nmpc_ros1")
    Ros1NmpcNode()
    rospy.spin()


if __name__ == "__main__":
    main()
