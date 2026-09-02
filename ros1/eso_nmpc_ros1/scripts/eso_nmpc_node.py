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
from nmpc.reference import stationary_reference
from nmpc.solver.acados_solver import AcadosNmpc
from eso_nmpc_ros1.frames import mavros_odometry_to_state


class Ros1NmpcNode:
    """Guarded ROS 1 offboard hover node using MAVROS topics/services."""

    def __init__(self) -> None:
        config_path = Path(rospy.get_param("~config_path", str(ROOT / "config/nmpc.yaml")))
        self.config = load_config(config_path)
        self.controller = AcadosNmpc(self.config)
        self.altitude = float(rospy.get_param("~altitude", 1.0))
        self.prestream_s = float(rospy.get_param("~prestream_s", 1.5))
        self.flight_duration_s = float(rospy.get_param("~flight_duration_s", 12.0))
        self.offboard_mode = str(rospy.get_param("~offboard_mode", "OFFBOARD"))
        self.output_csv = Path(rospy.get_param("~timing_log", ""))

        self.odom: Odometry | None = None
        self.mavros_state: State | None = None
        self.initial_position_ned: np.ndarray | None = None
        self.target_position_ned: np.ndarray | None = None
        self.phase = "WAIT"
        self.phase_started = time.monotonic()
        self.last_odom_receive = 0.0
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

    def _solve_and_publish(self) -> None:
        if self.target_position_ned is None:
            return
        state = self._state()
        reference = stationary_reference(
            self.target_position_ned,
            self.config.controller.horizon_steps,
            self.config.hover_thrust,
        )
        received = self.last_odom_receive
        started = time.monotonic()
        command = self.controller.solve(state, reference)
        published = self._publish(command.thrust, command.body_rate)
        self.records.append({
            "t_monotonic": published,
            "rx_to_pub_ms": (published - received) * 1000.0,
            "solve_ms": self.controller.last_solve_time * 1000.0,
            "position_x_ned": state[0],
            "position_y_ned": state[1],
            "position_z_ned": state[2],
        })
        _ = started  # retained as a clear boundary for future phase timing.

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
        if now - self.last_odom_receive > 1.0:
            rospy.logerr_throttle(2.0, "MAVROS odometry is stale")
            return
        state = self._state()
        if not np.all(np.isfinite(state)):
            return
        if self.phase == "WAIT":
            self.initial_position_ned = state[:3].copy()
            self.target_position_ned = self.initial_position_ned + np.array([0.0, 0.0, -self.altitude])
            self._change_phase("PRESTREAM")
        try:
            self._solve_and_publish()
        except Exception as error:
            rospy.logerr_throttle(2.0, "NMPC solve failed: %s", error)
            return
        if self.phase == "PRESTREAM" and now - self.phase_started >= self.prestream_s:
            self._change_phase("OFFBOARD")
        elif self.phase == "OFFBOARD":
            self._call_services()
            if self.mavros_state.mode == self.offboard_mode:
                self._change_phase("ARMING")
        elif self.phase == "ARMING":
            self._call_services()
            if self.mavros_state.armed:
                self._change_phase("FLIGHT")
        elif self.phase == "FLIGHT" and now - self.phase_started >= self.flight_duration_s:
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
