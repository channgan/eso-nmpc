#!/usr/bin/env python3
"""Observe PX4's native Offboard-loss transition.

The expected fallback is deliberately checked without sending a replacement
mode command.  Sending LAND as soon as PX4 leaves Offboard would hide whether
the Commander selected Position mode (companion loss) or Land mode (RC loss).
"""

from __future__ import annotations

import rclpy
from px4_msgs.msg import VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class OffboardLossMonitor(Node):
    def __init__(self, expected: str = "position") -> None:
        super().__init__("offboard_loss_monitor")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self.status_cb, qos)
        self.started = self.get_clock().now()
        self.offboard_seen = False
        self.loss_seen = False
        self.last_state = None
        if expected not in {"position", "land"}:
            raise ValueError("expected must be 'position' or 'land'")
        self.expected = expected
        self.expected_state = (
            VehicleStatus.NAVIGATION_STATE_POSCTL
            if expected == "position"
            else VehicleStatus.NAVIGATION_STATE_AUTO_LAND
        )
        self.expected_seen = False
        self.create_timer(0.1, self.tick)

    def status_cb(self, message: VehicleStatus) -> None:
        state = int(message.nav_state)
        if state != self.last_state:
            self.get_logger().info("nav_state=%d arming_state=%d" % (state, message.arming_state))
            self.last_state = state
        if state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.offboard_seen = True
        elif self.offboard_seen and not self.loss_seen and state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.loss_seen = True
            if state == self.expected_state:
                self.expected_seen = True
                self.get_logger().info(
                    "PX4 Offboard-loss fallback matched expected %s mode (nav_state=%d)"
                    % (self.expected, state)
                )
            else:
                self.get_logger().error(
                    "PX4 Offboard-loss fallback was %s, expected %s (nav_state=%d)"
                    % (state, self.expected, self.expected_state)
                )

    def tick(self) -> None:
        if (self.get_clock().now() - self.started).nanoseconds * 1.0e-9 > 35.0:
            self.get_logger().info(
                "monitor timeout; loss_seen=%s expected_seen=%s" %
                (self.loss_seen, self.expected_seen)
            )
            raise SystemExit(0)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", choices=("position", "land"), default="position")
    args = parser.parse_args()
    rclpy.init()
    node = OffboardLossMonitor(args.expected)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.expected_seen else 1


if __name__ == "__main__":
    raise SystemExit(main())
