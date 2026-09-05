#!/usr/bin/env python3
"""Inject a deterministic ManualControlSetpoint sequence into PX4 SITL."""

from __future__ import annotations

import rclpy
import argparse
from px4_msgs.msg import ManualControlSetpoint, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class RcInjector(Node):
    def __init__(self, aux_channel: int = 1, drop_after: float = -1.0,
                 topic: str = "/fmu/out/manual_control_setpoint") -> None:
        super().__init__("rc_nmpc_sitl_injector")
        if aux_channel < 1 or aux_channel > 6:
            raise ValueError("aux_channel must be in [1, 6]")
        self.aux_channel = int(aux_channel)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            ManualControlSetpoint, topic, qos
        )
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self.status_cb, qos)
        self.armed = False
        self.started = self.get_clock().now()
        self.flight_start_ns: int | None = None
        self.drop_after = float(drop_after)
        self.create_timer(0.01, self.publish)

    def status_cb(self, message: VehicleStatus) -> None:
        armed = message.arming_state == VehicleStatus.ARMING_STATE_ARMED
        if armed and not self.armed:
            self.flight_start_ns = self.get_clock().now().nanoseconds
        self.armed = armed

    def publish(self) -> None:
        message = ManualControlSetpoint()
        now_ns = self.get_clock().now().nanoseconds
        message.timestamp = now_ns // 1000
        message.timestamp_sample = message.timestamp
        message.valid = True
        message.data_source = ManualControlSetpoint.SOURCE_RC
        # The configured AUX channel selects RC-NMPC.  Keep it low until the
        # aircraft is armed and settled in the supervisor's first hover second.
        elapsed = ((now_ns - self.flight_start_ns) * 1.0e-9) if self.flight_start_ns else -1.0
        for channel in range(1, 7):
            setattr(message, f"aux{channel}", -1.0)
        setattr(message, f"aux{self.aux_channel}", 1.0 if elapsed >= 1.0 else -1.0)
        if self.drop_after >= 0.0 and elapsed >= self.drop_after:
            # Keep publishing an explicitly invalid, selected RC frame.  In
            # SITL PX4 may also publish its native ManualControlSetpoint on
            # the same output topic; simply stopping this injector would let
            # that native stream mask the intended node-side timeout.
            message.valid = False
            self.publisher.publish(message)
            return
        # Forward stick for 3 s, then release to test acceleration braking and
        # position hold.  Roll/pitch/yaw/throttle are normalized [-1, 1].
        if 1.0 <= elapsed < 4.0:
            message.pitch = 0.35
        else:
            message.pitch = 0.0
        message.roll = 0.0
        message.yaw = 0.0
        message.throttle = 0.0
        self.publisher.publish(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux-channel", type=int, choices=range(1, 7), default=1,
                        help="AUX channel to raise for RC-NMPC selection")
    parser.add_argument("--drop-after", type=float, default=-1.0,
                        help="send invalid RC frames this many seconds after arming")
    parser.add_argument("--topic", default="/fmu/out/manual_control_setpoint",
                        help="ManualControlSetpoint topic")
    args = parser.parse_args()
    rclpy.init()
    node = RcInjector(args.aux_channel, args.drop_after, args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
