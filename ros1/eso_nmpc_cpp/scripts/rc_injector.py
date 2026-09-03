#!/usr/bin/env python3
"""Deterministic ROS1 RC/AUX injector for the PX4 SITL RC-NMPC test."""

import rospy
from mavros_msgs.msg import ManualControl, RCIn
from std_msgs.msg import Bool


class RcInjector:
    def __init__(self):
        self.start = rospy.Time.now()
        self.manual_pub = rospy.Publisher(
            "/mavros/manual_control/control", ManualControl, queue_size=10
        )
        self.rc_pub = rospy.Publisher("/mavros/rc/in", RCIn, queue_size=10)
        self.enable_pub = rospy.Publisher("/nmpc/control_enabled", Bool, queue_size=2)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.tick)

    def tick(self, _event):
        elapsed = (rospy.Time.now() - self.start).to_sec()
        # Keep the controller enabled long enough to enter OFFBOARD and fly.
        enabled = elapsed < 18.0
        self.enable_pub.publish(Bool(data=enabled))

        # AUX6 high from 3 s to 14 s selects the RC-NMPC reference source.
        aux_enabled = 3.0 <= elapsed < 14.0
        channels = [1500] * 8
        channels[5] = 2000 if aux_enabled else 1000
        self.rc_pub.publish(RCIn(rssi=100, channels=channels))

        # Neutral, then a bounded forward command, then neutral braking.
        pitch = 0.0
        if 6.0 <= elapsed < 10.0:
            pitch = 0.35
        elif 10.0 <= elapsed < 12.0:
            pitch = -0.20
        self.manual_pub.publish(
            ManualControl(x=0.0, y=pitch, z=0.0, r=0.0, buttons=0)
        )


if __name__ == "__main__":
    rospy.init_node("eso_nmpc_rc_injector")
    RcInjector()
    rospy.spin()
