#!/usr/bin/env python3
"""Measure the raw ROS receive rate of PX4 VehicleOdometry without NMPC load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import monotonic

import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def interval_summary(intervals_s: list[float]) -> dict[str, float | int]:
    if not intervals_s:
        return {"count": 0}
    return {
        "count": len(intervals_s),
        "mean_ms": 1000.0 * mean(intervals_s),
        "median_ms": 1000.0 * median(intervals_s),
        "p95_ms": 1000.0 * percentile(intervals_s, 0.95),
        "p99_ms": 1000.0 * percentile(intervals_s, 0.99),
        "min_ms": 1000.0 * min(intervals_s),
        "max_ms": 1000.0 * max(intervals_s),
    }


class OdometryRateProbe(Node):
    def __init__(self) -> None:
        super().__init__("px4_odometry_rate_probe")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.receive_times_s: list[float] = []
        self.sample_timestamps_us: list[int] = []
        self.publication_timestamps_us: list[int] = []
        self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self._on_odometry,
            qos,
        )

    def _on_odometry(self, message: VehicleOdometry) -> None:
        self.receive_times_s.append(monotonic())
        sample = int(getattr(message, "timestamp_sample", 0))
        self.sample_timestamps_us.append(sample if sample > 0 else int(message.timestamp))
        self.publication_timestamps_us.append(int(message.timestamp))

    def report(self, requested_duration_s: float) -> dict[str, object]:
        receive_intervals = [
            following - previous
            for previous, following in zip(self.receive_times_s, self.receive_times_s[1:])
            if following > previous
        ]
        sample_intervals = [
            1.0e-6 * (following - previous)
            for previous, following in zip(
                self.sample_timestamps_us, self.sample_timestamps_us[1:]
            )
            if following > previous
        ]
        duplicates = sum(
            following == previous
            for previous, following in zip(
                self.sample_timestamps_us, self.sample_timestamps_us[1:]
            )
        )
        non_monotonic = sum(
            following < previous
            for previous, following in zip(
                self.sample_timestamps_us, self.sample_timestamps_us[1:]
            )
        )
        receive_span = (
            self.receive_times_s[-1] - self.receive_times_s[0]
            if len(self.receive_times_s) >= 2
            else 0.0
        )
        sample_span = (
            1.0e-6 * (self.sample_timestamps_us[-1] - self.sample_timestamps_us[0])
            if len(self.sample_timestamps_us) >= 2
            else 0.0
        )
        count = len(self.receive_times_s)
        return {
            "topic": "/fmu/out/vehicle_odometry",
            "requested_duration_s": requested_duration_s,
            "message_count": count,
            "receive_span_s": receive_span,
            "receive_rate_hz": (count - 1) / receive_span if receive_span > 0.0 else 0.0,
            "sample_span_s": sample_span,
            "sample_rate_hz": (count - 1) / sample_span if sample_span > 0.0 else 0.0,
            "receive_intervals": interval_summary(receive_intervals),
            "sample_intervals": interval_summary(sample_intervals),
            "duplicate_sample_timestamps": duplicates,
            "non_monotonic_sample_timestamps": non_monotonic,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.startup_timeout <= 0.0:
        parser.error("duration and startup-timeout must be positive")

    rclpy.init()
    node = OdometryRateProbe()
    first_message_deadline = monotonic() + args.startup_timeout
    while rclpy.ok() and not node.receive_times_s and monotonic() < first_message_deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.receive_times_s:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("no VehicleOdometry received before startup timeout")

    deadline = monotonic() + args.duration
    while rclpy.ok() and monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    report = node.report(args.duration)
    node.destroy_node()
    rclpy.shutdown()

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
