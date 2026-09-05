#!/usr/bin/env python3
"""Run guarded PX4 SITL trajectory regression tests with the NMPC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic

import numpy as np
import rclpy
from px4_msgs.msg import (
    ManualControlSetpoint,
    NmpcTrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleAttitude,
    VehicleLocalPosition,
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
from nmpc.observer import VelocityLESO
from nmpc.px4 import thrust_newton_to_px4
from nmpc.reference import inverse_dynamics_attitude_and_thrust
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
    altitude: float = 1.0
    radius: float = 0.5
    speed: float = 0.25
    rc_duration: float = 60.0
    rc_geofence_radius: float = 5.0
    validate_model: bool = False
    direct_callback: bool = False
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
    takeoff_attitude_gain_rp: float = 4.0
    takeoff_attitude_gain_yaw: float = 2.0
    takeoff_horizontal_drift_limit: float = 1.0
    takeoff_converge_tilt_rad: float = np.deg2rad(3.0)
    takeoff_converge_body_rate: float = 0.15
    takeoff_horizontal_kp: float = 1.0
    takeoff_horizontal_kd: float = 1.8
    takeoff_horizontal_acceleration_max: float = 2.0
    # PX4 status messages can arrive one sample late while DDS is catching up.
    # Treat a brief non-Offboard status as a transient and only trigger the
    # safety abort when it persists beyond this grace period.
    offboard_loss_grace_period: float = 0.5
    cg_offset_x_m: float = 0.0
    # Runtime disturbance declaration.  Wind is applied by the Gazebo world;
    # these values are recorded with the run so results remain self-describing.
    wind_velocity_x_m_s: float = 0.0
    wind_velocity_y_m_s: float = 0.0
    wind_velocity_z_m_s: float = 0.0

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
        if config.controller.horizon_steps + 1 > NmpcTrajectorySetpoint.MAX_POINTS:
            raise ValueError(
                "configured NMPC horizon exceeds the NmpcTrajectorySetpoint protocol capacity"
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
        # A complete trajectory is a latest-wins stream.  Keeping a reliable
        # depth-10 queue allows old, large horizon messages to accumulate when
        # the solver briefly occupies the executor; those messages then fail
        # the age check even though a newer trajectory is already available.
        # Best effort + depth one drops obsolete horizons instead of replaying
        # them.  It remains compatible with reliable external publishers.
        trajectory_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odometry, output_qos
        )
        # Gazebo/PX4 publishes these ground-truth topics from the simulation
        # bridge.  They are validation-only inputs: the controller and ESO
        # continue to use vehicle_odometry exactly as before.
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_groundtruth_v1",
            self._on_groundtruth_local_position,
            output_qos,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude_groundtruth",
            self._on_groundtruth_attitude,
            output_qos,
        )
        self.create_subscription(
            ManualControlSetpoint,
            "/fmu/out/manual_control_setpoint",
            self._on_manual_control,
            output_qos,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self._on_status,
            output_qos,
        )
        self.create_subscription(
            VehicleCommandAck,
            "/fmu/out/vehicle_command_ack",
            self._on_ack,
            output_qos,
        )
        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._on_land_detected,
            output_qos,
        )
        self.create_subscription(
            NmpcTrajectorySetpoint,
            "/nmpc/in/trajectory_setpoint",
            self._on_direct_trajectory,
            trajectory_qos,
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
        self.direct_trajectory_publisher = self.create_publisher(
            NmpcTrajectorySetpoint, "/nmpc/in/trajectory_setpoint", trajectory_qos
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
        self.groundtruth_local_position: VehicleLocalPosition | None = None
        self.groundtruth_attitude: VehicleAttitude | None = None
        self.measured_position: np.ndarray | None = None
        self.measured_velocity: np.ndarray | None = None
        self.status: VehicleStatus | None = None
        self.land_detected: VehicleLandDetected | None = None
        self.initial_position: np.ndarray | None = None
        self.reference_quaternion: np.ndarray | None = None
        self.reference_yaw = 0.0
        self.preset_source: PresetTrajectory | None = None
        self.last_reference_segment: str | None = None
        self.last_reference_timing: dict[str, float] = {}
        self.rc_source: RcVelocityReference | None = None
        self.manual_control: ManualControlSetpoint | None = None
        self.direct_trajectory: NmpcTrajectorySetpoint | None = None
        # Parsed once in the DDS callback.  The control thread only consumes
        # this immutable snapshot, avoiding repeated list->NumPy conversion
        # and reshape work on every solve.
        self._direct_trajectory_cache: KinematicTrajectory | None = None
        self.last_direct_trajectory_time = 0.0
        self.direct_trajectory_sequence = 0
        self.last_direct_trajectory_timestamp_age_s: float | None = None
        self.last_manual_control_time = 0.0
        self.flight_started = 0.0
        self.flight_started_px4_us = 0
        self.last_solved_odometry_us = 0
        self.last_trajectory_odometry_us = 0
        self.trajectory_elapsed = 0.0
        self.timestamp_jump_count = 0
        self.timestamp_sample_age_invalid_count = 0
        self.last_odometry_receive_steady_us = 0
        self.last_odometry_receive_steady_s = 0.0
        self.last_odometry_sample_steady_us = 0
        self._odometry_receive_lock = threading.Lock()
        self._odometry_receive_history: deque[float] = deque(maxlen=128)
        self.finished = False
        # ``_finish`` can be reached from the odometry-driven controller
        # thread or the ROS spin-loop shutdown watchdog.  Serialize this
        # transition so both paths cannot write duplicate artifacts.
        self._finish_lock = threading.Lock()
        self._finishing = False
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
        # World-frame acceleration disturbance estimate. On the ESO branch this
        # is produced by a velocity-channel LESO and shared by the OCP and
        # inverse-dynamics feed-forward reference.
        self.disturbance_estimate = np.zeros(3)
        self.eso_enabled = bool(config.eso.enabled)
        self.eso = VelocityLESO(
            config.eso.bandwidth_rad_s,
            config.eso.disturbance_clamp_m_s2,
            config.eso.innovation_limit_m_s,
        )
        self.eso_active = False
        # Keep the legacy estimator parameters for A/B comparison when ESO is
        # disabled in YAML.
        self.disturbance_tau = 1.0
        self.disturbance_clamp = config.eso.disturbance_clamp_m_s2
        self.last_measurement_state: np.ndarray | None = None
        self.last_measurement_dt = 0.0
        self.last_command_thrust = 0.0
        self.offboard_loss_started = 0.0
        self.offboard_loss_event_count = 0
        # OffboardControlMode is a watchdog heartbeat in PX4.  Publishing it
        # from the same executor callback as the NMPC solve is unsafe: a DDS
        # pause or a solver callback can temporarily block that callback and
        # make PX4 declare Offboard lost.  Keep the most recent valid rate
        # command so a short callback stall can also be bridged safely.
        self._rates_command_lock = threading.Lock()
        self._last_rates_command: tuple[float, np.ndarray, float] | None = None
        self._offboard_heartbeat_period = min(
            0.05, max(0.001, float(config.controller.control_period))
        )
        self._offboard_heartbeat_stop = threading.Event()
        self._offboard_heartbeat_thread: threading.Thread | None = None
        # Wake NMPC from the newest odometry sample instead of imposing a
        # fixed output-rate cap.  The solve runs on its own thread so the DDS
        # executor remains free to receive status and odometry messages.
        self._control_event = threading.Event()
        self._control_stop = threading.Event()
        self._control_thread: threading.Thread | None = None
        # Keep outbound PX4 timestamps in the synchronised epoch domain, but
        # advance them from CLOCK_MONOTONIC.  WSL can step CLOCK_REALTIME when
        # it resynchronises with the Windows host; publishing that step would
        # make PX4 treat otherwise healthy Offboard heartbeats as stale.
        self.timestamp_epoch_us = self.get_clock().now().nanoseconds // 1000
        self.timestamp_monotonic_origin = monotonic()
        # PX4's lockstep clock can advance in a burst when Gazebo catches up.
        # Keep the newest PX4 timestamp so outbound Offboard messages never
        # remain behind that clock after a simulator pause.
        self.last_px4_timestamp_us = 0
        # The control thread is woken by odometry.  It coalesces samples that
        # arrive while acados is solving and always processes the newest state.
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="px4-nmpc-control",
            daemon=True,
        )
        self._control_thread.start()
        self._offboard_heartbeat_thread = threading.Thread(
            target=self._offboard_heartbeat_loop,
            name="px4-offboard-heartbeat",
            daemon=True,
        )
        self._offboard_heartbeat_thread.start()

    def _control_loop(self) -> None:
        """Run NMPC on newest odometry without a fixed timer frequency."""
        while not self._control_stop.is_set():
            self._control_event.wait(0.1)
            if self._control_stop.is_set():
                return
            self._control_event.clear()
            try:
                self._control_tick()
            except Exception as error:
                self.get_logger().error("NMPC control thread failed: %s" % error)
                self._start_disarming("control thread failure", safety_abort=True)

    def _odometry_receive_before(self, time_s: float) -> float:
        """Return the newest odometry callback time not after ``time_s``."""
        with self._odometry_receive_lock:
            for receive_time in reversed(self._odometry_receive_history):
                if receive_time <= time_s:
                    return receive_time
        return time_s

    def _now_us(self) -> int:
        elapsed_us = round((monotonic() - self.timestamp_monotonic_origin) * 1.0e6)
        monotonic_timestamp_us = self.timestamp_epoch_us + elapsed_us
        return max(monotonic_timestamp_us, self.last_px4_timestamp_us)

    def _on_odometry(self, message: VehicleOdometry) -> None:
        t_rx = monotonic()
        self.odometry = message
        # Keep takeoff/convergence safety checks on the estimator measurement.
        # Inject only the requested post-EKF random-walk residual once formal
        # tracking starts, so simulated drift cannot cause a false takeoff
        # failure before NMPC has begun its benchmark.
        receive_steady = t_rx
        self.last_px4_timestamp_us = max(
            self.last_px4_timestamp_us, int(message.timestamp)
        )
        self.measured_position = (
            np.asarray(message.position, dtype=float)
        )
        self.measured_velocity = (
            np.asarray(message.velocity, dtype=float)
        )
        self.last_odometry_time = receive_steady
        timestamp_sample = int(getattr(message, "timestamp_sample", 0))
        if timestamp_sample <= 0:
            timestamp_sample = int(message.timestamp)
        receive_steady_us = round(receive_steady * 1.0e6)
        sample_age_us = int(message.timestamp) - timestamp_sample
        # timestamp and timestamp_sample are converted together by uXRCE, so
        # their difference remains meaningful even if the absolute epoch is
        # corrected.  Express the acquisition time on CLOCK_MONOTONIC by
        # subtracting that estimator age from the local receive time.  DDS
        # transport delay is intentionally retained as part of the real
        # command-to-response path seen by the companion computer.
        if 0 <= sample_age_us <= 100_000:
            sample_steady_us = receive_steady_us - sample_age_us
        else:
            self.timestamp_sample_age_invalid_count += 1
            sample_steady_us = receive_steady_us
        with self._odometry_receive_lock:
            self.last_odometry_receive_steady_us = receive_steady_us
            self.last_odometry_receive_steady_s = receive_steady
            self.last_odometry_sample_steady_us = sample_steady_us
            self._odometry_receive_history.append(receive_steady)
        if self.options.direct_callback:
            # Experimental path: solve synchronously in the DDS callback to
            # measure the minimum wake-up overhead.  The normal path keeps
            # this callback non-blocking and wakes the dedicated controller.
            self._control_tick()
        else:
            self._control_event.set()
        # Tie the Offboard heartbeat to the high-rate PX4 stream as well as the
        # wall timer so simulator clock catch-up cannot create a heartbeat gap.
        if self.phase != "DISARMING" and not self.finished:
            self._publish_mode()

    def _on_groundtruth_local_position(self, message: VehicleLocalPosition) -> None:
        self.groundtruth_local_position = message

    def _on_groundtruth_attitude(self, message: VehicleAttitude) -> None:
        self.groundtruth_attitude = message

    def _odometry_sample_timestamp_us(self) -> int:
        """Return the estimator sample time, falling back for old PX4 messages."""
        assert self.odometry is not None
        timestamp_sample = int(getattr(self.odometry, "timestamp_sample", 0))
        return timestamp_sample if timestamp_sample > 0 else int(self.odometry.timestamp)

    def _on_status(self, message: VehicleStatus) -> None:
        self.status = message

    def _on_manual_control(self, message: ManualControlSetpoint) -> None:
        self.manual_control = message
        self.last_manual_control_time = monotonic()

    def _on_ack(self, message: VehicleCommandAck) -> None:
        self.command_acks.append({"command": int(message.command), "result": int(message.result)})

    def _on_land_detected(self, message: VehicleLandDetected) -> None:
        self.land_detected = message

    def _on_direct_trajectory(self, message: NmpcTrajectorySetpoint) -> None:
        self.direct_trajectory = message
        self._direct_trajectory_cache = None
        try:
            points = int(message.points)
            expected_points = self.config.controller.horizon_steps + 1
            if points == expected_points:
                position = np.asarray(message.position, dtype=float).reshape(-1, 3)
                velocity = np.asarray(message.velocity, dtype=float).reshape(-1, 3)
                acceleration = np.asarray(message.acceleration, dtype=float).reshape(-1, 3)
                jerk = np.asarray(message.jerk, dtype=float).reshape(-1, 3)
                yaw = np.asarray(message.yaw, dtype=float).reshape(-1)
                if (
                    position.shape[0] >= points
                    and velocity.shape[0] >= points
                    and acceleration.shape[0] >= points
                    and jerk.shape[0] >= points
                    and yaw.shape[0] >= points
                ):
                    self._direct_trajectory_cache = KinematicTrajectory(
                        position=position[:points].copy(),
                        velocity=velocity[:points].copy(),
                        acceleration=acceleration[:points].copy(),
                        jerk=jerk[:points].copy(),
                        yaw=yaw[:points].copy(),
                        sample_time=float(message.sample_time),
                    )
        except (TypeError, ValueError):
            # Preserve the original message so the normal validation path can
            # report a precise malformed-trajectory error in the control loop.
            self._direct_trajectory_cache = None
        self.last_direct_trajectory_time = monotonic()
        timestamp = int(message.timestamp)
        if timestamp > 0:
            self.last_direct_trajectory_timestamp_age_s = (
                self._now_us() - timestamp
            ) * 1.0e-6

    def _publish_mode(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._now_us()
        message.body_rate = True
        self.mode_publisher.publish(message)

    def _offboard_heartbeat_loop(self) -> None:
        """Publish PX4's Offboard heartbeat independently of NMPC execution.

        The ROS timer and odometry callback share the executor with the OCP
        solve.  A separate lightweight thread keeps the PX4 heartbeat alive
        when that executor is briefly occupied.  The cached rate setpoint is
        only replayed while it is recent; it is never refreshed by replay, so
        a prolonged stall cannot hide a stale controller output.
        """
        next_error_log = 0.0
        while not self._offboard_heartbeat_stop.wait(self._offboard_heartbeat_period):
            if self.finished or self.phase == "DISARMING":
                continue
            try:
                self._publish_mode()
                with self._rates_command_lock:
                    cached = self._last_rates_command
                if cached is not None:
                    thrust, body_rate, command_time = cached
                    if monotonic() - command_time <= max(
                        0.2, 5.0 * float(self.config.controller.control_period)
                    ):
                        self._publish_rates(thrust, body_rate, cache=False)
            except Exception as error:  # shutdown/DDS races must not kill the heartbeat
                now = monotonic()
                if now >= next_error_log and not self._offboard_heartbeat_stop.is_set():
                    self.get_logger().warning("Offboard heartbeat publish failed: %s" % error)
                    next_error_log = now + 1.0

    def _publish_rates(
        self, thrust_newton: float, body_rate: np.ndarray, *, cache: bool = True
    ) -> int:
        body_rate = np.asarray(body_rate, dtype=float).copy()
        if cache:
            with self._rates_command_lock:
                self._last_rates_command = (
                    float(thrust_newton),
                    body_rate.copy(),
                    monotonic(),
                )
        message = VehicleRatesSetpoint()
        message.timestamp = self._now_us()
        message.roll = float(body_rate[0])
        message.pitch = float(body_rate[1])
        message.yaw = float(body_rate[2])
        message.thrust_body = [0.0, 0.0, thrust_newton_to_px4(thrust_newton, self.config)]
        publish_steady_us = round(monotonic() * 1.0e6)
        self.rates_publisher.publish(message)
        return publish_steady_us

    def destroy_node(self) -> bool:
        """Stop the heartbeat before tearing down the ROS publishers."""
        self._control_stop.set()
        self._control_event.set()
        control_thread = self._control_thread
        if control_thread is not None and control_thread is not threading.current_thread():
            control_thread.join(timeout=2.0)
        self._offboard_heartbeat_stop.set()
        thread = self._offboard_heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        return super().destroy_node()

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
        self._publish_initial_direct_trajectory()

    def _kinematic_horizon(self, elapsed: float) -> KinematicTrajectory:
        sample = self._trajectory_sample
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
        # Keep a local copy as well as publishing over DDS.  During a fresh
        # SITL restart the same-node DDS loopback can be delayed; the direct
        # trajectory is generated here, so it is safe to make it immediately
        # available to the takeoff convergence gate and the first solve.
        self._on_direct_trajectory(message)

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
        position = self.measured_position if self.measured_position is not None else np.asarray(self.odometry.position, dtype=float)
        velocity = self.measured_velocity if self.measured_velocity is not None else np.asarray(self.odometry.velocity, dtype=float)
        quaternion = np.asarray(self.odometry.q, dtype=float)
        body_rate = np.asarray(self.odometry.angular_velocity, dtype=float)
        return np.r_[position, velocity, quaternion, body_rate]

    def _manual_control_ready(self, require_neutral: bool = False) -> bool:
        message = self.manual_control
        if (
            message is None
            or not message.valid
            or message.data_source != ManualControlSetpoint.SOURCE_RC
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
        self.controller.reset_warm_start()
        self.last_reference_segment = None
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
        if self.options.trajectory != "rc":
            return self.preset_source.sample_direct(time_s)
        if time_s < self.options.ascent:
            return self.preset_source.sample(time_s)
        assert self.rc_source is not None
        return self.rc_source.sample(time_s - self.trajectory_elapsed)

    def _total_duration(self) -> float:
        if self.options.trajectory == "rc":
            return self.options.ascent + self.options.rc_duration
        assert self.preset_source is not None
        return self.preset_source.direct_duration

    def _estimate_disturbance(
        self, state: np.ndarray, dt: float, timestamp_valid: bool = True
    ) -> np.ndarray:
        """Estimate world-frame acceleration disturbance from velocity."""
        if self.eso_enabled:
            if not timestamp_valid:
                self.eso.reset(state[3:6], self.disturbance_estimate)
                self.eso_active = False
                return self.disturbance_estimate.copy()
            if self.initial_position is not None and state[2] >= self.initial_position[2] - 0.2:
                self.eso.hold(state[3:6])
                self.eso_active = False
                return self.disturbance_estimate.copy()
            if (
                self.flight_started > 0.0
                and self.trajectory_elapsed < self.config.eso.activation_delay_s
            ):
                self.eso.hold(state[3:6])
                self.eso_active = False
                return self.disturbance_estimate.copy()
            if not self.eso_active:
                self.eso.reset(state[3:6], self.disturbance_estimate)
                self.eso_active = True
                return self.disturbance_estimate.copy()
            rotation = quaternion_to_rotation(state[6:10])
            modelled_acceleration = np.array([0.0, 0.0, self.config.model.gravity])
            modelled_acceleration -= (
                self.last_command_thrust / self.config.model.mass * rotation[:, 2]
            )
            self.disturbance_estimate = self.eso.update(
                state[3:6], modelled_acceleration, dt
            )
            return self.disturbance_estimate.copy()

        # Compatibility path for the pre-ESO baseline.
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

    def _model_consistent_truth(
        self, odometry_us: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Return (a_truth, a_model, d_truth, timestamp_age_ms) for validation.

        ``d_truth`` is the lumped disturbance seen by the *unchanged* ESO
        model: measured Gazebo acceleration minus the model acceleration built
        from the same commanded thrust and ground-truth attitude.  This is not
        fed back into the controller and therefore cannot change the model or
        its estimate.
        """
        local = self.groundtruth_local_position
        attitude = self.groundtruth_attitude
        if local is None or attitude is None:
            return None
        local_ts = int(getattr(local, "timestamp_sample", 0) or getattr(local, "timestamp", 0))
        attitude_ts = int(getattr(attitude, "timestamp_sample", 0) or getattr(attitude, "timestamp", 0))
        if local_ts <= 0 or attitude_ts <= 0:
            return None
        # Groundtruth and odometry are generated by the same PX4 bridge.  A
        # 100 ms gate avoids comparing a current estimate with stale truth
        # during DDS startup/recovery while tolerating normal callback skew.
        age_us = int(odometry_us) - local_ts
        if abs(age_us) > 100_000 or abs(local_ts - attitude_ts) > 100_000:
            return None
        acceleration = np.array([local.ax, local.ay, local.az], dtype=float)
        quaternion = np.asarray(attitude.q, dtype=float)
        if not np.all(np.isfinite(acceleration)) or not np.all(np.isfinite(quaternion)):
            return None
        rotation = quaternion_to_rotation(quaternion)
        modelled_acceleration = np.array([0.0, 0.0, self.config.model.gravity])
        modelled_acceleration -= (
            self.last_command_thrust / self.config.model.mass * rotation[:, 2]
        )
        residual = acceleration - modelled_acceleration
        return acceleration, modelled_acceleration, residual, age_us * 1.0e-3

    def _reference_from_trajectory(self, trajectory: KinematicTrajectory) -> Reference:
        assert self.reference_quaternion is not None
        limits = self.config.limits
        timing: dict[str, float] = {}
        motion_validation_started = monotonic()
        trajectory.validate_motion_limits(
            horizontal_speed_max=limits.horizontal_speed_max,
            vertical_speed_max_up=limits.vertical_speed_max_up,
            vertical_speed_max_down=limits.vertical_speed_max_down,
            horizontal_acceleration_max=limits.horizontal_acceleration_max,
            vertical_acceleration_max_up=limits.vertical_acceleration_max_up,
            vertical_acceleration_max_down=limits.vertical_acceleration_max_down,
            jerk_max=limits.jerk_max,
        )
        motion_validation_finished = monotonic()
        timing["motion_validation_ms"] = 1.0e3 * (
            motion_validation_finished - motion_validation_started
        )
        reference = build_reference_from_trajectory(
            trajectory,
            horizon_steps=self.config.controller.horizon_steps,
            sample_time=self.config.controller.sample_time,
            mass=self.config.model.mass,
            gravity=self.config.model.gravity,
            thrust_min=limits.thrust_min,
            thrust_max=limits.thrust_max,
            body_rate_max=limits.body_rate_max,
            quaternion_anchor=self.reference_quaternion,
            disturbance=self.disturbance_estimate,
            timing=timing,
        )
        self.last_reference_timing = timing
        return reference

    def _direct_reference(self) -> Reference:
        message = self.direct_trajectory
        if message is None:
            raise RuntimeError("complete NMPC trajectory has not been received")
        timeout = self.config.controller.reference_timeout
        if monotonic() - self.last_direct_trajectory_time > timeout:
            raise RuntimeError("complete NMPC trajectory is stale")
        # Keep the PX4-synchronised publication timestamp check as a second
        # guard in addition to local receive age.  The depth-one latest-wins
        # queue above prevents delayed DDS samples from reaching this check.
        timestamp_age = self.last_direct_trajectory_timestamp_age_s
        if timestamp_age is None or abs(timestamp_age) > timeout:
            raise RuntimeError("complete NMPC trajectory timestamp is outside the allowed age")
        points = int(message.points)
        expected_points = self.config.controller.horizon_steps + 1
        if points != expected_points:
            raise RuntimeError(
                f"complete trajectory has {points} points, expected {expected_points}"
            )
        trajectory = self._direct_trajectory_cache
        if trajectory is None:
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
        """Return the first point of the latest complete upper-computer trajectory."""
        message = self.direct_trajectory
        last_time = self.last_direct_trajectory_time
        if message is None or monotonic() - last_time > 0.3:
            return None
        points = int(message.points)
        if points < 1:
            return None
        position = np.asarray(message.position, dtype=float).reshape(-1, 3)[0]
        velocity = np.asarray(message.velocity, dtype=float).reshape(-1, 3)[0]
        acceleration = np.asarray(message.acceleration, dtype=float).reshape(-1, 3)[0]
        return position, velocity, acceleration

    def _reference(
        self, elapsed: float, trajectory: KinematicTrajectory | None = None
    ) -> Reference:
        """Build the NMPC reference without forcing a local DDS round-trip.

        ``trajectory`` is supplied by the in-process test/planner path.  When
        it is absent, the method retains the external
        ``NmpcTrajectorySetpoint`` subscription path for Orin or another
        planner publishing over DDS.
        """
        del elapsed  # kept in the API for external/reference-source symmetry
        if trajectory is not None:
            return self._reference_from_trajectory(trajectory)
        return self._direct_reference()

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
        command_publish_us: int,
        t_rx: float,
        t_state: float,
        t_eso: float,
        t_ref: float,
        t_set: float,
        t_pre_end: float,
        t_solve_0: float,
        t_solve_1: float,
        t_pub: float,
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
        model_truth = self._model_consistent_truth(odometry_us)
        message_age = monotonic() - self.last_direct_trajectory_time
        record: dict[str, object] = {
            # timestamp_us remains the canonical state sample timestamp for
            # compatibility with existing analysis scripts.
            "timestamp_us": odometry_us,
            "state_sample_timestamp_us": odometry_us,
            "state_publication_timestamp_us": int(self.odometry.timestamp),
            "state_sample_steady_timestamp_us": self.last_odometry_sample_steady_us,
            "state_receive_steady_timestamp_us": self.last_odometry_receive_steady_us,
            "command_publish_steady_timestamp_us": command_publish_us,
            # Raw monotonic event timestamps for end-to-end NMPC timing.
            # t_rx is captured in the VehicleOdometry ROS callback; the other
            # events are captured in the control callback for this solve.
            "t_rx": t_rx,
            "t_state": t_state,
            "t_eso": t_eso,
            "t_ref": t_ref,
            "t_set": t_set,
            "t_pre_end": t_pre_end,
            "t_solve_0": t_solve_0,
            "t_solve_1": t_solve_1,
            "t_pub": t_pub,
            "t_rx_steady_us": round(t_rx * 1.0e6),
            "t_state_steady_us": round(t_state * 1.0e6),
            "t_eso_steady_us": round(t_eso * 1.0e6),
            "t_ref_steady_us": round(t_ref * 1.0e6),
            "t_set_steady_us": round(t_set * 1.0e6),
            "t_pre_end_steady_us": round(t_pre_end * 1.0e6),
            "t_solve_0_steady_us": round(t_solve_0 * 1.0e6),
            "t_solve_1_steady_us": round(t_solve_1 * 1.0e6),
            "t_pub_steady_us": round(t_pub * 1.0e6),
            "tau_state_ms": 1.0e3 * (t_state - t_rx),
            "tau_eso_ms": 1.0e3 * (t_eso - t_state),
            "tau_ref_ms": 1.0e3 * (t_ref - t_eso),
            "tau_set_ms": 1.0e3 * (t_set - t_ref),
            "rx_to_pre_end_ms": 1.0e3 * (t_pre_end - t_rx),
            "pre_end_to_solve_0_ms": 1.0e3 * (t_solve_0 - t_pre_end),
            "solve_ms_from_timestamps": 1.0e3 * (t_solve_1 - t_solve_0),
            "solve_1_to_pub_ms": 1.0e3 * (t_pub - t_solve_1),
            "rx_to_pub_ms": 1.0e3 * (t_pub - t_rx),
            "sample_to_command_latency_ms": 1.0e-3
            * (command_publish_us - self.last_odometry_sample_steady_us),
            "trajectory_time_s": elapsed,
            "segment": target.segment,
            "reference_source": "direct",
            "metric_enabled": int(self._collect_tracking(target, elapsed)),
            "solve_time_ms": 1000.0 * self.controller.last_solve_time,
            "warm_start_used": int(getattr(self.controller, "warm_start_used", False)),
            "reference_age_ms": 1000.0 * message_age,
            "saturated": int(saturated),
            "thrust_command_N": command.thrust,
            "thrust_feedforward_N": feedforward[0],
            "position_error_norm_m": np.linalg.norm(position_error),
            "velocity_error_norm_m_s": np.linalg.norm(velocity_error),
            "attitude_error_norm_rad": np.linalg.norm(attitude_error),
            "body_rate_error_norm_rad_s": np.linalg.norm(rate_error),
            "cg_offset_x_m": self.options.cg_offset_x_m,
            "wind_velocity_m_s": [
                self.options.wind_velocity_x_m_s,
                self.options.wind_velocity_y_m_s,
                self.options.wind_velocity_z_m_s,
            ],
            # Validation-only fields.  NaN keeps the CSV schema stable when a
            # non-Gazebo vehicle (or a bridge during startup) has no truth.
            "groundtruth_timestamp_age_ms": float("nan"),
            "groundtruth_acceleration_x_m_s2": float("nan"),
            "groundtruth_acceleration_y_m_s2": float("nan"),
            "groundtruth_acceleration_z_m_s2": float("nan"),
            "model_acceleration_x_m_s2": float("nan"),
            "model_acceleration_y_m_s2": float("nan"),
            "model_acceleration_z_m_s2": float("nan"),
            "model_residual_x_m_s2": float("nan"),
            "model_residual_y_m_s2": float("nan"),
            "model_residual_z_m_s2": float("nan"),
            "eso_model_residual_error_x_m_s2": float("nan"),
            "eso_model_residual_error_y_m_s2": float("nan"),
            "eso_model_residual_error_z_m_s2": float("nan"),
        }
        if model_truth is not None:
            truth_acceleration, model_acceleration, residual, age_ms = model_truth
            record["groundtruth_timestamp_age_ms"] = float(age_ms)
            for axis, truth_value, model_value, residual_value, eso_value in zip(
                ("x", "y", "z"),
                truth_acceleration,
                model_acceleration,
                residual,
                disturbance if disturbance is not None else self.disturbance_estimate,
            ):
                record[f"groundtruth_acceleration_{axis}_m_s2"] = float(truth_value)
                record[f"model_acceleration_{axis}_m_s2"] = float(model_value)
                record[f"model_residual_{axis}_m_s2"] = float(residual_value)
                record[f"eso_model_residual_error_{axis}_m_s2"] = float(eso_value - residual_value)
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

    def _write_trajectory_plot(
        self, trajectory_path: Path, summary: dict[str, object] | None = None
    ) -> Path | None:
        """Save a compact actual-vs-reference plot beside trajectory.csv."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            self.get_logger().warning("trajectory plot skipped: matplotlib is unavailable: %s", error)
            return None

        try:
            with trajectory_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                self.get_logger().warning("trajectory plot skipped: no trajectory samples")
                return None

            time_s = np.asarray([float(row["trajectory_time_s"]) for row in rows])
            position = np.asarray(
                [[float(row[f"position_{axis}"]) for axis in "xyz"] for row in rows]
            )
            reference_position = np.asarray(
                [
                    [float(row[f"position_reference_{axis}"]) for axis in "xyz"]
                    for row in rows
                ]
            )
            velocity = np.asarray(
                [[float(row[f"velocity_{axis}"]) for axis in "xyz"] for row in rows]
            )
            reference_velocity = np.asarray(
                [
                    [float(row[f"velocity_reference_{axis}"]) for axis in "xyz"]
                    for row in rows
                ]
            )
            position_error = np.asarray(
                [float(row["position_error_norm_m"]) for row in rows]
            )
            horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
            reference_horizontal_speed = np.linalg.norm(reference_velocity[:, :2], axis=1)

            figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            axis_xy, axis_position, axis_speed, axis_error = axes.flat
            axis_xy.plot(position[:, 0], position[:, 1], label="actual")
            axis_xy.plot(
                reference_position[:, 0], reference_position[:, 1], "--", label="reference"
            )
            axis_xy.set_title("Horizontal trajectory")
            axis_xy.set_xlabel("North x [m]")
            axis_xy.set_ylabel("East y [m]")
            axis_xy.axis("equal")
            axis_xy.grid(True, alpha=0.3)
            axis_xy.legend()
            if summary is not None:
                axis_xy.text(
                    0.02,
                    0.98,
                    "RMSE: pos={:.4f} m | vel={:.4f} m/s | att={:.4f} rad".format(
                        float(summary.get("tracking_position_rmse_m", float("nan"))),
                        float(summary.get("velocity_rmse_m_s", float("nan"))),
                        float(summary.get("attitude_rmse_rad", float("nan"))),
                    ),
                    transform=axis_xy.transAxes,
                    va="top",
                    fontsize="small",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                )

            for index, label in enumerate(("x", "y", "z")):
                axis_position.plot(time_s, position[:, index], label=f"actual {label}")
                axis_position.plot(
                    time_s,
                    reference_position[:, index],
                    "--",
                    label=f"reference {label}",
                )
            axis_position.set_title("Position")
            axis_position.set_xlabel("Trajectory time [s]")
            axis_position.set_ylabel("Position [m]")
            axis_position.grid(True, alpha=0.3)
            axis_position.legend(ncol=2, fontsize="small")

            axis_speed.plot(time_s, horizontal_speed, label="actual")
            axis_speed.plot(time_s, reference_horizontal_speed, "--", label="reference")
            axis_speed.axhline(
                self.config.limits.horizontal_speed_max,
                color="tab:red",
                linestyle=":",
                label="limit",
            )
            axis_speed.set_title("Horizontal speed")
            axis_speed.set_xlabel("Trajectory time [s]")
            axis_speed.set_ylabel("Speed [m/s]")
            axis_speed.grid(True, alpha=0.3)
            axis_speed.legend()

            axis_error.plot(time_s, position_error, color="tab:purple")
            axis_error.set_title("Position tracking error")
            axis_error.set_xlabel("Trajectory time [s]")
            axis_error.set_ylabel("Error norm [m]")
            axis_error.grid(True, alpha=0.3)

            plot_path = trajectory_path.with_name("trajectory.png")
            figure.savefig(plot_path, dpi=150)
            plt.close(figure)
            return plot_path
        except Exception as error:  # plotting must not abort a completed flight
            self.get_logger().warning("trajectory plot skipped: %s", error)
            return None

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
        trajectory_plot = self._write_trajectory_plot(trajectory_path, summary)
        summary["trajectory_plot"] = (
            str(trajectory_plot.resolve()) if trajectory_plot is not None else None
        )
        summary_path = directory / "summary.json"
        summary["summary_log"] = str(summary_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def _finish(self) -> None:
        with self._finish_lock:
            if self.finished or self._finishing:
                return
            self._finishing = True
        self.get_logger().info(
            "finish begin solves=%d records=%d validation=%d"
            % (self.solve_count, len(self.trajectory_records),
               len(self.model_validation.samples) if self.model_validation is not None else 0)
        )
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
        warm_start_records = [
            row for row in self.trajectory_records if bool(row.get("warm_start_used", False))
        ]
        truth_records = [
            row for row in self.trajectory_records
            if np.isfinite(float(row.get("model_residual_x_m_s2", float("nan"))))
        ]
        sample_to_command_latency = np.asarray(
            [row["sample_to_command_latency_ms"] for row in self.trajectory_records],
            dtype=float,
        )

        def latency_percentile(percentile: float) -> float:
            return (
                float(np.percentile(sample_to_command_latency, percentile))
                if sample_to_command_latency.size
                else float("nan")
            )

        control_timing_keys = (
            "executor_wait_ms",
            "preparation_ms",
            "state_conversion_ms",
            "disturbance_estimation_ms",
            "reference_construction_ms",
            "reference_sample_ms",
            "reference_horizon_eval_ms",
            "reference_publish_pack_ms",
            "reference_parse_build_ms",
            "reference_motion_validation_ms",
            "reference_trajectory_validation_ms",
            "reference_sample_eval_ms",
            "reference_inverse_dynamics_ms",
            "reference_quaternion_alignment_ms",
            "reference_body_rate_feedforward_ms",
            "reference_limits_check_ms",
            "reference_attach_states_ms",
            "solver_wall_ms",
            "command_publish_ms",
            "recording_ms",
            "control_callback_total_ms",
        )
        control_timing: dict[str, dict[str, float | int]] = {}
        for key in control_timing_keys:
            values = np.asarray(
                [float(row[key]) for row in self.trajectory_records if key in row],
                dtype=float,
            )
            if values.size:
                control_timing[key] = {
                    "count": int(values.size),
                    "median": float(np.percentile(values, 50.0)),
                    "p95": float(np.percentile(values, 95.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "max": float(np.max(values)),
                }

        timestamp_timing_keys = (
            "tau_state_ms",
            "tau_eso_ms",
            "tau_ref_ms",
            "tau_set_ms",
            "rx_to_pre_end_ms",
            "pre_end_to_solve_0_ms",
            "solve_ms_from_timestamps",
            "solve_1_to_pub_ms",
            "rx_to_pub_ms",
        )
        timestamp_timing: dict[str, dict[str, float | int]] = {}
        for key in timestamp_timing_keys:
            values = np.asarray(
                [float(row[key]) for row in self.trajectory_records if key in row],
                dtype=float,
            )
            if values.size:
                timestamp_timing[key] = {
                    "count": int(values.size),
                    "median": float(np.percentile(values, 50.0)),
                    "p95": float(np.percentile(values, 95.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "max": float(np.max(values)),
                }

        acados_timing_keys = sorted(
            {
                key
                for row in self.trajectory_records
                for key in row
                if key.startswith("acados_") and key.endswith("_ms")
            }
        )
        acados_timing: dict[str, dict[str, float | int]] = {}
        for key in acados_timing_keys:
            values = np.asarray(
                [float(row[key]) for row in self.trajectory_records if key in row],
                dtype=float,
            )
            if values.size:
                acados_timing[key] = {
                    "count": int(values.size),
                    "median": float(np.percentile(values, 50.0)),
                    "p95": float(np.percentile(values, 95.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "max": float(np.max(values)),
                }

        acados_stat_keys = sorted(
            {
                key[len("acados_stat_") : -len("_s")]
                for row in self.trajectory_records
                for key in row
                if key.startswith("acados_stat_") and key.endswith("_s")
            }
        )
        acados_stats_summary: dict[str, dict[str, float | int]] = {}
        for key in acados_stat_keys:
            field = f"acados_stat_{key}_s"
            values = np.asarray(
                [float(row[field]) for row in self.trajectory_records if field in row],
                dtype=float,
            )
            if values.size:
                acados_stats_summary[key] = {
                    "count": int(values.size),
                    "median": float(np.percentile(values, 50.0)),
                    "p95": float(np.percentile(values, 95.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "max": float(np.max(values)),
                }

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
            "measurement_noise_model": {
                "type": "pure_random_walk",
                "injection_location": "PX4 GZBridge before EKF2",
            },
            "landed_disarmed": landed,
            "solve_count": self.solve_count,
            "warm_start": {
                "enabled": bool(self.config.controller.warm_start),
                "used_count": len(warm_start_records),
                "usage_fraction": (
                    len(warm_start_records) / self.solve_count if self.solve_count else 0.0
                ),
            },
            "solve_p99_ms": p99_ms,
            "solve_max_ms": max_ms,
            "odometry_timestamp_jump_count": self.timestamp_jump_count,
            "reference_timeout_watchdog": {
                "mode": "direct_upper_computer_trajectory",
                "clock": "CLOCK_MONOTONIC receive time",
                "recovery_window_s": self.config.controller.reference_timeout,
                "episode_count": 0,
                "recovered_count": 0,
                "fallback_solve_count": 0,
                # A completed direct run intentionally stops publishing before
                # disarm; do not report that normal shutdown interval as an
                # active reference fault.
                "active_at_finish": bool(
                    not self.success
                    and monotonic() - self.last_direct_trajectory_time
                    > self.config.controller.reference_timeout
                ),
            },
            "offboard_watchdog": {
                "heartbeat_period_s": self._offboard_heartbeat_period,
                "loss_grace_period_s": self.options.offboard_loss_grace_period,
                "transient_loss_event_count": self.offboard_loss_event_count,
                "loss_active_at_finish": self.offboard_loss_started != 0.0,
            },
            "time_alignment": {
                "state_interval_clock": "PX4 VehicleOdometry.timestamp_sample",
                "delay_clock": "CLOCK_MONOTONIC corrected by PX4 sample age",
                "timestamp_sample_age_invalid_count": (
                    self.timestamp_sample_age_invalid_count
                ),
                "sample_to_command_latency_ms": {
                    "count": int(sample_to_command_latency.size),
                    "median": latency_percentile(50.0),
                    "p95": latency_percentile(95.0),
                    "p99": latency_percentile(99.0),
                    "max": (
                        float(np.max(sample_to_command_latency))
                        if sample_to_command_latency.size
                        else float("nan")
                    ),
                },
            },
            "control_timing_ms": control_timing,
            "nmpc_timestamp_timing_ms": timestamp_timing,
            "acados_solver_stats_s": acados_stats_summary,
            "acados_timing_ms": acados_timing,
            "direct_trajectory_timing": {
                "freshness_clock": "CLOCK_MONOTONIC receive time",
                "last_timestamp_age_s": self.last_direct_trajectory_timestamp_age_s,
                "timestamp_age_gate": "validated",
            },
            "tracking_position_rmse_m": tracking_rmse,
            "tracking_position_max_m": tracking_max,
            "input_saturation_fraction": saturation_fraction,
            "reference_source": "direct",
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
        if truth_records:
            residual = np.asarray(
                [[float(row[f"model_residual_{axis}_m_s2"]) for axis in "xyz"]
                 for row in truth_records], dtype=float
            )
            eso_error = np.asarray(
                [[float(row[f"eso_model_residual_error_{axis}_m_s2"]) for axis in "xyz"]
                 for row in truth_records], dtype=float
            )
            eso_estimate = residual + eso_error
            summary["eso_model_consistent_validation"] = {
                "definition": (
                    "d_truth = a_groundtruth - (g*e3 - T_command/m * R_groundtruth[:,2]); "
                    "ESO is compared to this residual without changing the model"
                ),
                "sample_count": len(truth_records),
                "truth_residual_mean_m_s2": np.mean(residual, axis=0).tolist(),
                "truth_residual_rmse_m_s2": np.sqrt(np.mean(residual * residual, axis=0)).tolist(),
                "eso_mean_m_s2": np.mean(eso_estimate, axis=0).tolist(),
                "eso_error_bias_m_s2": np.mean(eso_error, axis=0).tolist(),
                "eso_error_rmse_m_s2": np.sqrt(np.mean(eso_error * eso_error, axis=0)).tolist(),
                "groundtruth_timestamp_age_ms": {
                    "mean": float(np.mean([float(row["groundtruth_timestamp_age_ms"]) for row in truth_records])),
                    "max_abs": float(np.max(np.abs([float(row["groundtruth_timestamp_age_ms"]) for row in truth_records]))),
                },
            }
        if self.model_validation is not None and len(self.model_validation.samples) >= 2:
            self.get_logger().info("finish model validation begin")
            summary["model_validation"] = self.model_validation.summary()
            self.get_logger().info("finish model validation done")
        self.get_logger().info("finish artifacts begin")
        self._write_run_artifacts(summary)
        self.get_logger().info("finish artifacts done")
        print("NMPC_SITL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
        self.finished = True
        self._finishing = False

    def _control_tick(self) -> None:
        now = monotonic()
        if self.finished:
            return
        # Shutdown must keep progressing even if PX4 stops publishing a
        # status/odometry sample immediately after the land or disarm command.
        # Previously the generic startup/stale-data guard returned early in
        # that situation, so the parent runner waited until its hard timeout.
        if self.phase == "DISARMING":
            if now - self.last_command_time >= 0.5:
                if self.land_detected is not None and self.land_detected.landed:
                    self._request_disarm()
                else:
                    self._publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_command_time = now
            if (
                self.status is not None
                and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self._finish()
            elif now - self.phase_started > 12.0:
                self.finish_reason += "; disarm confirmation timeout"
                self.safety_abort = True
                self._finish()
            return
        if self.odometry is None or self.status is None:
            # The uXRCE-DDS client may need several seconds to create the
            # complete PX4 topic graph after a SITL restart.  Keep this
            # startup grace period separate from the flight-time stale-data
            # guard below so discovery latency cannot abort a valid run.
            if now - self.phase_started > 30.0:
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
            # Keep publishing a fresh stationary complete horizon throughout
            # takeoff; the direct interface has no separate point-reference
            # or PX4 smoothing stage.
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
            # The rate controller alone can stop a CG-induced rotation but it
            # cannot return an already tilted vehicle to level. Add the small
            # attitude outer loop that the takeoff handover previously lacked.
            horizontal_acceleration = (
                -self.options.takeoff_horizontal_kp
                * (state[:2] - self.initial_position[:2])
                - self.options.takeoff_horizontal_kd * state[3:5]
            )
            acceleration_norm = float(np.linalg.norm(horizontal_acceleration))
            if acceleration_norm > self.options.takeoff_horizontal_acceleration_max:
                horizontal_acceleration *= (
                    self.options.takeoff_horizontal_acceleration_max / acceleration_norm
                )
            takeoff_attitude, _ = inverse_dynamics_attitude_and_thrust(
                np.r_[horizontal_acceleration, 0.0],
                self.reference_yaw,
                self.config.model.mass,
                self.config.model.gravity,
            )
            attitude_error = quaternion_attitude_error(
                state[6:10], takeoff_attitude
            )
            takeoff_rate = -np.array(
                [
                    self.options.takeoff_attitude_gain_rp,
                    self.options.takeoff_attitude_gain_rp,
                    self.options.takeoff_attitude_gain_yaw,
                ]
            ) * attitude_error
            takeoff_rate = np.clip(
                takeoff_rate,
                -self.config.limits.body_rate_max,
                self.config.limits.body_rate_max,
            )
            self._publish_rates(thrust, takeoff_rate)
            self.last_command_thrust = thrust
            body_z_down = 1.0 - 2.0 * (state[7] ** 2 + state[8] ** 2)
            tilt = float(np.arccos(np.clip(body_z_down, -1.0, 1.0)))
            horizontal_drift = float(np.linalg.norm(state[:2] - self.initial_position[:2]))
            takeoff_age = now - self.phase_started
            if (
                tilt > np.deg2rad(25.0)
                or horizontal_drift > self.options.takeoff_horizontal_drift_limit
                or takeoff_age > self.options.takeoff_timeout
            ):
                self._start_disarming(
                    "takeoff failed: "
                    f"tilt={np.rad2deg(tilt):.1f}deg, "
                    f"horizontal_drift={horizontal_drift:.3f}m, "
                    f"elapsed={takeoff_age:.2f}s",
                    safety_abort=True,
                )
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
                    and tilt < self.options.takeoff_converge_tilt_rad
                    and float(np.linalg.norm(state[10:13]))
                    < self.options.takeoff_converge_body_rate
                )
                if converged:
                    anchor = state[:3].copy()
                    self.flight_started = now
                    sample_timestamp_us = self._odometry_sample_timestamp_us()
                    self.flight_started_px4_us = sample_timestamp_us
                    self.last_solved_odometry_us = 0
                    self.last_trajectory_odometry_us = sample_timestamp_us
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
                if self.offboard_loss_started == 0.0:
                    self.offboard_loss_started = now
                    self.offboard_loss_event_count += 1
                    self.get_logger().warning(
                        "PX4 status left Offboard; waiting %.3fs for recovery"
                        % self.options.offboard_loss_grace_period
                    )
                elif now - self.offboard_loss_started >= self.options.offboard_loss_grace_period:
                    self._start_disarming(
                        "offboard mode lost during flight", safety_abort=True
                    )
                return
            if self.offboard_loss_started != 0.0:
                self.get_logger().info(
                    "PX4 Offboard status recovered after %.3fs"
                    % (now - self.offboard_loss_started)
                )
                self.offboard_loss_started = 0.0
            if self.options.trajectory == "rc" and not self._manual_control_ready():
                self._start_disarming("manual-control input lost or invalid", safety_abort=True)
                return
            control_started = monotonic()
            odometry_us = self._odometry_sample_timestamp_us()
            if odometry_us == self.last_solved_odometry_us:
                return
            self.last_solved_odometry_us = odometry_us
            # This is the receive instant captured by the ROS VehicleOdometry
            # callback, not the later timer wake-up time.
            # Select a receive timestamp at or before this control callback's
            # start.  A newer odometry callback may run concurrently while
            # this thread is waking; using the global "last" value would make
            # the measured executor wait negative.
            t_rx = self._odometry_receive_before(control_started)
            raw_step = (odometry_us - self.last_trajectory_odometry_us) * 1.0e-6
            # The elapsed reference clock advances at the output cadence.  The
            # NMPC prediction/reference discretization remains sample_time.
            nominal_step = self.config.controller.control_period
            maximum_step = max(0.1, 10.0 * nominal_step)
            timestamp_valid = not (raw_step < 0.0 or raw_step > maximum_step)
            if not timestamp_valid:
                self.timestamp_jump_count += 1
                self.controller.reset_warm_start()
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
            t_state_start = monotonic()
            state = self._state()
            t_state = monotonic()
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
            # ESO is the first stage after the measured state conversion.
            try:
                disturbance = self._estimate_disturbance(
                    state, trajectory_step, timestamp_valid=timestamp_valid
                )
                t_eso = monotonic()
                t_ref_start = monotonic()
                t_ref_sample_start = t_ref_start
                target = self._trajectory_sample(elapsed)
                t_ref_sample_end = monotonic()
                if target.segment != self.last_reference_segment:
                    if self.last_reference_segment is not None:
                        self.controller.reset_warm_start()
                    self.last_reference_segment = target.segment
                t_ref_horizon_start = monotonic()
                kinematic_horizon = self._kinematic_horizon(elapsed)
                t_ref_horizon_end = monotonic()
                t_ref_publish_start = monotonic()
                self._publish_direct_trajectory(kinematic_horizon)
                t_ref_publish_end = monotonic()
                t_ref_parse_start = monotonic()
                # The trajectory was generated in this process.  Feed that
                # in-memory horizon directly into NMPC; the DDS publication
                # above remains available for external consumers but is not
                # serialized and parsed again before solving.
                reference = self._reference(elapsed, kinematic_horizon)
                t_ref = monotonic()
                t_ref_parse_end = t_ref
                t_pre_end = t_ref
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
                command = self.controller.solve(state, reference, disturbance)
                solver_timing = self.controller.last_timing
                t_set = float(solver_timing["t_set_steady_s"])
                t_solve_0 = float(solver_timing["t_solve_0_steady_s"])
                t_solve_1 = float(solver_timing["t_solve_1_steady_s"])
                solve_finished = t_solve_1
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
            # Publish immediately after a successful solve, then associate the
            # exact command publication time with the earlier estimator sample.
            t_pub = monotonic()
            publish_started = t_pub
            command_publish_us = self._publish_rates(command.thrust, command.body_rate)
            publish_finished = monotonic()
            recording_started = monotonic()
            self._record_solve(
                odometry_us,
                command_publish_us,
                t_rx,
                t_state,
                t_eso,
                t_ref,
                t_set,
                t_pre_end,
                t_solve_0,
                t_solve_1,
                t_pub,
                elapsed,
                target,
                state,
                reference,
                command,
                saturated,
                disturbance,
            )
            recording_finished = monotonic()
            self.trajectory_records[-1].update(
                {
                    "t_control_start": control_started,
                    "t_pub_end": publish_finished,
                    "t_state_start": t_state_start,
                    "t_ref_start": t_ref_start,
                    "t_ref_sample_start": t_ref_sample_start,
                    "t_ref_sample_end": t_ref_sample_end,
                    "t_ref_horizon_start": t_ref_horizon_start,
                    "t_ref_horizon_end": t_ref_horizon_end,
                    "t_ref_publish_start": t_ref_publish_start,
                    "t_ref_publish_end": t_ref_publish_end,
                    "t_ref_parse_start": t_ref_parse_start,
                    "t_ref_parse_end": t_ref_parse_end,
                    "executor_wait_ms": 1.0e3
                    * (
                        control_started
                        - t_rx
                    ),
                    "preparation_ms": 1.0e3
                    * (t_state - control_started),
                    "state_conversion_ms": 1.0e3 * (t_state - t_state_start),
                    "disturbance_estimation_ms": 1.0e3
                    * (t_eso - t_state),
                    "reference_construction_ms": 1.0e3
                    * (t_ref - t_eso),
                    "reference_sample_ms": 1.0e3 * (t_ref_sample_end - t_ref_sample_start),
                    "reference_horizon_eval_ms": 1.0e3
                    * (t_ref_horizon_end - t_ref_horizon_start),
                    "reference_publish_pack_ms": 1.0e3
                    * (t_ref_publish_end - t_ref_publish_start),
                    "reference_parse_build_ms": 1.0e3
                    * (t_ref_parse_end - t_ref_parse_start),
                    "solver_wall_ms": 1.0e3
                    * (t_solve_1 - t_solve_0),
                    "command_publish_ms": 1.0e3
                    * (publish_finished - publish_started),
                    "recording_ms": 1.0e3
                    * (recording_finished - recording_started),
                    "control_callback_total_ms": 1.0e3
                    * (recording_finished - control_started),
                    **{
                        f"reference_{key}": float(value)
                        for key, value in self.last_reference_timing.items()
                    },
                }
            )
            self.trajectory_records[-1].update(
                {
                    f"acados_{key}": float(value)
                    for key, value in getattr(self.controller, "last_timing", {}).items()
                    if not key.endswith("_steady_s")
                }
            )
            self.trajectory_records[-1].update(
                {
                    f"acados_stat_{key}_s": float(value)
                    for key, value in getattr(self.controller, "last_solver_stats", {}).items()
                }
            )
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
                    control_timestamp_us=command_publish_us,
                    measurement_timestamp_us=self.last_odometry_sample_steady_us,
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



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nmpc.yaml"))
    parser.add_argument(
        "--trajectory", choices=("hover", "step", "circle", "figure8", "rc"), default="hover"
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
    parser.add_argument(
        "--offboard-loss-grace-period",
        type=float,
        default=TestOptions.offboard_loss_grace_period,
        help="seconds to wait for a transient PX4 Offboard status recovery",
    )
    parser.add_argument("--validate-model", action="store_true")
    parser.add_argument(
        "--direct-callback", action="store_true",
        help="experimental: run NMPC synchronously in the VehicleOdometry callback",
    )
    parser.add_argument(
        "--thrust-weight",
        type=float,
        help="temporary physical thrust-correction cost weight R_T for this run",
    )
    parser.add_argument(
        "--eso-bandwidth",
        type=float,
        help="temporary ESO bandwidth in rad/s for this run",
    )
    parser.add_argument(
        "--model-mass",
        type=float,
        help="temporary NMPC model mass in kg for model-mismatch tests",
    )
    parser.add_argument(
        "--disable-eso", action="store_true",
        help="disable ESO for an A/B comparison run",
    )
    parser.add_argument(
        "--disable-warm-start", action="store_true",
        help="disable shifted Acados x/u/pi initialization for an A/B comparison run",
    )
    parser.add_argument("--cg-offset-x-m", type=float, default=0.0,
                        help="forward CG offset metadata; physical torque injection pending")
    parser.add_argument("--wind-velocity-x-m-s", type=float, default=0.0,
                        help="declared Gazebo wind velocity X (m/s); world is configured separately")
    parser.add_argument("--wind-velocity-y-m-s", type=float, default=0.0)
    parser.add_argument("--wind-velocity-z-m-s", type=float, default=0.0)
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
        arguments.offboard_loss_grace_period,
    )
    if any(value <= 0.0 for value in positive_arguments):
        parser.error("altitude, radius, speed, RC duration and geofence must be positive")
    config = load_config(arguments.config)
    if arguments.thrust_weight is not None:
        if arguments.thrust_weight <= 0.0 or not np.isfinite(arguments.thrust_weight):
            parser.error("thrust-weight must be finite and positive")
        thrust_scale = np.sqrt(
            config.cost_scales.weight_factor / arguments.thrust_weight
        )
        config = replace(
            config,
            cost_scales=replace(
                config.cost_scales, thrust_correction_n=float(thrust_scale)
            ),
        )
    if arguments.eso_bandwidth is not None:
        if arguments.eso_bandwidth <= 0.0 or not np.isfinite(arguments.eso_bandwidth):
            parser.error("eso-bandwidth must be finite and positive")
        config = replace(
            config,
            eso=replace(config.eso, bandwidth_rad_s=float(arguments.eso_bandwidth)),
        )
    if arguments.model_mass is not None:
        if arguments.model_mass <= 0.0 or not np.isfinite(arguments.model_mass):
            parser.error("model-mass must be finite and positive")
        config = replace(
            config,
            model=replace(config.model, mass=float(arguments.model_mass)),
        )
    if arguments.disable_eso:
        config = replace(config, eso=replace(config.eso, enabled=False))
    if arguments.disable_warm_start:
        config = replace(
            config,
            controller=replace(config.controller, warm_start=False),
        )
    options = TestOptions(
        trajectory=arguments.trajectory,
        altitude=arguments.altitude,
        radius=arguments.radius,
        speed=arguments.speed,
        rc_duration=arguments.rc_duration,
        rc_geofence_radius=arguments.rc_geofence_radius,
        offboard_loss_grace_period=arguments.offboard_loss_grace_period,
        validate_model=arguments.validate_model,
        direct_callback=arguments.direct_callback,
        log_directory=arguments.log_directory,
        hold=arguments.step_dwell if arguments.trajectory == "step" else TestOptions.hold,
        cg_offset_x_m=arguments.cg_offset_x_m,
        wind_velocity_x_m_s=arguments.wind_velocity_x_m_s,
        wind_velocity_y_m_s=arguments.wind_velocity_y_m_s,
        wind_velocity_z_m_s=arguments.wind_velocity_z_m_s,
    )
    rclpy.init()
    node = Px4NmpcHover(config, options)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
            # PX4 can stop sending status/odometry immediately after landing.
            # If no new sample arrives, the event-driven controller cannot
            # wake itself; bound the disarming phase from the ROS spin-loop.
            if (
                node.phase == "DISARMING"
                and monotonic() - node.phase_started > 12.0
            ):
                node.finish_reason += "; disarm confirmation timeout"
                node.safety_abort = True
                node._finish()
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
