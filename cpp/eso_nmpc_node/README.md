# eso_nmpc_node

This package is the C++ migration path for the controller. The internal hot
path is a direct function call chain in one process:

```text
VehicleOdometry callback
  -> velocity LESO
  -> complete NmpcTrajectorySetpoint reference conversion
  -> generated acados SQP-RTI solver
  -> VehicleRatesSetpoint publish
```

`OffboardControlMode` is published by a separate wall timer and also after a
successful solve. It is deliberately not routed through an internal ROS topic.
The complete trajectory input remains `/nmpc/in/trajectory_setpoint`; PX4 only
receives the final body-rate/thrust command.

The Python launch/supervision scripts may start PX4, Gazebo, the DDS agent and
the flight manager, but the NMPC process itself remains the continuously
running C++ node for the whole flight session.  Arming belongs to the initial
PX4 takeoff phase; an airborne NMPC handoff does not unlock again.

RC reference selection is local to this node.  When
`/fmu/out/manual_control_setpoint` is valid and the configured AUX channel is
high, the node maps the sticks to acceleration-limited RC-NMPC references and
constructs the complete horizon in memory.  The AUX channel is a runtime
parameter (`rc_aux_channel`); value `0` disables RC-NMPC selection.  With AUX low it uses the external
`NmpcTrajectorySetpoint` source.  The sources are mutually exclusive.  RC
timeout after RC-NMPC has been selected latches the NMPC output off and
publishes `/nmpc/rc_timeout=true`.  The external flight manager must disable
NMPC and request PX4 AUTO.LOITER/Position Hold (or the configured vehicle failsafe); the
node does not automatically re-enter Offboard or NMPC.  An explicit
`/nmpc/control_enabled=true` is required for recovery after a new manual
safety check.  Arming is never performed by this node.

The node also guards the PX4 odometry time chain.  The safety threshold uses
the local monotonic time between actual ROS2 odometry callbacks.  If that
receive gap exceeds `odometry_timestamp_gap_threshold` (default `0.10 s`), it
drops the sample, resets transient warm-start/ESO state, latches control off,
and publishes `/nmpc/odometry_timestamp_fault=true`.  A PX4 `timestamp` that
goes backward or repeats is treated as an out-of-order DDS sample: that sample
is dropped and logged, but it does not by itself trigger the flight fallback.
The external ROS2 flight manager must request Position/Hold and perform the
normal manual re-entry sequence before publishing
`/nmpc/control_enabled=true` again.  A real receive gap is a failed regression
condition even when the solver itself reports success, because the state
transition is no longer a valid continuous-time control sample.  The PX4
`timestamp_sample` field is retained for sample-age diagnostics and is not used
as the ROS2 continuity clock.

The default generated artifact is the current 13-state, 4-input, 7-parameter,
30-stage solver (`1c2d851e`, 0.01 s discretization). Build from the PX4 ROS 2 workspace after sourcing
the Humble and `px4_msgs` environments:

```bash
colcon build --packages-select eso_nmpc_node --cmake-args \
  -DESO_NMPC_ROOT=/path/to/eso_nmpc \
  -DACADOS_SOURCE_DIR=/path/to/acados
```

The generated solver hash is checked against the symbols compiled into this
node (`1c2d851e`); a new Acados formulation requires updating the generated
header/symbol names in the CMake configuration before rebuilding.  The repository
does not ship architecture-specific generated `.so` files; on a new machine
run `python3 solver/generate_solver.py` from the repository root first, then
build the C++ package.  This regenerates the solver natively for ARM64.

## Continuous flight logging

Set `flight_log_root` to a writable directory on the companion computer before
a flight.  Each node start creates a timestamped subdirectory below it.  The
odometry callback only copies a fixed-size record into a bounded queue; a
background thread writes the CSV files and flushes them periodically, so file
I/O is not on the NMPC real-time path.

```yaml
flight_log_root: "/data/flight_logs"
flight_log_path: ""
timing_log_path: ""
flight_log_buffer_size: 4096
flight_log_flush_period_ms: 250
odometry_timestamp_gap_threshold: 0.10
```

The files are written as `/data/flight_logs/YYYYMMDD_HHMMSS_mmm/nmpc_flight.csv`
and `nmpc_timing.csv`.  `nmpc_flight.csv` contains the PX4 timestamps, measured 13-state, current
reference, feed-forward, command, ESO disturbance estimate, control-source
flags, all stage timestamps, latency values and Acados timing/status fields.
Failed solves are recorded with `solve_success=0` and NaN commands.  The
bounded queue reports any dropped samples in `logger_dropped_samples`.

The timing CSV remains available as a smaller file for latency analysis.  PX4
must also record its normal `.ulg`; the two logs are joined after flight using
the PX4 timestamp domain, while the local steady-clock columns are used only
for host-side latency measurements.

After copying a flight directory to the analysis computer, generate the
trajectory and timing figures with:

```bash
python3 integration/plot_cpp_run.py /path/to/flight_directory
```

The plotting script accepts both the current `nmpc_flight.csv` format and the
older SITL `trajectory.csv` format.  In a regression directory containing both
files, it automatically uses the supervisor's complete `trajectory.csv` so
the figure and regression RMSE use the same data.  Use
`--source nmpc_flight` explicitly when inspecting the C++ node's per-solve
diagnostic log.
