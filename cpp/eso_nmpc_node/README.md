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

RC reference selection is local to this node.  When
`/fmu/out/manual_control_setpoint` is valid and the configured AUX channel is
high, the node maps the sticks to acceleration-limited RC-NMPC references and
constructs the complete horizon in memory.  The AUX channel is a runtime
parameter (`rc_aux_channel`); value `0` disables RC-NMPC selection.  With AUX low it uses the external
`NmpcTrajectorySetpoint` source.  The sources are mutually exclusive.  RC
timeout zeros the stick command and holds the current reference; it does not
continue the last non-zero command.  The flight manager must still explicitly
enable NMPC through `/nmpc/control_enabled` and handle the PX4 Offboard mode
switch; arming is never performed by this node.

The default generated artifact is the current 13-state, 4-input, 7-parameter,
30-stage solver (`e2d8d978`). Build from the PX4 ROS 2 workspace after sourcing
the Humble and `px4_msgs` environments:

```bash
colcon build --packages-select eso_nmpc_node --cmake-args \
  -DESO_NMPC_ROOT=/path/to/eso_nmpc \
  -DACADOS_SOURCE_DIR=/path/to/acados
```

The generated solver hash is checked against the symbols compiled into this
node (`e2d8d978`); a new Acados formulation requires updating the generated
header/symbol names in the C++ source before rebuilding.  The repository
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
```

The files are written as `/data/flight_logs/YYYYMMDD_HHMMSS_mmm/nmpc_flight.csv`
and `nmpc_timing.csv`.  `nmpc_flight.csv` contains the PX4 timestamps, measured 13-state, current
reference, feed-forward, command, ESO disturbance estimate, control-source
flags, all stage timestamps, latency values and Acados timing/status fields.
Failed solves are recorded with `solve_success=0` and NaN commands.  The
bounded queue reports any dropped samples in `logger_dropped_samples`.

The solver first uses its warm start.  If that solve fails, it immediately
retries once from a cold start.  A second failure in the same callback holds
the last valid command for at most 50 ms (when available); after three
consecutive failed callbacks the node disables NMPC until
`/nmpc/control_enabled` is set true again.

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
older SITL `trajectory.csv` format.
