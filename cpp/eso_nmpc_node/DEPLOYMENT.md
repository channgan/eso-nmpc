# Orin deployment checklist

This package is compiled on the target companion computer.  Do not copy the
WSL executable or generated x86 shared libraries to an ARM64 Orin.

## 1. Install the target dependencies

For a fresh Ubuntu 22.04/ROS 2 Humble Orin, the repository includes a single
installer. It installs the system and ROS packages, builds ARM64
Micro-XRCE-DDS-Agent and acados, regenerates the solver on the target, links
the package into the PX4 ROS workspace, compiles it, and runs the read-only
environment check:

```bash
cd /path/to/eso_nmpc
./tools/install_orin_dependencies.sh \
  --acados "$HOME/acados" \
  --agent "$HOME/Micro-XRCE-DDS-Agent" \
  --px4-ws "$HOME/px4_ros2_ws"
```

The script is safe to rerun: existing checkouts and generated files are
reused. It must be run as a normal user with `sudo` available. The ROS apt
repository and the PX4 workspace/`px4_msgs` checkout still need to exist; the
script does not guess or replace a PX4 branch. Use `--skip-build` when only
installing/checking dependencies, or `--skip-acados`/`--skip-agent` when those
components are already provisioned.

On Ubuntu 22.04 with ROS 2 Humble:

```bash
sudo apt install build-essential cmake ninja-build git \
  python3-dev python3-pip python3-venv python3-colcon-common-extensions \
  libeigen3-dev ros-humble-ros-base ros-humble-rclcpp \
  ros-humble-std-msgs ros-humble-rmw-cyclonedds-cpp
```

Install or build `Micro-XRCE-DDS-Agent` for ARM64.  Build acados with shared
libraries on the Orin and export `ACADOS_SOURCE_DIR` to that checkout.

## 2. Copy the source and match PX4 messages

Copy the complete repository (including `nmpc/`, `solver/` and
`cpp/eso_nmpc_node/`) to the Orin.  The `generated/` directory may be copied as
source/reference, but never copy a prebuilt `.so` from WSL; regenerate it on
the Orin in the next step.  Put the C++ package in the ROS workspace:

```bash
cp -a /path/to/eso_nmpc/cpp/eso_nmpc_node ~/px4_ros2_ws/src/eso_nmpc_node
```

The `px4_msgs` checkout in the same workspace must be the exact branch used by
the PX4 firmware and must contain the custom `NmpcTrajectorySetpoint` message.

## 3. Generate the solver natively

The generated solver is architecture-specific and is intentionally not a
portable prebuilt artifact.  From the repository root:

```bash
source /opt/ros/humble/setup.bash
export ACADOS_SOURCE_DIR=/path/to/acados
python3 -m venv .venv
source .venv/bin/activate
pip install numpy PyYAML casadi==3.7.2
pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
python3 solver/generate_solver.py
```

The current C++ node expects the generated `e2d8d978` symbols.  If the
formulation changes, update the generated symbol names in the C++ source and
the CMake hash together.

## 4. Build on the Orin

```bash
cd ~/px4_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select eso_nmpc_node --cmake-clean-cache \
  --cmake-args -DESO_NMPC_ROOT=/path/to/eso_nmpc \
  -DACADOS_SOURCE_DIR=/path/to/acados
```

Run the read-only dependency check before starting the node:

```bash
python3 /path/to/eso_nmpc/cpp/eso_nmpc_node/check_environment.py \
  --root /path/to/eso_nmpc \
  --acados /path/to/acados \
  --target-aarch64 \
  --require-generator \
  --executable ~/px4_ros2_ws/install/eso_nmpc_node/lib/eso_nmpc_node/eso_nmpc_node
```

It returns a non-zero status for missing required commands, ROS packages,
custom PX4 messages, acados libraries, generated solver files, or unresolved
runtime libraries.  Warnings (for example an unset CycloneDDS environment on
a build-only host) are reported separately.

## 5. Run and verify the runtime environment

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:/path/to/eso_nmpc/generated/quadrotor_nmpc:$LD_LIBRARY_PATH"
source ~/px4_ros2_ws/install/setup.bash
MicroXRCEAgent udp4 -p 8888
```

Start the node with a persistent writable log root.  A timestamped directory
is created for each node/flight session:

```bash
ros2 run eso_nmpc_node eso_nmpc_node --ros-args \
  --params-file ~/px4_ros2_ws/install/eso_nmpc_node/share/eso_nmpc_node/config/eso_nmpc_cpp.yaml \
  -p flight_log_root:=/data/flight_logs
```

Before arming, verify that `VehicleOdometry` arrives, the complete trajectory
topic is valid, `OffboardControlMode.body_rate` is true, and both CSV files are
growing.  Save the PX4 `.ulg` alongside the CSV files after every flight.
