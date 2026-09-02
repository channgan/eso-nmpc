# ESO-NMPC ROS 1 适配层

这是 `ros1` 分支的第一阶段适配：保留 `nmpc/` 数学核心，新增 ROS 1
catkin 包，通过 MAVROS 接入 PX4。ROS 1 不直接依赖 ROS 2 的 `rclpy`、
`px4_msgs` 或 DDS 话题。

## 接口映射

| 功能 | ROS 1/MAVROS 接口 |
|---|---|
| 状态 | `/mavros/local_position/odom` (`nav_msgs/Odometry`) |
| 飞控状态 | `/mavros/state` (`mavros_msgs/State`) |
| 控制输出 | `/mavros/setpoint_raw/attitude` (`mavros_msgs/AttitudeTarget`) |
| Offboard 模式 | `/mavros/set_mode` |
| 解锁/上锁 | `/mavros/cmd/arming` |

MAVROS 的局部位姿是 ENU/FLU，NMPC 内部仍使用 PX4 NED/FRD。转换集中在
`src/eso_nmpc_ros1/frames.py`，避免把坐标转换散落在控制器中。

## 构建

本机当前只有 ROS 2 Humble，不能在 WSL 当前环境直接编译此包。目标 ROS 1
机器需要 Ubuntu 20.04/ROS Noetic、MAVROS、Python 3、NumPy、PyYAML 和
与目标架构匹配的 acados 运行库。

```bash
cd ~/catkin_ws/src
ln -s /path/to/eso_nmpc/ros1/eso_nmpc_ros1 eso_nmpc_ros1
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

启动前设置核心仓库和 acados：

```bash
export PYTHONPATH=/path/to/eso_nmpc:$PYTHONPATH
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:/path/to/eso_nmpc/generated/quadrotor_nmpc:$LD_LIBRARY_PATH"
roslaunch eso_nmpc_ros1 eso_nmpc.launch \
  timing_log:=/data/flight_logs/ros1_nmpc_timing.csv
```

该第一阶段节点验证的是 ROS 1/MAVROS 的 hover 闭环和 `rx_to_pub` 记录；
完整四项回归、ESO 扰动矩阵和 C++ ROS 1 节点将在接口确认后继续迁移，
不会把 ROS 1 的结果混入 `sitl_regression_cpp`。
