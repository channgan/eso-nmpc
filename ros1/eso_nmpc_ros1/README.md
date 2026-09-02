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

## 虚拟环境

可以使用 venv，但必须先加载系统 ROS。`rospy`、`mavros_msgs`、`nav_msgs`
等 ROS 1 模块不能从普通 PyPI venv 安装；建议使用与 ROS Noetic 相同的
Python 小版本创建带系统包的虚拟环境：

```bash
source /opt/ros/noetic/setup.bash
python3 -m venv --system-site-packages ~/venvs/eso-nmpc-ros1
source ~/venvs/eso-nmpc-ros1/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy PyYAML casadi==3.7.2
```

如果 acados 使用本地 Python 接口，还要安装其 `acados_template`；ROS 1
节点使用的解释器必须与 `/opt/ros/noetic` 的 Python 版本一致。启动前继续
保留 ROS 环境，并把仓库加入 Python 搜索路径：

```bash
source /opt/ros/noetic/setup.bash
source ~/venvs/eso-nmpc-ros1/bin/activate
export PYTHONPATH=/path/to/eso_nmpc:$PYTHONPATH
```

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
