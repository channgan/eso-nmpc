# ESO-NMPC ROS 1 适配层

这是 ROS 1 Noetic 适配：保留 `nmpc/` 数学核心，新增 ROS 1
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

MAVROS 的局部位姿是 ENU/FLU，且 `Odometry.twist` 位于子坐标系（机体系
FLU）；NMPC 内部仍使用 PX4 NED/FRD。位置、机体系速度、角速度和姿态的
完整转换集中在
`src/eso_nmpc_ros1/frames.py`，避免把坐标转换散落在控制器中。

## 构建

本机当前只有 ROS 2 Humble，不能在 WSL 当前环境直接编译此包。目标 ROS 1
机器需要 Ubuntu 20.04/ROS Noetic、MAVROS、C++ 编译器和与目标架构匹配的
acados 运行库；控制器本身不再依赖 Python。

## ROS1 + PX4 联合仿真

当前工作区将 PX4 与 NMPC 分成两个根目录：`/home/cy2/apx` 和
`/home/cy2/nmpc`。先初始化 PX4 的 Gazebo Classic 子模块，并把两个仓库加入
ROS 包搜索路径：

```bash
cd /home/cy2/apx
git submodule update --init Tools/simulation/gazebo-classic/sitl_gazebo-classic
export PX4_ROOT=/home/cy2/apx
export ESO_NMPC_ROOT=/home/cy2/nmpc
export ROS_PACKAGE_PATH="$PX4_ROOT/Tools/simulation/gazebo-classic/sitl_gazebo-classic:$PX4_ROOT:${ROS_PACKAGE_PATH:-}"
```

在 ROS Noetic catkin 工作区中链接本适配包并编译：

```bash
mkdir -p ~/ros1_ws/src
ln -sfn "$ESO_NMPC_ROOT/ros1/eso_nmpc_ros1" ~/ros1_ws/src/eso_nmpc_ros1
cd ~/ros1_ws
catkin_make
source devel/setup.bash
```

生成 acados solver 并完成 catkin 编译后，可由一个 launch 同时启动
Gazebo Classic、PX4 SITL、MAVROS 和原生 ROS1 C++ NMPC 节点：

```bash
roslaunch eso_nmpc_ros1 px4_ros1_sitl_nmpc.launch \
  gui:=false
```

当前 C++ 节点使用已按 Gazebo Classic Iris 对齐的配置（1.535 kg、28.2656 N
总推力上限、100 Hz 网格）；控制器要求独立规划器持续发布完整 horizon：

```bash
roslaunch eso_nmpc_ros1 px4_ros1_sitl_nmpc.launch \
  gui:=false
```

该链路使用 `/mavros/local_position/odom` 输入和
`/mavros/setpoint_raw/attitude` 输出；MAVROS 的 ENU/FLU 会在适配层转换为
NMPC 所需的 NED/FRD。若 PX4、MAVROS 或 Gazebo 尚未安装，launch 会保留为
可复现入口，但必须先按 ROS Noetic/MAVROS 安装说明补齐系统依赖。

启动前设置核心仓库和 acados：

```bash
export PYTHONPATH=/path/to/eso_nmpc:$PYTHONPATH
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:/path/to/eso_nmpc/generated/quadrotor_nmpc:$LD_LIBRARY_PATH"
roslaunch eso_nmpc_ros1 eso_nmpc.launch \
  timing_log:=/data/flight_logs/ros1_nmpc_timing.csv
```

如果不指定 `timing_log`，节点会自动在 `/home/cy2/nmpc_log/YYYYMMDD_HHMMSS/`
下写入 `nmpc_timing.csv`。日志根目录可通过 `log_root:=...` 修改；显式
`timing_log` 的优先级高于自动路径。

ROS1 C++ 控制器不再内置 Python 轨迹控制器；它通过
`/nmpc/in/trajectory_setpoint` 接收完整 NED horizon。四项回归的轨迹发布器
必须作为独立规划器运行，不能把规划器误认为控制器。RC 注入示例：

```bash
roslaunch eso_nmpc_cpp px4_ros1_rc_nmpc.launch gui:=false paused:=false
```

每个新的 MAVROS 里程计样本只触发一次求解；CSV 同时记录 `rx_to_pub_ms`、
`solve_ms`、NED 位置/参考和跟踪误差。里程计超时、求解失败、进入 Offboard
超时或解锁超时都会停止任务；飞行中求解失败会请求 `AUTO.LAND`。

### 与 ROS2 10 ms 基线对齐

ROS1 原生 C++ 节点和 ROS2 C++ 节点都按每个里程计样本触发一次 latest-wins
求解，并使用完整 `KinematicTrajectory`/`build_reference_from_trajectory`
接口；四项测试的默认时序为 ascent=4 s、hold=6 s、transition=3 s、descent=4 s、
settle=1.5 s。ROS2 基线运行在 Gazebo `gz_x500`
（质量约 2.0643 kg），本包默认运行 Gazebo Classic `iris`（1.535 kg），两者
不是同一物理对象；要比较位置 RMSE，必须先统一机体模型，不能只把代码或频率
改成一样。

当前 parity 回归使用 `ros1/eso_nmpc_cpp` 原生 C++ 节点，与 ROS2 C++ 节点
共享同一 acados 算法结构、ESO 状态机和完整 horizon 接口；具体质量和推力参数
仍必须按各自仿真机体配置。旧 Python 控制器已移除，不再参与构建、启动或结果比较。

## 隔离开发工作区

迁移开发建议使用独立 worktree 和保留 ROS 系统包的虚拟环境：

```bash
git worktree add ../eso_nmpc_ros1_migration -b ros1-migration ros1
cd ../eso_nmpc_ros1_migration
python3 -m venv --system-site-packages .venv
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```
