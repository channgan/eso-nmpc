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

如果不指定 `timing_log`，节点会自动在 `/home/ljt/nmpc_log/YYYYMMDD_HHMMSS/`
下写入 `nmpc_timing.csv`。日志根目录可通过 `log_root:=...` 修改；显式
`timing_log` 的优先级高于自动路径。

节点支持与基线一致的四种预设轨迹：`hover`、`step`、`circle`、`figure8`。
例如：

```bash
roslaunch eso_nmpc_ros1 eso_nmpc.launch trajectory:=circle \
  timing_log:=/data/flight_logs/ros1_circle/nmpc_timing.csv
```

每个新的 MAVROS 里程计样本只触发一次求解；CSV 同时记录 `rx_to_pub_ms`、
`solve_ms`、NED 位置/参考和跟踪误差。里程计超时、求解失败、进入 Offboard
超时或解锁超时都会停止任务；飞行中求解失败会请求 `AUTO.LAND`。

本目录保留 Python 适配层，便于接口回归和坐标转换单测。正式部署的原生 ROS1
C++ 节点位于同级目录 `ros1/eso_nmpc_cpp`，使用 `roscpp + MAVROS + acados
C solver`，日志和完整轨迹消息接口保持一致。ESO 尚未接入该 ROS1 C++ 热路径；
先完成无扰动 hover 和四项轨迹验证，再接入观测器。

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
