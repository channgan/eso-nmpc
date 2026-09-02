# ROS1 C++ ESO-NMPC 适配层

这是原生 ROS1 `roscpp` 版本，控制热路径为：

```text
/mavros/local_position/odom
  -> ENU/FLU 转 PX4 NED/FRD
  -> 完整轨迹参考与逆动力学前馈
  -> ARM64 生成的 acados SQP-RTI solver
  -> /mavros/setpoint_raw/attitude
```

发生推进系统故障后，真机配置已恢复为 `shadow_mode:=true`、
`control_enabled_at_start:=false`，并保持 `auto_manage_flight:=false`。在完成坐标系、
推力映射、PX4 ULog分析和无桨/测功台复验前，不得开放控制输出。

非影子模式下还会检查 `/mavros/estimator_status`；姿态、水平/垂直速度及水平/垂直
位置没有全部有效时，节点硬门控 setpoint 和自动飞行状态机。

关闭影子模式后，节点默认只负责控制输出；将 `auto_manage_flight:=true` 后，同一节点还会订阅
`/mavros/state`，调用 `/mavros/set_mode`、`/mavros/cmd/arming`，并在禁用、超时
或配置的飞行时长结束时请求 `AUTO.LAND`。关闭该参数时则由外部飞行管理器接管。
节点持续发布 `mavros_msgs/AttitudeTarget`，作为 Offboard 的 body-rate/thrust setpoint。

## 构建

先在本机生成匹配架构的 acados solver：

```bash
cd /home/ljt/eso_nmpc_ros1_migration
source .venv/bin/activate
source /opt/ros/noetic/setup.bash
export ACADOS_SOURCE_DIR=/home/ljt/acados
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
python solver/generate_solver.py \
  --sample-time 0.01 --control-period 0.01 \
  --model-name quadrotor_nmpc_ros1_100hz \
  --code-export-directory generated/quadrotor_nmpc_ros1_100hz \
  --json-file quadrotor_nmpc_ros1_100hz.json
```

再构建 catkin workspace：

```bash
cd .catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release \
  -DACADOS_SOURCE_DIR=/home/ljt/acados \
  -DESO_NMPC_ROOT=/home/ljt/eso_nmpc_ros1_migration
source devel/setup.bash
```

## 运行与日志

```bash
roslaunch eso_nmpc_cpp eso_nmpc_cpp.launch
```

NMPC 由 `/mavros/local_position/odom` 回调触发。节点连接飞控后默认通过
`/mavros/set_message_interval` 请求 MAVLink ID 31（姿态）、105（原始 IMU）和
32（融合局部位置）均为 100 Hz；其中 ID 32 的实际到达频率决定 NMPC 的触发上限。
生成求解器与节点的 `sample_time` 均为 0.01 s。

以上默认命令已退出影子模式并启动 NMPC 预发送，但自动飞行管理仍关闭。

需要由本节点负责飞行状态机时显式打开（会请求 OFFBOARD、解锁和结束后的
`AUTO.LAND`）：

```bash
roslaunch eso_nmpc_cpp eso_nmpc_cpp.launch auto_manage_flight:=true
```

RC-NMPC 通过 `/mavros/manual_control/control` 获取摇杆、通过 `/mavros/rc/in`
读取 AUX；真机默认使用第 6 通道，AUX 高于阈值才切换到 RC 参考源。

默认情况下，节点首次收到里程计后建立当前位姿的悬停参考；外部规划器可以向
`/nmpc/in/trajectory_setpoint` 发布 `NmpcTrajectorySetpoint`，数组按点展开，
位置/速度/加速度使用 NED，航向为 NED yaw。每次启动默认写入：

```text
/home/ljt/nmpc_log/YYYYMMDD_HHMMSS_mmm/nmpc_timing.csv
```

完整轨迹接口和 ROS1 消息定义位于 `msg/NmpcTrajectorySetpoint.msg`。

当前 ROS1 C++ 适配已在单节点内接入 ESO、RC-NMPC 参考源以及 PX4 模式/解锁/降落
状态机（自动状态机默认关闭，需显式打开）；
尚未宣称完成真实 PX4 飞行验收。
