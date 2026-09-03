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

节点会订阅 `/mavros/estimator_status` 供诊断和飞行状态机使用；控制计算与
setpoint 发布不再额外依赖该状态门控，和 ROS2 C++ 节点保持一致。

关闭影子模式后，节点默认只负责控制输出；将 `auto_manage_flight:=true` 后，同一节点还会订阅
`/mavros/state`，调用 `/mavros/set_mode`、`/mavros/cmd/arming`，并在禁用、超时
或配置的飞行时长结束时请求 `AUTO.LAND`。关闭该参数时则由外部飞行管理器接管。
节点持续发布 `mavros_msgs/AttitudeTarget`，作为 Offboard 的 body-rate/thrust setpoint。

## 构建

先在本机生成匹配架构的 acados solver：

```bash
cd /home/cy2/nmpc
source .venv/bin/activate
source /opt/ros/noetic/setup.bash
export ACADOS_SOURCE_DIR=/home/cy2/acados
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
python solver/generate_solver.py \
  --sample-time 0.01 --control-period 0.01 \
  --model-name quadrotor_nmpc_ros1_iris_100hz \
  --code-export-directory generated/quadrotor_nmpc_ros1_iris_100hz \
  --json-file quadrotor_nmpc_ros1_iris_100hz.json
```

再构建 catkin workspace：

```bash
cd .catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release \
  -DACADOS_SOURCE_DIR=/home/cy2/acados \
  -DESO_NMPC_ROOT=/home/cy2/nmpc
source devel/setup.bash
```

## 运行与日志

```bash
roslaunch eso_nmpc_cpp eso_nmpc_cpp.launch
```

NMPC 由 `/mavros/local_position/odom` 回调触发。节点连接飞控后默认通过
`/mavros/set_message_interval` 请求 MAVLink ID 31（姿态）、105（原始 IMU）和
32（融合局部位置）均请求为 100 Hz；其中 ID 32 的实际到达频率决定 NMPC 的触发上限。
当前 ROS1 C++ 试验配置的 `sample_time` 为 0.01 s（100 Hz 设计周期）。

以上默认命令已退出影子模式并启动 NMPC 预发送，但自动飞行管理仍关闭。

需要由本节点负责飞行状态机时显式打开（会请求 OFFBOARD、解锁和结束后的
`AUTO.LAND`）：

```bash
roslaunch eso_nmpc_cpp eso_nmpc_cpp.launch auto_manage_flight:=true
```

RC-NMPC 通过 `/mavros/manual_control/control` 获取摇杆、通过 `/mavros/rc/in`
读取 AUX；真机默认使用第 6 通道，AUX 高于阈值才切换到 RC 参考源。
在随包提供的 SITL 注入回归中，`rc_injector.py` 以 20 Hz（50 ms）发布摇杆和
AUX；这是输入更新频率，不是 NMPC 输出频率。RC 值在两次消息之间保持，NMPC
仍由里程计回调重算并发布 `/mavros/setpoint_raw/attitude`；当前试验配置理想情况下约 100 Hz，
实际频率以上述里程计到达频率为上限。

PX4 SITL 的 RC 注入回归可直接运行：

```bash
source /home/cy2/ros1_ws/devel/setup.bash
roslaunch eso_nmpc_cpp px4_ros1_rc_nmpc.launch gui:=false paused:=false
```

该 launch 会自动起 PX4/Gazebo、C++ NMPC 和 `rc_injector.py`；AUX6 在 3--14 s
置高，6--10 s 前推、10--12 s 后拉，18 s 后关闭控制使状态机进入降落。

接入阶段会先等待定位有效且机体稳定（默认持续 1 s、水平速度不超过 0.15 m/s、
垂直速度不超过 0.10 m/s、位置漂移不超过 0.25 m），随后发送 1.5 s 的安全保持
setpoint；确认进入 OFFBOARD 后再等待 0.2 s，才放开轨迹或 AUX6 输入。将
`auto_manage_flight` 设为 `false` 时，节点仍执行稳定判定和保持预发送，但不会
替操作者切 OFFBOARD 或解锁。

当 Acados 连续求解失败达到 `max_consecutive_solve_failures` 时，节点不会退出，
而是锁存故障、停止发布 NMPC setpoint，并请求 PX4 进入 `position_mode`（默认
`AUTO.LOITER` 定点）。节点仍在后台用当前状态保持参考继续求解；连续
`solver_recovery_time` 秒成功后只清除求解故障标志，不会自动重新接管。操作者
必须再次手动选择 OFFBOARD，节点才恢复 NMPC 输出。

确认 PX4 已处于 `AUTO.LOITER` 且求解器已经连续正常 `solver_recovery_time` 秒后，
只需操作者手动切回 OFFBOARD；在此之前节点会继续阻止 NMPC 输出。

节点不再根据里程计自动生成悬停参考；必须由外部规划器向
`/nmpc/in/trajectory_setpoint` 发布完整的 `NmpcTrajectorySetpoint`，数组按点展开，
位置/速度/加速度使用 NED，航向为 NED yaw。每次启动默认写入：

```text
/home/cy2/nmpc_log/YYYYMMDD_HHMMSS_mmm/nmpc_timing.csv
```

完整轨迹接口和 ROS1 消息定义位于 `msg/NmpcTrajectorySetpoint.msg`。

当前 ROS1 C++ 适配已在单节点内接入 ESO、RC-NMPC 参考源以及 PX4 模式/解锁/降落
状态机（自动状态机默认关闭，需显式打开）；
尚未宣称完成真实 PX4 飞行验收。

控制回调只入队日志记录，文件由后台线程异步写入，默认每 250 ms 刷盘。
acados 先使用热启动；热启动失败时立即冷启动重试一次。冷启动仍失败时，仅在最近
50 ms 内有有效命令的情况下短暂保持该命令，否则不发布控制量；随后累计失败次数，连续 3 次后自动停用 NMPC，需重新发送
`control_enabled=true` 才会恢复。ROS1 的心跳由 MAVROS setpoint 链路承担，
不像 ROS2 有独立的 `OffboardControlMode` 消息。
