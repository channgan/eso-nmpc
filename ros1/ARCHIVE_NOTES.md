# ROS1 NMPC 工作封存记录

封存日期：2026-09-03

## 内容

- `eso_nmpc_cpp/`：ROS1 C++ ESO+NMPC 节点、生成消息、启动文件、RC 注入器和配置。
- `eso_nmpc_ros1/`：原 ROS1 接口兼容包与 SITL 启动文件；Python 控制器已移除。
- 生成求解器：`generated/quadrotor_nmpc_ros1_iris_100hz`，`Tsim=0.01`、`N=30`。

## 当前控制流程

1. 等待估计器有效，并连续满足稳定条件：1 s 内水平速度 ≤ 0.15 m/s、垂直速度 ≤ 0.10 m/s、位置漂移 ≤ 0.25 m。
2. 发送 1.5 s 的当前位置保持 setpoint。
3. 自动管理模式请求 `OFFBOARD`；手动管理模式等待操作者切换。
4. 确认 `OFFBOARD` 后再延迟 0.2 s，才接入轨迹控制。
5. `AUX6` 高选择 RC-NMPC 参考，低选择外部轨迹参考。
6. ACADOS 连续失败 3 次时停止 NMPC 输出，请求 PX4 `AUTO.LOITER`，后台继续求解；连续正常 2 s 后等待人工重新切回 `OFFBOARD`，不自动接管。

## 验证

- ROS1 Release 构建：`catkin_make -DCMAKE_BUILD_TYPE=Release` 通过。
- `git diff --check` 通过。
- PX4/Gazebo SITL 已观察到：稳定判定 → 安全保持预发送 → OFFBOARD 确认 → 0.2 s 延迟 → NMPC flight started。
- 当前环境无 ROS2 运行时，ROS2 未在本封存中做运行验证。

## 启动

```bash
source /opt/ros/noetic/setup.bash
source /home/cy2/ros1_ws/devel/setup.bash
export ROS_PACKAGE_PATH=/home/cy2/apx/Tools/simulation/gazebo-classic/sitl_gazebo-classic:/home/cy2/apx:/home/cy2/ros1_ws/src:$ROS_PACKAGE_PATH
export PX4_ROOT=/home/cy2/apx
roslaunch eso_nmpc_cpp px4_ros1_rc_nmpc.launch gui:=false paused:=false
```

归档只读副本不包含 PX4 源码、Gazebo 日志和 ROS 编译缓存；原工作目录仍保留。
