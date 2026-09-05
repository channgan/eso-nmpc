# NMPC 使用逻辑与条件（统一操作版）

本文只规定“什么时候允许 NMPC 接管、什么时候切换参考源、什么时候必须交回 PX4”。ROS1 和 ROS2 的消息接口可以不同，但操作顺序和安全条件相同。当前控制周期、MPC 预测和轨迹采样步长均为 `0.01 s`（100 Hz），实际求解频率由状态消息到达频率决定。当前默认 YAML 与 C++ 无参数兜底均为热启动开启、ESO 开启、带宽 `2.5 rad/s`；关闭它们只能作为独立对照测试。

启动职责明确为：Python 脚本可以负责启动和监督 PX4、Gazebo、MicroXRCE-DDS Agent 以及飞行流程；正式 NMPC 控制节点必须使用 C++，并在整个飞行会话中持续运行。Python 脚本不替代 C++ NMPC 节点。

## 一、先分清两件事

1. **OFFBOARD 切换**：决定由 PX4 的位置控制器还是上位机 NMPC 输出控制。
2. **AUX6 切换**：只决定 NMPC 使用哪一种参考源，不切换 PX4 模式、不解锁、不启动节点。

地面起飞和空中接管是两个不同阶段。解锁只发生在起飞阶段，不属于 NMPC 空中接管步骤。

正常使用顺序固定为：

```text
PX4 已解锁并在安全高度定点保持
  → 机体稳定
  → 发送安全保持 setpoint
  → 切入 OFFBOARD（不重新解锁）
  → NMPC 先接管外部轨迹
  → AUX6 打高后才切到 RC-NMPC
```

AUX6 不是“启动 NMPC”的开关。没有确认 OFFBOARD 和 NMPC 接入完成时，拨 AUX6 不应产生飞行控制效果。

## 二、启动和接入前检查

启动阶段默认不输出真实 NMPC 控制：

- ROS1 使用 `control_enabled_at_start=false`；无桨联调还应使用 `shadow_mode=true`。
- ROS2 使用 `control_enabled_at_start=false`，由外部飞行管理器在检查完成后发布 `/nmpc/control_enabled=true`；C++ 节点提前启动并持续运行。
- 必须使用 C++ 节点和与模型匹配的 acados solver，Python 控制器不属于正式链路。

允许接入前，以下条件必须同时满足：

| 检查项 | 条件 |
| --- | --- |
| PX4 连接 | 已连接，状态消息未超时；ROS1 当前状态超时阈值为 `1.0 s`。 |
| 定位 | 姿态、水平/垂直速度、水平/垂直位置估计有效。带桨时不能接受 armed 状态下的 const/fake position。 |
| 控制使能 | `control_enabled=true`，无锁存的求解故障或人工接管锁存。 |
| 外部轨迹 | 完整 `N+1` 个点；当前 `N=30`，所以为 31 点；轨迹采样 `0.01 s`、NED 坐标、有限值、时间戳有效。 |
| 轨迹新鲜度 | 外部轨迹接收时间距当前不超过 `reference_timeout=0.20 s`。 |

## 三、正确的接入顺序

### 1. 先让 PX4 定点稳住

操作者先完成解锁和起飞，使 PX4 已处于安全高度的定点/位置保持状态。此处不再次解锁。节点以当前状态建立保持锚点，连续满足以下条件 `handoff_stable_time=1.0 s`：

- 水平速度不超过 `0.15 m/s`；
- 垂直速度绝对值不超过 `0.10 m/s`；
- 相对保持锚点的位置漂移不超过 `0.25 m`。

任意条件被破坏，计时清零并重新建立锚点。这个阶段只说明“可以准备接入”，不等于已经进入 OFFBOARD。

### 2. 发送安全保持 setpoint

稳定判定完成后，节点发送当前点的安全保持 setpoint，持续 `prestream_s=1.5 s`。这 1.5 s 内不使用未来轨迹，也不使用摇杆产生的运动参考。

### 3. 切入 OFFBOARD

- **手动管理**（默认）：预发送完成后，由操作者切 PX4 到 `OFFBOARD`；节点只确认模式，不替操作者切模式或解锁。
- **自动管理**：Python 启动/监督脚本或外部飞行管理器可在预发送完成后请求 `OFFBOARD`；此时飞行器已经 armed，不重复发送解锁请求。

节点观察到 PX4 已确认 `OFFBOARD` 后，还要等待 `offboard_entry_delay=0.20 s`。延时结束且飞控保持 armed，才记录 NMPC 接管完成并允许运动参考生效。

ROS2 当前不在节点内管理 PX4 模式和解锁；外部飞行管理器必须执行与上面完全相同的稳定判定、1.5 s 预发送、OFFBOARD 确认和 0.2 s 延时。

### 4. 先由外部轨迹接管

进入 FLIGHT 后，AUX6 默认置低，NMPC 使用 `/nmpc/in/trajectory_setpoint` 的完整外部轨迹。确认轨迹跟踪正常后，才允许切换 RC-NMPC。

## 四、AUX6 输入源切换

当前约定 `rc_aux_channel=6`，阈值 `rc_aux_enable_threshold=0.5`。AUX6 保持低位即可持续使用外部轨迹，不需要操作者切换 AUX6；只有明确需要 RC-NMPC 时才拨高。

| AUX6 状态 | NMPC 参考源 | 行为 |
| --- | --- | --- |
| 低 | 外部轨迹 | 使用最新且未超时的完整轨迹。 |
| 高，RC 消息新鲜（`rc_timeout_s=0.5 s` 内） | RC-NMPC | 摇杆经过死区、加速度、速度和位置前视限制后，在节点内生成完整预测时域。 |
| 高，但 RC 消息超时 | 失效保护 | 锁存关闭 NMPC 输出，发布 `/nmpc/rc_timeout=true`，由外部飞行管理器请求 PX4 Position/Hold；不自动改用外部轨迹。 |
| 从高切回低 | 外部轨迹 | 清除 RC 参考状态，切回最新有效外部轨迹。 |

AUX6 只切换 NMPC 的**输入源**。它不负责切换 PX4 到 OFFBOARD、解锁或上锁、启停 NMPC 节点，也不能绕过定位、轨迹新鲜度或故障门控。

RC 超时不是“保持当前参考”的正常控制状态，而是安全故障。C++ 节点停止发布
`VehicleRatesSetpoint` 并保持进程和 Offboard 心跳；外部飞行管理器订阅
`/nmpc/rc_timeout` 后必须关闭 `/nmpc/control_enabled`，请求 PX4 AUTO.LOITER 定点保持；如果 PX4
同时检测到原生 RC loss，则允许由已配置的 Land failsafe 接管。恢复必须重新完成准备检查，由操作者手动确认 OFFBOARD，再显式发布
`control_enabled=true`；节点不会自动重新接管。

ROS1 从 `/mavros/manual_control/control` 和 `/mavros/rc/in` 读取输入；ROS2 从
`/fmu/out/manual_control_setpoint` 读取输入。ROS2 C++ 节点只接受
`data_source=SOURCE_RC`、`valid=true` 且在 `rc_timeout=0.5 s` 内更新的输入；输入更新频率
（例如 RC 注入 20 Hz）不等于 NMPC 输出频率。

## 五、每次状态更新的统一控制逻辑

ROS1/ROS2 的控制计算顺序统一为：

1. 收到状态，检查有限值和时间戳，转换到 PX4 NED/FRD；
2. 使用传感器时间间隔更新 ESO；启动后经过 `eso_activation_delay=3 s` 才启用 ESO 更新；
3. 按 AUX6 选择外部轨迹或 RC-NMPC，并校验完整时域；
4. 构造参考状态、前馈和扰动参数；
5. 当前默认 `warm_start=true`，先将上一轮解按一个控制周期平移，再把 `x0` 锚定到最新实测状态；若对照关闭热启动则直接冷启动；
6. 热启动（若启用）或冷启动失败时 reset，并按故障策略处理；
7. 求解成功才发布最终 body-rate/thrust，并异步写日志；
8. 控制目标周期、MPC 离散和轨迹步长均为 `0.01 s`（100 Hz），实际求解频率不能高于状态输入频率。

### 状态时间戳连续性保护

状态消息的实际 ROS2 到达间隔必须不超过
`odometry_timestamp_gap_threshold=0.10 s`，判据使用本机单调时钟，而不是 PX4 消息时间戳。
这个条件独立于控制周期：控制器不能在收到停顿后的下一帧时，把 `dt` 强行替换成 `0.01 s`
继续积分、更新 ESO 或求解，因为这会把仿真/通信停顿造成的状态跳变伪装成真实飞行动力学。

`timestamp_sample` 仍需记录并检查样本年龄，但它属于 PX4 估计器采样时刻，不能单独作为
ROS2 消息连续性的判据；估计器重置或 DDS 排队可能让它与发布时刻出现不同步。

一旦检测到实际接收间隔超过阈值：

1. C++ 节点丢弃这次状态更新，清除热启动和 ESO 瞬态状态，停止发布 NMPC 控制输出，并锁存
   `/nmpc/odometry_timestamp_fault=true`；
2. ROS2 外部监督器关闭 `control_enabled`，请求 PX4 `Position/Hold`（可配置为
   `AUTO.LOITER`），不自动重新进入 OFFBOARD；
3. 状态时间连续恢复后也不自动重新接管。操作者必须重新完成稳定检查、setpoint 预发送、
   OFFBOARD 确认和 0.2 s 延时，再显式发布 `control_enabled=true`；如果已经降落，还必须
   人工重新解锁；
4. PX4 `timestamp` 回退或重复只说明 DDS 送达顺序异常：C++ 丢弃该旧消息并继续等待新消息，
   不单独触发保护；回归日志仍记录该事件。实际接收间隔出现一次超阈值即判该用例失败，但应优先归类
   为仿真/DDS/调度时间链问题，而不是直接归类为 solver 发散。

该保护只隔离故障状态，不会消除 Gazebo、PX4 或 DDS 本身的停顿。正式回归仍必须使用标准
PX4 SITL 启动顺序，并在结果中同时检查状态时间间隔、`rx→pub` 延迟和 Acados 求解时间。

### ESO 带宽选择与发散判据

ESO 带宽是估计器的响应速度，不是 NMPC 控制周期。带宽增大后，估计器会更快跟踪速度
残差；如果残差主要来自模型失配、测量噪声或时间戳/执行器延迟，就可能被误认为真实
扰动，并通过前馈和预测模型快速反馈，造成过补偿甚至闭环发散。带宽降低会使估计更慢、
更平滑，但也会减弱对快速真实扰动的补偿。

当前默认 YAML 为 `2.5 rad/s`。在三扰动条件（重心前偏 `0.02 m`、X 向风 `0.5 m/s`、
位置/速度随机游走 `0.002/0.005`）下，`3 rad/s` 轮次四项失败，而 `2.5 rad/s` 轮次
四项通过；这只能作为支持“带宽过大可能引发发散”的初步证据，不能单凭一轮测试定型。
三扰动验证期间暂以 `2.5 rad/s` 作为候选，并保持独立 best 模式记录。

带宽变更后的检查必须同时包括：

- `d_hat` 是否持续接近或撞到 `eso_clamp=1 m/s²`；
- ESO 启用前后位置误差和控制量是否出现突变；
- Acados 连续失败次数及四项位置 RMSE；
- `rx_to_pub` 与 Acados solve 的 P99，确认不是计算超时导致的表面发散。

推荐在固定三扰动下按 `1.5 → 2.0 → 2.5 → 3.0 rad/s` 顺序扫描。每个带宽必须使用
独立输出目录和 best 模式，不能覆盖其他带宽的记录。

## 六、求解失败和交回 PX4

### 单次失败

- 热启动和冷启动都失败时，只能在最近 `fallback_hold_time=0.05 s` 内短暂保持上一条有效命令；
- 超过 50 ms 不得继续发布旧的非零命令；
- 节点进程保持运行并记录失败，不因单次失败自动退出。

### 连续失败达到 3 次

当 `max_consecutive_solve_failures=3`：

1. 锁存 NMPC 故障，立即停止发布 NMPC `VehicleRatesSetpoint`；
2. 请求 PX4 切换到 `AUTO.LOITER` 定点模式；ROS2 由外部飞行管理器发送该模式请求，ROS1 由对应管理脚本发送；
3. C++ 节点不退出，继续运行；后台使用当前状态构造保持轨迹并继续求解，用于判断 solver 是否恢复；
4. 求解连续正常 `2.0 s` 后，只清除求解故障状态，不自动重新接管；
5. 不自动重新切回 OFFBOARD，也不自动重新解锁。若已经降落，必须人工重新解锁。

故障状态下 C++ 节点会继续后台尝试保持参考求解；求解连续正常 `solver_recovery_time=2.0 s` 后，仅标记 solver 已恢复。恢复输出前仍需操作者重新选择 OFFBOARD，并由外部管理器重新发布 `control_enabled=true`，不能把“求解器恢复”当作“已经重新接管”。

以上是正式飞行必须达到的目标策略。求解连续失败的计数、锁存、后台恢复判定和人工恢复流程
仍需按本节验收；状态时间戳跳变保护已经在当前 C++ 节点和 ROS2 监督器中实现。

## 七、故障后的人工恢复

故障后必须按以下顺序恢复：

```text
NMPC 故障锁存
  → PX4 离开 OFFBOARD
  → PX4 AUTO.LOITER 定点保持
  → 确认 solver 连续正常 2 s
  → 操作者再次手动选择 OFFBOARD
  → 等待 0.2 s
  → 外部管理器重新发布 control_enabled=true
  → NMPC 恢复输出
```

重新进入 OFFBOARD 前再次失败，继续保持故障，不通过自动切换模式试探。AUX6 只有在 NMPC 已重新进入 FLIGHT 后才重新有效。

## 八、退出和验收口径

- 主动退出：先发布 `control_enabled=false`，确认 NMPC 不再输出，再由 PX4 执行降落或其他模式切换。
- 离开 OFFBOARD：NMPC 输出暂停；需要重新切回 OFFBOARD 时，必须重新满足稳定检查、安全预发送、模式确认和 0.2 s 延时，不能自动进入。
- Shadow 模式就是关闭真实控制输出、不进入控制接管；C++ 进程可以继续运行用于状态/健康检查，但不得发布 NMPC 控制 setpoint。
- 当前 `control_enabled=false` 会关闭控制回调输出；若要满足“关闭输出但继续计算/记录健康状态”的完整 shadow 语义，还需在 C++ 节点中补充独立 shadow 状态，而不能把它误认为已经实现。
- 每次回归必须记录 PX4 模式、armed、参考源、AUX6、轨迹新鲜度、稳定计时、预发送完成、OFFBOARD 确认、solver 成功/失败和故障恢复时间。

ROS1 与 ROS2 的通信映射见各自 README；如果修改上述任一条件，必须同步修改两套配置、本文档和回归测试，不能只改一端。

## 九、修改参数后的固定流程

任何参数修改后，不得直接复用旧的可执行文件或旧的回归结果。必须先判断该参数是否改变 Acados 的 OCP/离散模型：

| 参数类型 | 例子 | 修改后动作 |
| --- | --- | --- |
| 运行时参数 | `mass`、`hover_throttle`、ESO 限幅、RC 阈值 | 确认 YAML、重新编译/安装 C++ 节点（如配置已安装到工作空间），然后重新做环境检查和 SITL 冒烟。 |
| 影响 OCP 或离散化的参数 | `sample_time`、`horizon_steps`、模型、代价权重、输入/状态约束 | 必须重新生成 solver、编译生成的 `.o/.so`、更新 solver hash 引用，再重新编译 ROS2 C++ 节点。 |
| PX4 参数 | `COM_OF_LOSS_T`、`COM_RC_LOSS_T`、`MPC_*` 等 | 通过参数守卫写入并读回确认；正式回归不能使用 `--skip-params`。 |

标准命令顺序如下：

```bash
cd /home/cy/eso_nmpc
source /opt/ros/humble/setup.bash
source /home/cy/eso_nmpc/.venv/bin/activate
export ACADOS_SOURCE_DIR=/home/cy/acados
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=/home/cy/acados/lib:/home/cy/eso_nmpc/generated/quadrotor_nmpc:$LD_LIBRARY_PATH

# 只有 OCP/离散化相关参数改变时执行生成步骤
python3 solver/generate_solver.py
make -C generated/quadrotor_nmpc

# 每次 C++ 源码、solver 或 C++ 参数接口改变后都重新构建
cd /home/cy/px4_ros2_ws
colcon build --packages-select eso_nmpc_node --cmake-clean-cache \
  --cmake-args -DESO_NMPC_ROOT=/home/cy/eso_nmpc \
  -DACADOS_SOURCE_DIR=/home/cy/acados
```

编译后必须确认以下内容全部一致：

1. `cpp/eso_nmpc_node/CMakeLists.txt`、C++ 源码、检查脚本和 `generated/quadrotor_nmpc/` 使用同一个 solver hash；
2. 回归脚本使用 `/home/cy/px4_ros2_ws/install/eso_nmpc_node/` 下的当前 C++ 可执行文件，不使用旧的本地 `build_cpp/install_cpp`；
3. `check_environment.py --require-generator` 在 source 正确的 ROS2/PX4 工作空间后无 failure；
4. 新 PX4/Gazebo 实例先通过悬停冒烟，再执行悬停、1 m 指点、圆形、8 字四项回归；
5. 绘图默认读取监督器的 `trajectory.csv`，`nmpc_flight.csv` 只用于单独分析 C++ 节点 per-solve 日志；
6. 新结果必须使用新目录保存，不能覆盖旧基线，且要同时保留 `suite_summary.md`、机器可读 JSON、CSV 和长图。

若只修改了运行时 YAML 参数，也必须重新执行环境检查和至少一次悬停冒烟；若修改了 `sample_time` 或其他 OCP 参数，则在 solver、C++ 节点和四项回归全部重新验证前，不得用于正式飞行。
