# NMPC 统一使用逻辑与条件

本文是 ROS1 C++ 和 ROS2 C++ 控制器共用的运行契约。两套节点只允许在通信接口、消息类型和飞控桥接方式上不同；状态转换、参考源选择、接入时机、求解失败处理和人工恢复规则必须保持一致。本文中的数值以当前 0.01 s（100 Hz 设计周期）配置为准，实际控制回调频率由状态消息到达频率决定。

## 1. 运行角色

| 角色 | 职责 |
| --- | --- |
| PX4 | 负责飞行器底层姿态/角速度闭环、模式切换和原生 failsafe。 |
| NMPC C++ 节点 | 读取状态、估计扰动、生成或接收完整预测时域、求解 acados，并发布最终 body-rate/thrust。 |
| 飞行管理器/操作者 | 负责确认起飞条件、选择 OFFBOARD、解锁，以及故障后的人工重新接管。ROS1 可由节点代管；ROS2 默认由外部管理器代管。 |
| 轨迹规划器 | 发布完整的 NED 轨迹参考；不把单点位置或 PX4 位置控制器输出当作 NMPC 参考。 |

## 2. 统一状态机

```text
WAIT
  └─ 状态和定位有效，且机体稳定持续 1 s
       ↓
STABILIZE → PRESTREAM（安全保持 setpoint 持续 1.5 s）
       ↓
READY（手动管理） 或  OFFBOARD_REQ（自动管理）
       ↓
检测到 PX4=OFFBOARD
       ↓
ENGAGING（再等待 0.2 s）
       ↓
FLIGHT（接入轨迹/RC-NMPC）

FLIGHT ──连续求解失败达到阈值──→ FAULT_HOLD
                                      ↓
                         PX4 定点保持，NMPC 后台恢复求解
                                      ↓
                         求解连续正常 2 s，等待人工 OFFBOARD
                                      ↓
                         ENGAGING → FLIGHT
```

任何阶段只要 `control_enabled=false` 或 `shadow_mode=true`，都不得向 PX4 输出实际 NMPC 控制。离开 OFFBOARD 后，NMPC 输出立即暂停；重新选择 OFFBOARD 后仍须经过接入延时，不能直接跳过安全门。

## 3. 接入前必须同时满足的条件

### 3.1 软件和数据条件

1. 使用 C++ 节点和与当前模型匹配的 acados 生成 solver；Python 控制器不属于正式运行链路。
2. 节点已连接 PX4，状态消息未超时（当前 `state_timeout_s=1.0`）。
3. 姿态、水平/垂直速度、水平/垂直位置估计有效；带桨飞行不得把 armed 状态下的无效或 const/fake position 当作有效定位。
4. `control_enabled=true`，且没有锁存的求解故障或人工重新接管标志。
5. 外部轨迹必须满足：完整 `N+1` 个点（当前为 31 点）、`sample_time=0.01`、NED 坐标、有限值、单调时间戳；超过 `reference_timeout=0.20 s` 即失效。

### 3.2 机体稳定条件

以首次有效状态为保持锚点，以下条件必须连续满足 `handoff_stable_time=1.0 s`：

- 水平速度不超过 `0.15 m/s`；
- 垂直速度绝对值不超过 `0.10 m/s`；
- 相对锚点的位置漂移不超过 `0.25 m`。

任一条件被破坏，稳定计时清零并重新建立锚点。稳定判定只决定“可以开始接入”，不代表已经进入 OFFBOARD。

## 4. 标准接入流程

1. 上电启动节点，先保持 `shadow_mode=true` 或 `control_enabled=false` 完成连接、定位、话题和 solver 检查。
2. 确认 PX4 已在定点/位置保持状态，机体稳定条件满足 1 s。
3. 节点发布当前状态的安全保持 setpoint，持续 1.5 s，期间不使用未来轨迹或摇杆指令驱动机体。
4. 手动管理模式下，操作者在安全保持预发送完成后选择 PX4 `OFFBOARD`；节点只观察并确认模式，不代替切换或解锁。
5. 自动管理模式下，节点可以请求 `OFFBOARD` 和解锁，但仍必须先完成稳定判定和 1.5 s 预发送；请求超时不得强行输出，按飞行管理策略处理。
6. 观察到 PX4 确认 `OFFBOARD` 后，再等待 `offboard_entry_delay=0.20 s`。
7. 延时结束且飞控保持 armed 后，才允许接入完整轨迹或 RC-NMPC 参考，记录 `NMPC flight started`。

ROS1 的 `auto_manage_flight` 默认关闭；只有显式打开才由节点请求模式/解锁。ROS2 默认由外部飞行管理器执行第 4、5 步，但顺序和安全条件必须相同。

## 5. NMPC 参考源选择

参考源在每个控制回调中互斥选择，不允许两套参考叠加：

| 条件 | 采用的参考 |
| --- | --- |
| AUX6（`rc_aux_channel=6`）低于阈值 | 外部 `/nmpc/in/trajectory_setpoint` 完整轨迹 |
| AUX6 高于 `rc_aux_enable_threshold=0.5`，且手动控制消息新鲜（`rc_timeout_s=0.5 s`） | RC-NMPC；摇杆经死区、加速度、速度和位置前视限制后生成完整预测时域 |
| AUX6 高，但手动控制消息已超时 | 拒绝继续使用旧摇杆指令，保持当前 RC 参考/归零摇杆；不自动切到外部轨迹 |
| AUX6 从高切低 | 退出 RC-NMPC，清除 RC 参考状态并切回最新有效外部轨迹 |

AUX6 只切换“NMPC 的输入参考源”，不负责切换 PX4 的飞行模式，也不负责解锁。必须先处于已确认的 OFFBOARD/FLIGHT 阶段，AUX6 才有控制意义。

## 6. 正常控制回调顺序

每次新状态到达时，ROS1/ROS2 均按以下顺序执行：

1. 校验状态和时间戳，并转换到 PX4 NED/FRD；
2. 按传感器时间计算采样间隔，更新 ESO（启动后先经过 `eso_activation_delay=3 s`）；
3. 选择外部轨迹或 RC 参考，检查时域、有限值和运动包线；
4. 构造完整预测状态、前馈和扰动参数；
5. acados 先热启动求解；
6. 热启动失败时立即 reset 并冷启动重试一次；
7. 成功后发布归一化 thrust 和 body-rate，并异步记录日志；
8. 控制周期目标为 0.01 s，不能用 RC 输入频率替代 NMPC 输出频率。RC 注入 20 Hz 只代表摇杆更新频率，状态回调仍决定求解和发布频率上限。

## 7. 求解失败、保底和恢复

### 7.1 单次失败

- 热启动和冷启动都失败时，只允许在最近 `fallback_hold_time=0.05 s` 内短暂保持上一条有效命令；
- 超过该窗口不得继续发布旧命令；
- 进程保持运行，记录失败原因和计数，不能因一次失败自动退出节点。

### 7.2 连续失败达到阈值

当前阈值为 `max_consecutive_solve_failures=3`（按控制回调计数）：

1. 锁存 NMPC 故障，停止 NMPC 控制输出；
2. 节点不退出，后台使用当前状态的安全保持参考继续尝试求解；
3. `auto_manage_flight=true` 时请求 PX4 `AUTO.LOITER`（定点保持）；手动管理模式则由外部飞行管理器/操作者切到定点保持；
4. 求解连续正常 `solver_recovery_time=2.0 s` 后，只清除“求解已恢复”标志，不自动回到 OFFBOARD，也不自动恢复轨迹控制。

### 7.3 人工重新接管

故障后必须先确认 PX4 已离开 OFFBOARD 并处于定点保持，再由操作者手动重新选择 OFFBOARD。节点确认“已经离开过 OFFBOARD + solver 已连续健康 + 当前再次进入 OFFBOARD”三个条件后，才解除人工接管锁存，并重新经过 0.2 s 接入延时。

若重新进入 OFFBOARD 前再次发生求解失败，保持故障状态；不得通过自动反复切换模式来试探恢复。

## 8. 正常退出和人工干预

- 主动结束控制：先将 `control_enabled=false`，确认 NMPC 不再发布，再由 PX4 执行降落或其他模式切换。
- 退出 OFFBOARD：节点立即暂停 NMPC 输出；重新进入 OFFBOARD 仍要经过确认延时。
- `shadow_mode=true`：允许计算、记录和验证，但禁止真实控制输出，适合无桨、台架和通信联调。
- 任何定位、RC、轨迹、PX4 状态不满足条件时，安全动作是保持/交回 PX4，而不是发布最后一条非零控制命令。

## 9. ROS1/ROS2 通信映射（逻辑保持一致）

| 逻辑数据 | ROS1 | ROS2 |
| --- | --- | --- |
| 状态输入 | `/mavros/local_position/odom` | `/fmu/out/vehicle_odometry` |
| 完整轨迹 | `/nmpc/in/trajectory_setpoint` | `/nmpc/in/trajectory_setpoint` |
| 控制使能 | `/nmpc/control_enabled` | `/nmpc/control_enabled` |
| RC/手动输入 | `/mavros/manual_control/control`、`/mavros/rc/in` | `/fmu/out/manual_control_setpoint` |
| 控制输出 | `/mavros/setpoint_raw/attitude` | `/fmu/in/vehicle_rates_setpoint` |
| Offboard 心跳 | 由 MAVROS setpoint 链路承担 | `/fmu/in/offboard_control_mode` |

ROS1 与 ROS2 的消息桥接不同，不得据此改变上述状态机、参考源优先级、失败阈值或人工恢复条件。修改任一条件时，应同时更新两套配置、本文档和对应回归测试。

### 当前代码边界

统一契约不等于两套节点当前已经拥有完全相同的飞行管理能力：

- ROS1 C++ 节点已经包含稳定判定、1.5 s 安全保持预发送、OFFBOARD 确认延时、`AUTO.LOITER` 故障转移和人工重新接管锁存。
- ROS2 C++ 节点把模式切换、解锁和飞行状态管理交给外部飞行管理器；外部管理器必须按本文第 2～4 节执行相同的门控顺序，不能仅通过“节点进程在运行”判定可以接管。
- ROS2 当前节点的求解故障停用由 `/nmpc/control_enabled` 重新使能；若要实现与 ROS1 相同的“后台保持求解、连续健康后等待人工 OFFBOARD”行为，应在 ROS2 飞行管理器或节点中补齐该状态机后，才能宣称两端完全一致。

因此，本文是后续统一实现和回归验收的基准；通信接口不同不是放宽安全条件的理由。

## 10. 最低验收记录

每次使用或回归至少记录：控制是否实际输出、当前 PX4 模式、armed 状态、参考源（external/RC）、AUX6 状态、稳定计时、预发送完成时间、OFFBOARD 确认时间、solver 成功/失败次数、故障恢复时间和人工重新接管时间。没有这些记录，不把“节点运行”判定为“NMPC 已安全接管”。
