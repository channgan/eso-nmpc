# NMPC 使用逻辑与条件（统一操作版）

本文只规定“什么时候允许 NMPC 接管、什么时候切换参考源、什么时候必须交回 PX4”。ROS1 和 ROS2 的消息接口可以不同，但操作顺序和安全条件相同。当前控制周期配置为 `0.01 s`；100 Hz 是设计频率，实际求解频率由状态消息到达频率决定。

## 一、先分清两件事

1. **OFFBOARD 切换**：决定由 PX4 的位置控制器还是上位机 NMPC 输出控制。
2. **AUX6 切换**：只决定 NMPC 使用哪一种参考源，不切换 PX4 模式、不解锁、不启动节点。

正常使用顺序固定为：

```text
PX4 定点保持
  → 机体稳定
  → 发送安全保持 setpoint
  → 切入 OFFBOARD
  → NMPC 先接管外部轨迹
  → AUX6 打高后才切到 RC-NMPC
```

AUX6 不是“启动 NMPC”的开关。没有确认 OFFBOARD 和 NMPC 接入完成时，拨 AUX6 不应产生飞行控制效果。

## 二、启动和接入前检查

启动阶段默认不输出真实 NMPC 控制：

- ROS1 使用 `control_enabled_at_start=false`；无桨联调还应使用 `shadow_mode=true`。
- ROS2 使用 `control_enabled_at_start=false`，由外部飞行管理器在检查完成后发布 `/nmpc/control_enabled=true`。
- 必须使用 C++ 节点和与模型匹配的 acados solver，Python 控制器不属于正式链路。

允许接入前，以下条件必须同时满足：

| 检查项 | 条件 |
| --- | --- |
| PX4 连接 | 已连接，状态消息未超时；ROS1 当前状态超时阈值为 `1.0 s`。 |
| 定位 | 姿态、水平/垂直速度、水平/垂直位置估计有效。带桨时不能接受 armed 状态下的 const/fake position。 |
| 控制使能 | `control_enabled=true`，无锁存的求解故障或人工接管锁存。 |
| 外部轨迹 | 完整 `N+1` 个点；当前 `N=30`，所以为 31 点；`sample_time=0.01`、NED 坐标、有限值、时间戳有效。 |
| 轨迹新鲜度 | 外部轨迹接收时间距当前不超过 `reference_timeout=0.20 s`。 |

## 三、正确的接入顺序

### 1. 先让 PX4 定点稳住

操作者先让 PX4 处于定点/位置保持状态。节点以当前状态建立保持锚点，连续满足以下条件 `handoff_stable_time=1.0 s`：

- 水平速度不超过 `0.15 m/s`；
- 垂直速度绝对值不超过 `0.10 m/s`；
- 相对保持锚点的位置漂移不超过 `0.25 m`。

任意条件被破坏，计时清零并重新建立锚点。这个阶段只说明“可以准备接入”，不等于已经进入 OFFBOARD。

### 2. 发送安全保持 setpoint

稳定判定完成后，节点发送当前点的安全保持 setpoint，持续 `prestream_s=1.5 s`。这 1.5 s 内不使用未来轨迹，也不使用摇杆产生的运动参考。

### 3. 切入 OFFBOARD

- **手动管理**（默认）：预发送完成后，由操作者切 PX4 到 `OFFBOARD`；节点只确认模式，不替操作者切模式或解锁。
- **自动管理**（ROS1 `auto_manage_flight=true`）：节点可在预发送完成后请求 `OFFBOARD`，随后请求解锁；请求超时不得强行输出。

节点观察到 PX4 已确认 `OFFBOARD` 后，还要等待 `offboard_entry_delay=0.20 s`。延时结束且飞控保持 armed，才记录 NMPC 接管完成并允许运动参考生效。

ROS2 当前不在节点内管理 PX4 模式和解锁；外部飞行管理器必须执行与上面完全相同的稳定判定、1.5 s 预发送、OFFBOARD 确认和 0.2 s 延时。

### 4. 先由外部轨迹接管

进入 FLIGHT 后，AUX6 默认置低，NMPC 使用 `/nmpc/in/trajectory_setpoint` 的完整外部轨迹。确认轨迹跟踪正常后，才允许切换 RC-NMPC。

## 四、AUX6 输入源切换

当前约定 `rc_aux_channel=6`，阈值 `rc_aux_enable_threshold=0.5`：

| AUX6 状态 | NMPC 参考源 | 行为 |
| --- | --- | --- |
| 低 | 外部轨迹 | 使用最新且未超时的完整轨迹。 |
| 高，RC 消息新鲜（`rc_timeout_s=0.5 s` 内） | RC-NMPC | 摇杆经过死区、加速度、速度和位置前视限制后，在节点内生成完整预测时域。 |
| 高，但 RC 消息超时 | 不接受旧摇杆 | 摇杆归零并保持当前 RC 参考；不自动改用外部轨迹。 |
| 从高切回低 | 外部轨迹 | 清除 RC 参考状态，切回最新有效外部轨迹。 |

AUX6 只切换 NMPC 的**输入源**。它不负责切换 PX4 到 OFFBOARD、解锁或上锁、启停 NMPC 节点，也不能绕过定位、轨迹新鲜度或故障门控。

ROS1 从 `/mavros/manual_control/control` 和 `/mavros/rc/in` 读取输入；ROS2 从 `/fmu/out/manual_control_setpoint` 读取输入。输入更新频率（例如 RC 注入 20 Hz）不等于 NMPC 输出频率。

## 五、每次状态更新的统一控制逻辑

ROS1/ROS2 的控制计算顺序统一为：

1. 收到状态，检查有限值和时间戳，转换到 PX4 NED/FRD；
2. 使用传感器时间间隔更新 ESO；启动后经过 `eso_activation_delay=3 s` 才启用 ESO 更新；
3. 按 AUX6 选择外部轨迹或 RC-NMPC，并校验完整时域；
4. 构造参考状态、前馈和扰动参数；
5. acados 先热启动求解；
6. 热启动失败时 reset，立即冷启动重试一次；
7. 求解成功才发布最终 body-rate/thrust，并异步写日志；
8. 目标周期是 `0.01 s`，实际频率不能高于状态输入频率。

## 六、求解失败和交回 PX4

### 单次失败

- 热启动和冷启动都失败时，只能在最近 `fallback_hold_time=0.05 s` 内短暂保持上一条有效命令；
- 超过 50 ms 不得继续发布旧的非零命令；
- 节点进程保持运行并记录失败，不因单次失败自动退出。

### 连续失败达到 3 次

当 `max_consecutive_solve_failures=3`：

1. 锁存 NMPC 故障，停止 NMPC 控制输出；
2. **ROS1 当前实现**：若 `auto_manage_flight=true`，请求 PX4 `AUTO.LOITER`；否则由操作者/外部飞行管理器切到定点保持；
3. **ROS2 当前实现**：节点停用控制，外部飞行管理器负责把 PX4 切到定点保持；
4. 不自动反复切回 OFFBOARD，不自动恢复轨迹控制。

ROS1 故障状态下会继续后台尝试保持参考求解；求解连续正常 `solver_recovery_time=2.0 s` 后，仅标记 solver 已恢复。ROS2 当前需要由外部管理器重新使能控制并执行同样的人工恢复流程，不能把“求解器恢复”当作“已经重新接管”。

## 七、故障后的人工恢复

故障后必须按以下顺序恢复：

```text
NMPC 故障锁存
  → PX4 离开 OFFBOARD
  → PX4 定点保持
  → 确认 solver 连续正常 2 s
  → 操作者再次手动选择 OFFBOARD
  → 等待 0.2 s
  → NMPC 恢复输出
```

重新进入 OFFBOARD 前再次失败，继续保持故障，不通过自动切换模式试探。AUX6 只有在 NMPC 已重新进入 FLIGHT 后才重新有效。

## 八、退出和验收口径

- 主动退出：先发布 `control_enabled=false`，确认 NMPC 不再输出，再由 PX4 执行降落或其他模式切换。
- 离开 OFFBOARD：NMPC 输出暂停；重新进入 OFFBOARD 仍需经过确认和 0.2 s 延时。
- `shadow_mode=true`：允许计算和记录，禁止真实控制输出。
- 每次回归必须记录 PX4 模式、armed、参考源、AUX6、轨迹新鲜度、稳定计时、预发送完成、OFFBOARD 确认、solver 成功/失败和故障恢复时间。

ROS1 与 ROS2 的通信映射见各自 README；如果修改上述任一条件，必须同步修改两套配置、本文档和回归测试，不能只改一端。
