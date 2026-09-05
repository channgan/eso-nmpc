# NMPC 时间同步工作记录

更新时间：2026-08-30  
适用分支：`test/no-nmpc-jerk`

## 目标

统一 PX4/uXRCE、ROS、上位机 NMPC 求解和控制输出之间的时间含义，避免 WSL
系统时间跳变、DDS 排队或求解耗时导致状态、轨迹和控制命令错位。

## 当前实现

1. 控制定时器使用 ROS `STEADY_TIME`，控制/心跳周期为 `0.01 s`（100 Hz），所有控制周期基于 `CLOCK_MONOTONIC`；MPC 离散和轨迹采样步长为 `0.02 s`。
2. PX4 输出状态优先使用 `VehicleOdometry.timestamp_sample`，回退时使用
   `VehicleOdometry.timestamp`。
3. 根据 `timestamp - timestamp_sample` 估计 PX4 内部采样年龄，并将采样时间
   映射到上位机单调时钟；无效年龄会计数并回退到接收时间。
4. NMPC 只对新的 PX4 状态采样时间求解，重复状态不会重复求解。
5. 轨迹 elapsed time 使用 PX4 状态采样时间差推进；负跳变或超过 100 ms 的
   时间间隔替换为 nominal `sample_time=0.02 s`，并记录跳变次数。
6. 输出给 PX4 的 `OffboardControlMode`、`VehicleRatesSetpoint`、命令和参考消息，
   使用“启动时同步时间戳 + 单调时钟增量”生成时间戳。
7. 实际控制命令发布时间单独记录为单调时钟，用于 command-to-measurement 延迟分析。
8. PX4 平滑轨迹和 direct 完整轨迹均检查本机单调时钟下的接收年龄。
9. direct 轨迹接口采用 `KEEP_LAST + depth=1 + BEST_EFFORT`，只保留最新轨迹，
   避免可靠队列积压旧 horizon。
10. direct 轨迹仍保留 PX4 同步域 `timestamp` 硬校验；本机接收年龄和消息时间戳
    均通过才允许进入 NMPC。
11. 模型验证分别保存 PX4 状态时间、单调时钟测量时间和控制命令发布时间，
    不再假设状态采样和命令输出同时发生。
12. 延迟分析对命令进行时间插值并搜索最佳 command-to-rate 延迟；当前三扰动 C++
    日志初步得到 roll 约 `0.131--0.216 s`（中位数约 `0.151 s`），pitch 约
    `0.216--0.307 s`，yaw 激励不足不能可靠辨识。该结果是“命令发布到
    VehicleOdometry 角速度反馈”的有效延迟，不能直接等同于 PX4 纯速率 PID 延迟。

## 已验证结果

最新四项基线目录：
`background/baseline_runs/20260830_113020/`

- 悬停、1 m 指点、圆形、8 字均完整执行并通过。
- `timestamp_sample_age_invalid_count = 0`。
- direct 轨迹消息时间戳年龄约 20--22 ms，低于 0.20 s 超时阈值。
- 修复前圆形和 8 字曾因 DDS 旧轨迹积压触发
  `complete NMPC trajectory timestamp is outside the allowed age`；改为深度 1
  最新优先后不再复现。

## 已知问题 / 真机 Debug 检查项

1. 尚未运行时订阅并监控 PX4 `TimesyncStatus`，当前依赖 uXRCE 的时间转换和
   启动时钟锚定；真机需要确认同步偏移和健康度。
2. PX4 里程计仍可能出现时间跳变，目前只做检测、计数和 nominal 步长替换，
   没有恢复真实时间。
3. `rate_tau=0.15 s` 是 SITL 辨识值，真机需用日志重新辨识。
4. 外部 direct 规划器必须发布 PX4 同步时间域的 `timestamp`；若时间域不一致，
   应先检查 uXRCE/ROS 时间同步，不能直接放宽超时校验。
5. 真机复盘优先查看：状态采样时间、状态接收时间、命令发布时间、轨迹序号、
   timestamp 年龄、sample-to-command latency、时间跳变计数和求解耗时。
6. 当前 SITL ULog 已记录 `vehicle_angular_velocity`，但未记录
   `vehicle_rates_setpoint`；若要隔离 PX4 速率内环本体，必须补记该 setpoint 及其
   PX4 时间戳，再与角速度做同域辨识。当前有效延迟辨识报告为
   `background/analysis/rate_loop_delay_20260903_230000/rate_loop_delay.md`。

## 关联代码

- `integration/px4_sitl_hover.py`：控制时钟、PX4 时间戳、轨迹新鲜度和日志。
- `nmpc/validation.py`：采样/测量/控制三种时间及延迟对齐分析。
- `config/nmpc.yaml`：`sample_time`、`reference_timeout` 和 `rate_tau`。
