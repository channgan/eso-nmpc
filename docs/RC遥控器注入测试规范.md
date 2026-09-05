# C++ RC 遥控器注入测试规范

本文是 ROS 2 + C++ NMPC 的 RC-NMPC 注入测试规范。RC 测试验证遥控器参考源切换、
摇杆输入映射、松杆保持和 RC 超时行为；它是单案例接口测试，不替代四项
`hover`、`point_1m`、`circle`、`figure8` 回归，也不更新四项 best。

## 1. 测试职责与配置

- 正式控制节点必须是持续运行的 C++ `eso_nmpc_node`；Python 只负责 RC 注入、飞行监督
  和结果整理。
- 当前部署约定使用 AUX6；C++ 节点和注入器的 AUX 通道必须一致。
- 生产默认读取 `/fmu/out/manual_control_setpoint`；SITL 注入使用独立的
  `/nmpc/test/manual_control_setpoint`，避免 PX4 原生发布者覆盖测试输入。
- 控制周期和 MPC/轨迹采样步长均为 `0.01 s`（100 Hz），时域为 `N=30`，solver hash 为 `1c2d851e`。
- 默认先使用无风、无重心偏置的 nominal SITL；RC 测试不与四项性能 RMSE 混用。

## 2. 改动检查和启动顺序

每次修改 C++ 节点、RC 注入器、节点 YAML、PX4/Gazebo 配置或 solver 后，先按
[`docs/四项回归操作流程.md`](四项回归操作流程.md) 第 1 节判断是否需要重新生成 solver、
重建 C++ 节点和执行环境检查。至少确认：

```bash
cd /home/cy/eso_nmpc
git status --short --branch
git diff --check
```

外部服务仍按固定顺序启动：

1. `MicroXRCEAgent udp4 -p 8888`；
2. Gazebo server；
3. `HEADLESS=1 PX4_SIM_SPEED_FACTOR=1.0 make px4_sitl gz_x500`，等待 `Ready for takeoff!`。

一轮测试只允许一套 PX4、一套 Gazebo 和一个 Agent。测试脚本启动并关闭 C++ 节点、
RC 注入器和监督器，不要手动重复启动这些进程。

RC 正式测试也必须遵守完整重启规则：先关闭上一轮 PX4、Gazebo、Agent，确认进程和端口
释放，再启动一套全新的外部服务。脚本的服务检查只能确认服务存在，不能证明服务是新启动
的；已有服务复用只能用于诊断，不得作为正式通过结果。

## 3. 正常接管测试

起飞后，注入器保持 AUX6 低位；飞行器进入 `OFFBOARD` 并解锁后，等待 1 s，再将 AUX6
拉高接管 RC-NMPC，前推 3 s，随后松杆验证减速和位置保持：

```bash
python3 integration/run_cpp_rc_injection.py \
  --rc-aux-channel 6 \
  --output-directory background/sitl_regression_cpp/rc_aux6_hover_<日期>
```

这里的前推参考由 C++ 节点根据摇杆生成，不是外部 `NmpcTrajectorySetpoint`；AUX 低位时
才使用外部轨迹，AUX 高位时切换为 RC-NMPC。

## 4. RC 超时失效保护测试

正常接管测试通过后，使用 `--drop-after` 将独立注入帧标记为无效，验证 RC 超时后立即锁存
关闭 NMPC 输出，并由外部飞行管理器请求 PX4 进入 AUTO.LOITER 定点保持：

```bash
python3 integration/run_cpp_rc_injection.py \
  --rc-aux-channel 6 \
  --drop-after 3.0 \
  --output-directory background/sitl_regression_cpp/rc_aux6_timeout_<日期>
```

## 5. 完成判据

正常接管和超时失效保护分别判定，只有满足以下条件才算该项测试通过：

- `landed_disarmed=true`，且没有触发飞行安全边界；
- `rc_active_sample_count > 0`，日志出现 `RC-NMPC reference enabled (AUX<n>)`；
- 每个被 C++ 节点接受的 ManualControlSetpoint 必须是 `valid=true` 且
  `data_source=SOURCE_RC`；AUX6 高但来源错误、无效或超时都进入 RC 失效保护；
- `solve_failure_count == 0`；
- 超时测试日志出现 `RC input timeout; NMPC output latched off`；
- 超时测试收到 `/nmpc/rc_timeout=true`，监督器确认 PX4 进入 AUTO.LOITER/Position mode，或由
  PX4 已配置的 RC-loss Land failsafe 接管；
- 超时后不再发布 NMPC `VehicleRatesSetpoint`，不自动重新进入 Offboard 或重新使能 NMPC；
- 测试结束时可由监督器显式执行降落并确认 `landed_disarmed=true`；
- RC 测试的运动参考由摇杆生成，因此监督器的 hover 轨迹 RMSE 不作为四项轨迹指标，
  也不作为 RC 测试的通过条件，只检查接管、保持、安全降落和求解状态；该 RMSE 仅作诊断；
  RC 轨迹图必须使用 C++ 节点的
  `nmpc_flight.csv`，不能使用监督器的 hover `trajectory.csv`；
- 报告中记录 AUX 通道、RC 超时设置、是否停止注入、C++ 节点路径、solver hash、
  控制周期、时域和参数快照。

## 6. 结果和日志保留

结果目录 `background/sitl_regression_cpp/rc_aux6_*/` 保留：

- `nmpc_flight.csv`、`nmpc_timing.csv`；
- `trajectory.png`、`controller_timing.png`、`summary.md`；
- `cpp_params.yaml` 参数快照。

机器报告统一保存到 `background/json/cpp_rc_injection_<run_id>.json`。临时
`cpp_node.log`、`rc_injector.log`、`supervisor.log` 和 PX4 ULog 保存到：

```text
/tmp/eso_nmpc_sitl_logs/<结果目录名>/
```

RC 测试不更新 `background/best_sitl_history/`。测试完成后按
PX4 → Gazebo → Agent 顺序关闭外部服务，并检查没有残留进程和端口。

## 7. 2026-09-04 RC 逻辑修复验证

本次修复包括：

- C++ 节点只接受 `data_source=SOURCE_RC` 的有效手动控制消息；
- AUX6 高且 RC 从未有效、消息无效或消息超时，均触发统一 NMPC 失效保护；
- RC 超时会锁存关闭 NMPC、清除 warm start、发布 `/nmpc/rc_timeout=true`，不改用外部轨迹；
- 监督器在飞行前收到 RC 故障也会立即关闭 NMPC 并进入安全降落流程；
- RC 接管、超时 watchdog、Acados 求解和 warm-start 状态访问已避免线程并发冲突。

按本规范逐项重启完成两项验证：

| 场景 | 结果 | RC 有效样本 | RC 超时 | Position/Hold fallback | 求解失败 | 安全落地解锁 |
|---|---:|---:|---:|---:|---:|---:|
| AUX6 接管、前推、松杆保持 | `PASS` | 884 | 否 | 不适用 | 0 | 是 |
| `drop-after=3.0 s` 超时保护 | `PASS` | 200 | 是 | 是 | 0 | 是 |

正常 RC 接管时飞机会按摇杆离开外部 hover 参考，因此监督器 hover RMSE 只保留为诊断值，
不参与 RC 测试通过判定。第一次复测曾因错误沿用 hover RMSE 门限被误判，已修复测试脚本。

结果目录：

- `background/sitl_regression_cpp/rc_logic_fix_normal2_20260904/`
- `background/sitl_regression_cpp/rc_logic_fix_timeout2_20260904/`

对应机器报告为 `background/json/cpp_rc_injection_*.json`，临时日志位于
`/tmp/eso_nmpc_sitl_logs/rc_logic_fix_normal2_20260904/` 和
`/tmp/eso_nmpc_sitl_logs/rc_logic_fix_timeout2_20260904/`。
