# 四旋翼 NMPC（acados）

这是设计文档中“WSL 阶段”的可执行实现。控制器使用 PX4 一致的 NED 世界系和 FRD 机体系：

```text
x = [p_NED(3), v_NED(3), q_body_to_NED(wxyz), actual_body_rate(3)]
u = [T_N, body_rate_sp(3)]
p_model = d_hat_NED(3)
```

连续模型为（角速度指令经一阶滞后作用到实际角速度，见 `rate_tau`）：

```text
p_dot = v
v_dot = [0, 0, g] - T/m * R(q)[:, 2] + d_hat
q_dot = 0.5 * q ⊗ [0, actual_body_rate]
actual_body_rate_dot = (body_rate_sp - actual_body_rate) / rate_tau
```

速率滞后状态让 OCP 可以“看穿” PX4 角速度环的指令延迟（SITL 实测
0.20～0.21 s），否则悬停时会与 +/-0.8 rad/s 的速率包线形成自持极限环。

目标部署后端是 acados 的 `SQP_RTI + PARTIAL_CONDENSING_HPIPM + ERK + GAUSS_NEWTON`。仓库还提供一个较慢的 SciPy direct-shooting 后端，只用于没有 acados 时验证模型和接口，不能用于 100 Hz 部署。

## 目录

```text
config/nmpc.yaml                  质量、时域、权重和输入约束
nmpc/model/quadrotor.py           NumPy 动力学与 CasADi/acados 模型
nmpc/reference.py                 悬停参考、平坦性姿态/推力映射
nmpc/solver/acados_solver.py      OCP 和实时控制器包装
nmpc/solver/scipy_solver.py       离线验证后端
solver/generate_solver.py         C solver 生成入口
simulation/closed_loop_sim.py     悬停到位置阶跃的闭环仿真
simulation/trajectory_tracking_sim.py  平滑圆轨迹闭环跟踪与验收
tests/benchmark_solver.py         mean/p99/p99.9/max benchmark
```

## 先运行不依赖 acados 的检查

```bash
python3 -m pytest
python3 simulation/closed_loop_sim.py --backend scipy --duration 0.3 --step-time 0.1
```

设置好 `ACADOS_SOURCE_DIR` 后，同一个测试命令还会自动执行原生 solver 的悬停和扰动参数集成测试；未安装 acados 时该测试会跳过。

SciPy 后端对 30 步、120 个决策变量使用数值优化，速度会明显慢于实时；短时仿真时可在配置副本中减小 `horizon_steps`。

## 安装并生成 acados solver

先按 [acados 官方安装说明](https://docs.acados.org/installation/) 编译 acados（HPIPM 是默认依赖），再以 editable 模式安装仓库内的 Python 模板。典型环境设置如下，其中路径按本机安装位置调整：

```bash
git clone https://github.com/acados/acados.git /path/to/acados
git -C /path/to/acados submodule update --recursive --init
cmake -S /path/to/acados -B /path/to/acados/build -DBUILD_SHARED_LIBS=ON
cmake --build /path/to/acados/build --target install -j

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e /path/to/acados/interfaces/acados_template

export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib"
```

然后从仓库根目录生成并编译：

```bash
python3 solver/generate_solver.py
```

运行原生 solver 闭环和时延测试：

```bash
python3 simulation/closed_loop_sim.py --backend acados --duration 2.0
python3 simulation/trajectory_tracking_sim.py --duration 26 --strict
python3 tests/benchmark_solver.py --iterations 2000 --warmup 100
```

圆轨迹验收会同时检查 solver 状态、有限数、四元数范数、位置/速度/姿态误差、
输入约束与饱和比例，以及 100 Hz 控制周期下的 p99 和最大求解时间。可以通过
`--disturbance-x 0.5` 添加恒定扰动，并通过 `--estimate-disturbance-x` 单独设置
送入 NMPC 的扰动估计。

生成目录由 `config/nmpc.yaml` 的 `code_generation` 配置，默认在 `generated/` 下。迁移到 Orin NX 时，应在 ARM64 目标机上使用相同配置重新生成、编译，不要复制 x86 动态库。

## 控制循环接口

```python
import numpy as np

from nmpc.config import load_config
from nmpc.reference import stationary_reference
from nmpc.solver.acados_solver import AcadosNmpc

cfg = load_config()
controller = AcadosNmpc(cfg)

x = np.r_[position_ned, velocity_ned, quaternion_wxyz, body_rate_xyz]
d_hat = np.zeros(3)  # 后续替换为 ESO 的世界系等效加速度估计
ref = stationary_reference(np.array([0.0, 0.0, -1.0]), cfg.controller.horizon_steps, cfg.hover_thrust)
command = controller.solve(x, ref, d_hat)

thrust_newton = command.thrust
body_rate_sp = command.body_rate
```

`thrust_newton` 必须经过独立的推力模型转换后再写入 PX4 归一化 thrust；不要把牛顿值直接发送给 `VehicleRatesSetpoint`。参考四元数在每次求解前自动与当前姿态对齐符号，输入四元数也会归一化。

## PX4 x500 SITL 轨迹验收

`config/nmpc.yaml` 已按 Gazebo x500 的 `2.0643076923 kg` 总质量配置。直接发布
`VehicleRatesSetpoint` 时的实测悬停输入为 `0.73`，归一化范围为 `0.12～1.00`；
它不同于 PX4 位置控制器使用的 `MPC_THR_HOVER=0.60`。
启动 PX4、MicroXRCEAgent 并 source ROS 2 工作区后运行。若 WSL/Gazebo
仿真时钟偶发追赶，可将 SITL 固定为半速：

```bash
HEADLESS=1 PX4_SIM_SPEED_FACTOR=0.5 make px4_sitl gz_x500
```

悬停测试：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ACADOS_SOURCE_DIR=/home/cy/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/home/cy/eso_nmpc:$PYTHONPATH
.venv/bin/python integration/px4_sitl_hover.py --altitude 1.0
```

起飞后连续进入半径 0.5 m、速度 0.25 m/s 的圆轨迹，运行一圈再返回并降落：

```bash
.venv/bin/python integration/px4_sitl_hover.py \
  --trajectory circle --altitude 1.0 --radius 0.5 --speed 0.25
```

模型与 SITL 的定量对照可在悬停或圆轨迹命令后增加 `--validate-model`：

```bash
.venv/bin/python integration/px4_sitl_hover.py \
  --trajectory circle --altitude 1.0 --radius 0.5 --speed 0.25 \
  --validate-model
```

最终的 `NMPC_SITL_RESULT.model_validation` 会报告实测角速度与指令的误差、平动
加速度残差，以及 0.01、0.1、0.5 秒开环预测的位置、速度和姿态误差。超过
0.1 秒的里程计时间戳间断会自动剔除，不参与模型误差统计。

每次修改模型、控制器、PX4 PID、滤波、控制频率或推力映射后，必须运行固定回归集：
定点悬停、PX4 平滑后的点到点位置指令、圆轨迹和 8 字航线。原始位置硬阶跃不用于
部署调参与验收。点到点用例使用 PX4 `PositionSmoothing` 输出的完整 NMPC 预测时域，
再由上位机完成逆动力学参考转换和求解。

参考接口分为两条显式链路。默认 `--reference-source px4-smoothed` 将 Offboard 目标点送入
PX4 平滑器，再订阅 `/fmu/out/nmpc_trajectory`；`--reference-source direct` 从
`/nmpc/in/trajectory_setpoint` 接收 `NmpcTrajectorySetpoint`，供已经完成时间参数化的
完整轨迹直接进入 NMPC。基线节点在 direct 模式会用同一话题发布测试轨迹，之后可直接
替换为外部规划器。两条链路不会自动猜测或混用，最终统一经过
`KinematicTrajectory` 的时域、有限值和运动包线检查。

自动基线固定运行四项：悬停与 1 m 多方向点到点使用 PX4 指点平滑接口，圆形与
8 字使用完整轨迹接口：

```bash
.venv/bin/python integration/run_sitl_regression.py
```

脚本逐项安全起降，并写入
`background/baseline_runs/<timestamp>/<case>/trajectory.csv`、`summary.json` 和
`run.log`。测试组根目录同时生成 `suite_summary.json` 与便于直接阅读的
`suite_summary.md`；单项失败后仍继续运行其余用例，以保留完整基线结果。

### 一键运行的前提：三个 SITL 服务

套件**不会**自己启动或重启服务，运行前先拉起（开跑时自动检查，缺失会
打印确切的启动命令并以退出码 2 中止）：

```bash
# 终端 1：Gazebo（PX4 官方 default.sdf 世界）
gz sim --verbose=1 -r -s <px4树>/Tools/simulation/gz/worlds/default.sdf

# 终端 2：PX4 SITL（保持该终端存活；make 会随源码变更重新编译）
cd <px4树> && make px4_sitl gz_x500

# 终端 3：MicroXRCE-DDS 代理
MicroXRCEAgent udp4 -p 8888
```

注意：若 PX4 已在运行但 MAVLink 长时间无客户端（例如跑过一次套件后隔了很久），
其 14580 流可能固定到已失效的客户端地址而不再应答 —— 检查会提示
`restart it`，重启 `make px4_sitl gz_x500` 即可。套件自身通过**一条持久
MAVLink 连接**完成全部参数读写，不会制造这个状态。

### 参数守卫

套件通过 MAVLink（udp:14580）自动设置飞行依赖的 PX4 参数并在结束时恢复：
`SIM_BAT_DRAIN=0`（防止长时仿真电池耗尽触发失效保护）、`NAV_DLL_ACT=0`
（无地面站的 headless SITL 会因数据链丢失中止任务）。`--skip-params`
可跳过守卫，`--skip-service-check` 跳过服务检查；单用例超时默认 300 s
（`--case-timeout`），超时或 Ctrl-C 均会终止子进程并保留已有结果
（退出码 130）。单独跑一个用例：

```bash
.venv/bin/python integration/run_sitl_regression.py --cases point_1m
```

### 运行测试

ROS 2 工作区 source 之后，系统里的 ROS pytest 插件（`launch_testing`）会与
仓库 venv 的 pytest 冲突，测试请带环境变量：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

`tests/fake_px4.py` 提供进程内假 PX4（真实 MAVLink 报文编解码），参数守卫与
服务探测的握手行为（先发心跳再收流、读改写校验、退出恢复）均有单测覆盖。

节点会依次等待估计器稳定、预发送 Offboard 心跳、切换 Offboard、解锁、执行平滑
起飞/轨迹/下降，最后请求 PX4 Land 并确认解锁。圆轨迹的进入和退出段同时连续匹配
位置、速度与加速度。

## 当前边界

- 已实现第一阶段 NMPC 数学核心、扰动参数接口、C solver 生成配置、离线模型/接口测试与 benchmark。
- 已包含 PX4 x500 SITL 的 ROS 2 悬停/圆轨迹验收节点和线性悬停点推力归一化；该映射仅用于当前 x500 仿真。
- 尚未包含 ESO；角速度以 PX4 内环加一阶滞后建模（`rate_tau`）。接真机前必须重新测量质量并标定推力模型。
