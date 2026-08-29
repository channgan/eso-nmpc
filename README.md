# 四旋翼 NMPC（acados）

这是设计文档中“WSL 阶段”的可执行实现。控制器使用 PX4 一致的 NED 世界系和 FRD 机体系：

```text
x = [p_NED(3), v_NED(3), q_body_to_NED(wxyz)]
u = [T_N, body_rate_sp(3)]
p_model = d_hat_NED(3)
```

连续模型为：

```text
p_dot = v
v_dot = [0, 0, g] - T/m * R(q)[:, 2] + d_hat
q_dot = 0.5 * q ⊗ [0, body_rate_sp]
```

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

x = np.r_[position_ned, velocity_ned, quaternion_wxyz]
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

节点会依次等待估计器稳定、预发送 Offboard 心跳、切换 Offboard、解锁、执行平滑
起飞/轨迹/下降，最后请求 PX4 Land 并确认解锁。圆轨迹的进入和退出段同时连续匹配
位置、速度与加速度。无地面站的纯 headless SITL 需要临时将 `NAV_DLL_ACT` 设为
`0`；测试结束后恢复原值。

## 当前边界

- 已实现第一阶段 NMPC 数学核心、扰动参数接口、C solver 生成配置、离线模型/接口测试与 benchmark。
- 已包含 PX4 x500 SITL 的 ROS 2 悬停/圆轨迹验收节点和线性悬停点推力归一化；该映射仅用于当前 x500 仿真。
- 尚未包含 ESO 和旋转动力学；角速度由 PX4 内环跟踪。接真机前必须重新测量质量并标定推力模型。
