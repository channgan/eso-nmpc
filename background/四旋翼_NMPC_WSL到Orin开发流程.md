# 基于 acados 的四旋翼 NMPC 开发流程说明

## 1. 项目目标

本项目采用如下控制架构：

\[
[p,v,q]
\rightarrow
\mathrm{ESO}
\rightarrow
\mathrm{NMPC}
\rightarrow
[T,\omega_{sp}]
\rightarrow
\mathrm{PX4\ Rate\ PID}
\rightarrow
\mathrm{Control\ Allocator}
\rightarrow
\mathrm{Motor}
\]

其中：

- NMPC 运行于上位机；
- 上位机目标平台为 NVIDIA Jetson Orin NX；
- 通信框架采用 ROS2；
- NMPC 控制频率目标为 100 Hz；
- NMPC 输出：
  - 总推力 \(T\)
  - 机体系角速度设定值 \(\omega_{sp}=[p_{sp},q_{sp},r_{sp}]^T\)
- PX4 保留：
  - 角速度控制器；
  - 控制分配器；
  - 电机输出；
- PX4 原生位置、速度、姿态角控制层不参与 NMPC 模式下的主控制；
- 后续使用 ESO 对外部扰动和模型失配产生的等效平动扰动进行估计，并将估计值直接写入 NMPC 预测模型。

---

# 2. 推荐开发顺序

第一阶段不直接在 Orin NX 真机环境开发，而是在 WSL 中完成 NMPC 算法主体。

推荐流程：

```text
WSL
│
├── CasADi 建立非线性动力学模型
│
├── acados 定义 OCP
│
├── acados 生成 C solver
│
├── 单次求解测试
│
├── 离线闭环仿真
│
├── 轨迹跟踪测试
│
└── solver 求解时间 benchmark
│
└─────────────── 验证算法正确
                  ↓
Orin NX
│
├── 重新生成 / 编译 ARM64 solver
├── ROS2 C++ NMPC 节点
├── PX4 VehicleOdometry 输入
├── VehicleRatesSetpoint 输出
├── 100 Hz 实时线程
├── ESO
└── 真机测试
```

核心原则：

\[
\boxed{\text{WSL 负责把算法写对，Orin NX 负责把算法跑实时}}
\]

---

# 3. WSL 阶段需要完成的工作

WSL 阶段建议只关注以下四件核心工作：

\[
\boxed{
\mathrm{CasADi建模}
\rightarrow
\mathrm{acados\ OCP}
\rightarrow
\mathrm{solver\ code\ generation}
\rightarrow
\mathrm{离线闭环仿真}
}
\]

暂时不处理：

- PX4 DDS 通信；
- Orin NX 实时线程；
- CPU affinity；
- PREEMPT_RT；
- LIO 接入；
- 真机推力映射；
- 电池电压影响；
- 实际风扰动；
- 真机 ESO 调参。

这样可以避免将算法问题和平台问题混在一起。

---

# 4. NMPC 状态定义

第一版状态固定为：

\[
x=
\begin{bmatrix}
p\\
v\\
q
\end{bmatrix}
\]

展开为：

\[
x=
[p_x,p_y,p_z,
v_x,v_y,v_z,
q_w,q_x,q_y,q_z]^T
\]

因此：

\[
n_x=10
\]

其中：

- \(p\)：世界坐标系位置；
- \(v\)：世界坐标系速度；
- \(q\)：机体坐标系到世界坐标系的姿态四元数。

推荐 NMPC 内部直接采用 PX4 的：

- 世界系：NED；
- 机体系：FRD。

这样后续接 PX4 时可以减少坐标系转换错误。

---

# 5. NMPC 控制量定义

控制量定义为：

\[
u=
\begin{bmatrix}
T\\
\omega_{sp}
\end{bmatrix}
\]

即：

\[
u=
[T,\omega_x^*,\omega_y^*,\omega_z^*]^T
\]

因此：

\[
n_u=4
\]

其中：

- \(T\)：总推力，建议 NMPC 内部使用物理单位 N；
- \(\omega_x^*\)：期望滚转角速度；
- \(\omega_y^*\)：期望俯仰角速度；
- \(\omega_z^*\)：期望偏航角速度。

NMPC 不直接输出：

- 姿态角；
- 力矩；
- 单电机推力。

PX4 原生角速度环负责将：

\[
\omega_{sp}
\]

转换为：

\[
\tau
\]

随后控制分配器转换为电机输出。

---

# 6. NMPC 动力学模型

## 6.1 位置动力学

\[
\dot p=v
\]

---

## 6.2 平移动力学

采用：

\[
\boxed{
\dot v
=
g e_3
-
\frac{T}{m}R(q)e_3
+
d
}
\]

其中：

\[
e_3=
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}
\]

在 NED 坐标系中：

\[
g e_3=
\begin{bmatrix}
0\\
0\\
g
\end{bmatrix}
\]

\(R(q)\) 表示由四元数得到的机体系到世界系旋转矩阵。

\(d\) 为预留的等效加速度扰动：

\[
d=
[d_x,d_y,d_z]^T
\]

---

## 6.3 姿态运动学

姿态四元数满足：

\[
\boxed{
\dot q=
\frac{1}{2}
q
\otimes
\begin{bmatrix}
0\\
\omega_{sp}
\end{bmatrix}
}
\]

这里采用近似：

\[
\omega\approx\omega_{sp}
\]

该近似成立的原因是：

- PX4 角速度内环运行频率高；
- NMPC 不负责电机级和力矩级快速动态；
- NMPC 将角速度闭环看作较快的低层执行动态。

---

# 7. ESO 接口必须现在预留

即使第一阶段不实现 ESO，也建议从第一版模型开始就定义：

\[
\boxed{
d=
[d_x,d_y,d_z]^T
}
\]

并作为 acados runtime parameter。

因此：

\[
\boxed{
model.p=d
}
\]

当前没有 ESO 时：

\[
d=
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
\]

以后接入 ESO 时：

\[
d=\hat d_{ESO}
\]

这样不需要修改 NMPC 的基本动力学结构。

---

# 8. 为什么 ESO 扰动不作为 NMPC 状态

第一版不采用：

\[
x=[p,v,q,d]
\]

而保持：

\[
x=[p,v,q]
\]

ESO 独立运行：

```text
PX4 / 仿真状态
      ↓
     ESO
      ↓
    d_hat
      ↓
NMPC model parameter
```

因此：

\[
d
\]

是：

\[
\boxed{\text{在线参数}}
\]

而不是：

\[
\boxed{\text{优化状态}}
\]

这样可以保持：

\[
n_x=10
\]

对 100 Hz 实时 NMPC 更有利。

---

# 9. CasADi 模型接口建议

建议在 Python 中建立：

```python
model.x = x
model.u = u
model.p = d
model.f_expl_expr = f_expl
```

结构示意：

```python
x = vertcat(
    px, py, pz,
    vx, vy, vz,
    qw, qx, qy, qz
)

u = vertcat(
    T,
    wx_sp,
    wy_sp,
    wz_sp
)

d = vertcat(
    dx,
    dy,
    dz
)
```

最终动力学：

```python
f_expl = vertcat(
    velocity,
    acceleration_world,
    q_dot
)
```

其中：

```python
acceleration_world = gravity + thrust_acceleration + d
```

---

# 10. acados OCP 配置

第一版推荐使用：

```python
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
```

对应：

| 模块 | 建议 |
|---|---|
| NLP 算法 | SQP-RTI |
| QP 求解器 | Partial Condensing HPIPM |
| 连续模型积分 | ERK |
| Hessian | Gauss-Newton |
| 运行环境 | C/C++ generated solver |

第一阶段如需检查数学问题，也可以临时使用普通 SQP 做验证。

---

# 11. 预测周期设置

最终 NMPC 目标频率：

\[
f_c=100Hz
\]

因此：

\[
T_s=0.01s
\]

建议初始测试三组 horizon：

\[
N=20
\]

\[
N=30
\]

\[
N=40
\]

分别对应：

| \(N\) | 预测时间 |
|---:|---:|
| 20 | 0.2 s |
| 30 | 0.3 s |
| 40 | 0.4 s |

第一版建议：

\[
\boxed{
N=30,\qquad T_p=0.3s
}
\]

之后根据 Orin NX 上真实 benchmark 再确定。

---

# 12. Cost Function 建议

第一版只保留必要项：

\[
J=
\sum_{i=0}^{N-1}
\left[
\|p_i-p_i^r\|_{Q_p}^2
+
\|v_i-v_i^r\|_{Q_v}^2
+
\|e_{q,i}\|_{Q_q}^2
+
\|u_i-u_i^r\|_R^2
\right]
+
J_N
\]

建议主要调：

- \(Q_p\)：位置跟踪；
- \(Q_v\)：速度跟踪；
- \(Q_q\)：姿态跟踪；
- \(R\)：控制量平滑程度。

后续可增加：

\[
\Delta u
\]

惩罚：

\[
\|\Delta u\|_{R_\Delta}^2
\]

用于限制 body-rate 和推力变化过快。

---

# 13. 约束建议

第一版至少加入：

## 13.1 推力约束

\[
T_{min}\le T\le T_{max}
\]

---

## 13.2 角速度约束

\[
|\omega_x|\le\omega_{x,max}
\]

\[
|\omega_y|\le\omega_{y,max}
\]

\[
|\omega_z|\le\omega_{z,max}
\]

---

## 13.3 后续约束

后续再逐步增加：

- 最大倾角；
- 最大速度；
- 最大推力变化率；
- 最大角速度变化率；
- 障碍物距离约束；
- 避障速度约束。

第一版不要同时加入过多 nonlinear constraints。

---

# 14. 参考轨迹接口

轨迹规划器建议提供：

\[
p_r,\quad
v_r,\quad
a_r,\quad
\psi_r
\]

通过 differential flatness 计算：

\[
q_r
\]

即：

\[
a_r+\psi_r
\rightarrow
R_r
\rightarrow
q_r
\]

最终 NMPC reference horizon 为：

\[
[p_r,v_r,q_r]
\]

预测窗口中每一个节点都应有对应参考值。

---

# 15. WSL 中第一阶段文件结构建议

```text
nmpc_project/
│
├── model/
│   └── quadrotor_model.py
│
├── solver/
│   └── generate_solver.py
│
├── tests/
│   ├── test_single_solve.py
│   ├── test_hover.py
│   ├── test_position_step.py
│   ├── test_circle.py
│   └── benchmark_solver.py
│
├── simulation/
│   └── closed_loop_sim.py
│
└── config/
    └── nmpc.yaml
```

---

# 16. WSL 第一阶段必须完成的测试

## 16.1 单次求解

给定：

\[
x_0
\]

和悬停参考：

\[
p_r=[0,0,-1]
\]

检查输出：

\[
T\approx mg
\]

以及：

\[
\omega_{sp}\approx0
\]

---

## 16.2 悬停闭环

仿真：

\[
x_{k+1}=F(x_k,u_k)
\]

检查：

- 是否稳定；
- 是否收敛；
- 四元数是否正常；
- 控制量是否饱和。

---

## 16.3 位置阶跃

例如：

\[
[0,0,-1]
\rightarrow
[1,0,-1]
\]

检查：

- 响应速度；
- 超调；
- body-rate 输出；
- 推力变化。

---

## 16.4 圆轨迹

检查：

- 位置 RMSE；
- 速度 RMSE；
- 姿态轨迹；
- 控制连续性。

---

## 16.5 人工扰动参数测试

即使 ESO 尚未实现，也应直接人为设置：

\[
d=
\begin{bmatrix}
0.5\\
0\\
0
\end{bmatrix}
m/s^2
\]

比较：

\[
d=0
\]

和：

\[
d\neq0
\]

时 NMPC 输出是否发生合理变化。

这一步可以提前验证：

\[
\boxed{
ESO\rightarrow NMPC
}
\]

接口是否正确。

---

# 17. Solver Benchmark

100 Hz 控制意味着：

\[
T_s=10ms
\]

因此 WSL 和后续 Orin NX 都需要记录：

\[
t_{mean}
\]

\[
t_{p99}
\]

\[
t_{p99.9}
\]

\[
t_{max}
\]

最终 Orin NX 实机目标应满足：

\[
\boxed{
t_{control,max}<10ms
}
\]

并建议 solver 本身保留足够裕度，例如：

\[
t_{solver,p99.9}<3\sim4ms
\]

实际数值需要在 Orin NX 全负载条件下测试。

---

# 18. 推力模型单独设计

NMPC 内部建议输出：

\[
T[N]
\]

PX4 接口需要归一化推力，因此中间单独建立：

```text
NMPC
 ↓
T [N]
 ↓
ThrustModel
 ↓
normalized thrust
 ↓
VehicleRatesSetpoint
```

不要把推力归一化映射直接写死在 NMPC 中。

以后可以扩展为：

- 固定推力映射；
- RLS 在线辨识；
- 电池电压补偿；
- 桨效率补偿。

---

# 19. Orin NX 阶段架构

WSL 验证完成后迁移到 Orin NX。

推荐结构：

```text
PX4 VehicleOdometry
          ↓
     Latest State Buffer
          ↓
100 Hz NMPC Dedicated Thread
          ↓
       ESO Update
          ↓
        d_hat
          ↓
Reference Horizon Sampling
          ↓
    acados C Solver
          ↓
      T + body_rate_sp
          ↓
 VehicleRatesSetpoint
          ↓
         PX4
          ↓
   mc_rate_control
          ↓
 control_allocator
          ↓
        motors
```

---

# 20. ROS2 实时运行建议

最终 Orin NX 版本建议：

- NMPC 使用独立控制线程；
- 控制线程目标 100 Hz；
- solver 使用生成的 C/C++；
- 不在 Python runtime 中实时求解；
- 状态订阅和轨迹订阅只更新 buffer；
- NMPC 线程读取最新数据；
- LIO、地图、视觉等尽量不与 NMPC 争抢同一 CPU 核。

后续如需要更严格实时性，可考虑：

- CPU affinity；
- `SCHED_FIFO`；
- `mlockall()`；
- performance governor；
- PREEMPT_RT；
- 控制线程独占核心。

---

# 21. ESO 后续接入方式

ESO 第一版只估计三维平动等效扰动：

\[
\hat d_a=
[\hat d_x,\hat d_y,\hat d_z]^T
\]

控制周期：

```text
x_k
 ↓
ESO(x_k, u_{k-1})
 ↓
d_hat_k
 ↓
NMPC parameter update
 ↓
solve
 ↓
u_k
```

预测窗口采用：

\[
\boxed{
\hat d_{k+i|k}=\hat d_k
}
\]

即当前 disturbance estimate 在一个 horizon 内保持常值。

---

# 22. 最终阶段划分

## 阶段 A：WSL

目标：

\[
\boxed{\text{证明 NMPC 数学和 solver 正确}}
\]

完成：

- CasADi 模型；
- acados OCP；
- C solver 生成；
- hover；
- step；
- circle；
- 人工 disturbance；
- benchmark。

---

## 阶段 B：Orin NX 离线

目标：

\[
\boxed{\text{证明 100 Hz 实时性能}}
\]

完成：

- ARM64 编译；
- C++ solver wrapper；
- 100 Hz thread；
- timing benchmark；
- CPU 调度优化。

---

## 阶段 C：PX4 SITL

目标：

\[
\boxed{\text{证明 NMPC → body-rate → PX4 rate loop 数据链正确}}
\]

完成：

- VehicleOdometry；
- VehicleRatesSetpoint；
- thrust mapping；
- NED/FRD；
- Offboard mode；
- hover；
- trajectory tracking。

---

## 阶段 D：真机纯 NMPC

暂时：

\[
d=0
\]

先完成：

- 悬停；
- 小范围位置阶跃；
- 低速轨迹；
- 100 Hz 稳定运行。

---

## 阶段 E：ESO-NMPC

加入：

\[
d=\hat d_{ESO}
\]

测试：

- 恒定风；
- 突风；
- 质量变化；
- 推力模型误差；
- 轨迹跟踪。

最终进行：

\[
\boxed{
PX4
\quad vs\quad
NMPC
\quad vs\quad
ESO-NMPC
}
\]

对比。

---

# 23. 当前最重要的第一步

当前应先在 WSL 中完成：

\[
\boxed{
x=[p,v,q]
}
\]

\[
\boxed{
u=[T,\omega_{sp}]
}
\]

\[
\boxed{
p_{model}=d_{ESO}
}
\]

其中暂时：

\[
d_{ESO}=0
\]

然后完成：

\[
\boxed{
CasADi
\rightarrow
acados OCP
\rightarrow
C solver
\rightarrow
离线闭环仿真
}
\]

只要这一阶段跑通，后续 Orin NX、ROS2、PX4、ESO 都是在这个核心控制器外围逐步接入，而不需要重新推翻 NMPC 主体。
