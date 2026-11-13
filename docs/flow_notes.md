# Flow Policy 源码拆解：与 Diffusion Policy 的结构差异

Flow Matching（流匹配）是 DDPM 之后动作生成的主流路线（π0、GR00T 的动作头均为
flow matching）。本仓库的最小实现位于 [baselines/flow/](../baselines/flow/)
（`flow_model.py` 约 120 行，与 `baselines/bc` 共享观测编码骨架）。

## 1. 数学机制对比

| | Diffusion Policy（DDPM） | Flow Policy（Flow Matching） |
|---|---|---|
| 前向过程 | 按 β_t 调度表逐步加噪，100 档 | **直线插值**：x_t = (1-t)·x₀ + t·x₁，t~U[0,1] 连续 |
| 学习目标 | 预测所加噪声 ε（MSE(ε̂, ε)） | 预测速度场 v = x₁ - x₀（MSE(v̂, v)） |
| 推理 | N 步**随机**去噪（DDPM）/ 确定性跳步（DDIM） | N 步 **Euler 积分**（确定性 ODE） |
| 噪声调度 | 需要 beta schedule（linear/cosine） | **不需要**，无调度表 |
| 推理随机性 | 每次结果不同（多样性好） | 起点固定则结果确定（可复现） |

核心直觉：Diffusion 学"每一步往哪个方向去掉一点噪声"；Flow 直接学"从噪声到
数据的整条直线怎么走"——回归目标更直接，低步数下更稳。

## 2. 代码结构对照

| 模块 | Diffusion（lerobot） | Flow（本仓库最小实现） |
|---|---|---|
| 观测编码 | `DiffusionRgbEncoder`（ResNet18+SpatialSoftmax） | 同 BC v2 的 ResNet18 编码器 |
| 时间嵌入 | `DiffusionSinusoidalPosEmb` + MLP | `TimeEmbedding`（同构，输入 t∈[0,1] 而非整数档位） |
| 去噪/速度网络 | 1D 卷积 U-Net（251M，3 级下采样） | `VelocityField`：2 层 1D 卷积 + FiLM 时间条件（0.3M） |
| 采样器 | `noise_scheduler.step()`（DDPM/DDIM） | Euler：`x += Δt · v(x,t)` |
| 训练损失 | `F.mse_loss(pred, eps)` | `F.mse_loss(v_pred, x1 - x0)` |

结构上 Diffusion 的"调度器 + U-Net"两件套，在 Flow 里坍缩成"一个速度场网络"——
代码量、超参数（beta_start/beta_end/schedule 全部消失）、推理复杂度都更小。

## 3. 完整基线（与 Diffusion 严格同协议，7680 步）

见 [experiments/flow_baseline/](../experiments/flow_baseline/RESULTS.md)：
Flow 2%（1/50）vs Diffusion 6%（3/50）；Flow 以 1/500 参数、1/10 显存拿到同量级行为，
推理快 137 倍。结论：极限质量选 Diffusion，工程性价比选 Flow。

## 4. 实测数据（PushT 最小 demo，2000 步）

- 训练：val loss 0.549（epoch 0）→ 0.303（epoch 3），全程平稳无异常；
- 模型参数量 11.5M（观测编码 11.2M + 速度场仅 0.3M）；
- 闭环评估（10 episodes，开环执行 8 步块 + 重规划）：0/10 成功，
  但平均峰值覆盖率 **0.44**——参照系：ACT 训 7.7k 步（3.9 倍预算）才到 0.47，
  flow matching 的收敛速度优势肉眼可见；
- 推理（实测）：10 步 Euler 积分单次重规划 **9.0 ms**——与 ACT（10.3ms）同数量级，
  比 Diffusion DDPM-100（1235ms）快 137 倍，比 DDIM-5（51ms）还快 5.7 倍。

## 4. 选型备注

- 需要动作多样性（多次采样选优）→ Diffusion（随机采样是特性）；
- 需要低延迟、确定性、快速收敛 → Flow（π0/GR00T 的选择）；
- 两者共享"迭代精修动作块"的失效模式：到位后保持、开环块内纠偏，
  缓解方式相同（缩短执行步数、增加重规划频率）。


## 5. 定向对比实验（推理步数消融）

Flow 的推理步数不是越多越好：实测 1 步 Euler 反而优于 10 步（峰值覆盖率 0.727 vs 0.549），
单步成本 0.567 ms 是 Diffusion 的 1/16。完整步数消融、训练稳定性量化、动作平滑度与选型判据见
[docs/flow_vs_diffusion.md](flow_vs_diffusion.md) 与
[experiments/flow_vs_diffusion/](../experiments/flow_vs_diffusion/RESULTS.md)。

## 6. 失效模式与双臂迁移

Flow 1 步为何在部分初态满分、另一些初态几乎不动，见
[failure_mode_analysis.md](failure_mode_analysis.md) 的同种子 1/10 步对照；
14 维双臂动作空间与 50 Hz 预算对 Flow 的影响见 [aloha_dataset_probe.md](aloha_dataset_probe.md)。
