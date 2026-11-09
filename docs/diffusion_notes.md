# Diffusion Policy 源码拆解笔记

源码：`lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py`（853 行）
配置：`configuration_diffusion.py`（v0.7.1 已无独立 yaml，配置项与含义一一对应）。
形状均用官方 `diffusion_pusht` checkpoint 实测验证。

## 1. 网络架构图

```
训练/推理公共的观测编码（与 ACT 同源）
  图像 (B, S=2, N=1, 3, 96, 96)            状态 (B, S=2, 2)
        │ DiffusionRgbEncoder ×N相机               │
        │ ResNet18 → SpatialSoftmax(32关键点)      │
        │ → MLP → 64 维/相机 (11.2M 参数)          │
        └──────────────┬───────────────────────────┘
                       │ concat 后展平
              global_cond (B, 2×(64+2)=132)   ← 观测条件

扩散去噪网络 DiffusionConditionalUnet1d（251.5M 参数，down_dims 512/1024/2048）
  输入: 带噪动作序列 x_t (B, horizon=16, 2)
  条件: global_cond ──FiLM──┐
  时间步: t ─SinusoidalPosEmb→ MLP → 128 维时间嵌入 ─┤
        │                                              │
        └─ 1D 卷积 U-Net（3 级时序下采样 16→8→4→2）──┘
                     │
  输出: 预测噪声 ε̂ (B, 16, 2)   ← 与 x_t 同形状

推理采样（conditional_sample）
  x_T ~ N(0, I) → for t in 推理步数: ε̂=U-Net(x_t,t,cond); x_{t-1}=scheduler.step(...)
  → 取后 n_action_steps=8 步执行（前 2 步对齐观测历史）
```

## 2. 四个模块的输入输出

| 模块 | 输入 | 输出 | 源码位置 |
|---|---|---|---|
| 观测编码 | 图像 (B,S,N,3,H,W)、状态 (B,S,state_dim) | global_cond (B, S×(feat+state)) | `DiffusionRgbEncoder`，L493 |
| 时间步嵌入 | 标量 t | 128 维向量 | `DiffusionSinusoidalPosEmb`，L615 |
| 噪声预测 U-Net | 带噪动作 (B,horizon,A) + t + cond | 噪声 (B,horizon,A) | `DiffusionConditionalUnet1d`，L657 |
| 采样器 | 纯噪声 + 条件 | 干净动作序列 | `conditional_sample`，L253 |

## 3. 训练 vs 推理

- **训练**（`compute_loss`，L354）：取演示动作块 → 随机采样时间步 t ~ U[0,100) →
  按调度加噪 `x_t = ᾱ_t·x₀ + σ_t·ε` → U-Net 预测噪声 → **MSE(ε̂, ε)**。
  网络从不直接回归动作，学的是"在当前观测条件下，如何一步步把噪声去掉"。
- **推理**：从纯噪声出发，循环 `num_inference_steps` 次（默认=训练 100 步）去噪生成
  动作序列，取当前对齐的 n_action_steps 步执行，队列耗尽后重新生成。

| 算法 | 训练输出 | 损失 | 推理输出 | 推理成本 |
|---|---|---|---|---|
| BC | 动作 a_t | MSE(a, a*) | 单步动作 | 1 次前向 |
| ACT | 动作块 | L1 + 10·KLD | chunk（1 次前向） | 1 次前向 + 队列 |
| Diffusion | 噪声 ε | MSE(ε, ε*) | 动作序列 | **N 次前向（N=推理步数）** |

关键区别：ACT 的 CVAE 用 latent 建模多模态、推理一次出整块；Diffusion 把多模态交给
"多次随机采样 + 迭代去噪"，表达力更强但推理成本与步数成正比。

## 4. 核心超参与调参方向

| 参数（checkpoint 实测值） | 作用 | 调参方向 |
|---|---|---|
| `num_train_timesteps`（100） | 前向加噪的总档位数 | 一般不动；改它需要同步重训，档位越多单步去噪越简单、训练越易 |
| `num_inference_steps`（默认=训练步数） | 推理去噪循环次数 | **精度/速度主旋钮**：100→50 步约快一倍，动作略粗；DDIM 下可压到 10–20 步。机器人 10–50Hz 控制下这是首要优化项 |
| `noise_scheduler_type`（DDPM） | 采样器 | 换 DDIM 可在少步数下保持质量；DDPM 步数少时质量掉得快 |
| `beta_schedule`（squaredcos_cap_v2，即 cosine） | β_t 增长曲线 | cosine 在高噪声段衰减慢，适合低维动作；linear 适合图像类高维数据。动作分布异常时先试 cosine |
| `horizon`（16） | 预测序列总长（含观测历史对齐） | 对应 ACT chunk_size：太长→开环误差累积，太短→重规划频繁、抖动。须为 2^len(down_dims)=8 的倍数 |
| `n_action_steps`（8） | 每次实际执行步数 | 决定开环时长 = 8/10s=0.8s（pusht 10Hz）；调小更稳更慢 |
| `n_obs_steps`（2） | 观测历史帧数 | Diffusion 原生支持多帧（lerobot 的 ACT 砍掉了这个能力） |
| `prediction_type`（epsilon） | 预测噪声还是直接预测样本 | 论文经验 epsilon 更稳；改 sample 通常无收益 |
| `clip_sample`（true, range 1.0） | 每步去噪后裁剪 | 依赖动作归一化到 [-1,1]（MIN_MAX 归一化），否则裁剪会削坏动作 |
| `down_dims`（512/1024/2048） | U-Net 容量 | 动作维度低（2 维）时 251M 参数明显过剩，可砍到 (256,512,1024) 提速省显存 |
