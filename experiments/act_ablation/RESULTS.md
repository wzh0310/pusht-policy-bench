# ACT 消融实验：chunk_size 与 CVAE

日期：2026-11-11 ｜ 环境：RTX A6000 / py312-robot（torch 2.7.1, lerobot 0.7.1）

**统一协议**：50000 步、batch 64、lr 2e-5（backbone 2e-6）、本地 pusht、
每 3200 步 eval_loss（eval_split=0.04）、评估 50 episodes × 300 步、
`n_action_steps = chunk_size`（开环执行整块，不开时间集成）。
每组只改一个变量，其余与 [act_v3](../act_v3/) 基线完全一致。

## 消融一：chunk_size（1 / 10 / 30）

| 组 | chunk_size | 成功率（50 ep） | 峰值覆盖率 | 最优 eval_loss |
|---|---|---|---|---|
| 1 | **1**（单步 ≈ BC） | **0/50（0%）** | 0.083 | 0.132* |
| 2 | **10** | **9/50（18%）** | 0.631 | 0.240* |
| 3 | **30**（= act_v3） | 7/50（14%） | 0.613 | 0.362 |

\* chunk=1/10 的 eval_loss 是对 1/10 步动作的回归误差，与 chunk=30 不同量纲，横向不可比。

**结论**：

1. **chunk=1 灾难性失败（0%）**：同样的网络、数据、步数，仅取消动作分块，
   成功率从 18% 归零——这是"动作分块抑制误差累积"最直接的因果证据。
2. **甜点在 10 附近**：chunk 10（18%）> chunk 30（14%）。块太长会跨越状态显著
   变化的开环区间，误差重新累积；块太短则失去平滑收益。
3. chunk=1 并非完全学不会——最优单条 episode 覆盖率摸到 **0.932**（阈值 0.95），
   但多数 episode 覆盖率仅 0.08 量级：**单步决策的方差极大**，偶尔推得很好，
   多数时候夹爪在块边犹豫抖动（见失败案例分析）。

## 消融二：CVAE（`use_vae=false`）

> 原计划做 `n_obs_steps` 消融，但 lerobot 的 ACT 在 `configuration_act.py:149`
> 显式拒绝多帧观测（"Multiple observation steps not handled yet"），
> 原生不支持、无法命令行传参，故改为 CVAE 消融（同为 ACT 核心设计、原生支持）。

| 组 | use_vae | 成功率（50 ep） | 峰值覆盖率 | 最优 eval_loss |
|---|---|---|---|---|
| 基线（= act_v3） | true | **7/50（14%）** | 0.613 | 0.362 |
| 无 VAE | false | **3/50（6%）** | 0.680 | 0.371 |

**结论**：关闭 CVAE 后成功率腰斩（14%→6%），尽管峰值覆盖率反而更高（0.68）——
没有 latent 多模态建模，策略在局部推得同样准，但在"从哪一侧绕、往哪推"的
分叉决策上更容易选到演示分布外的分支。CVAE 的价值不在平均精度，在**减少
灾难性分叉**。

## 失败案例对比分析

| 模型 | 典型失效模式 | 回放 |
|---|---|---|
| chunk=1 | 夹爪在块边缘小幅抖动（单步决策噪声直接执行），偶尔推得很好但方差极大；最好一条停在 0.932 未过线 | [ablation_chunk1_best_effort.gif](../../assets/ablation_chunk1_best_effort.gif) |
| 无 VAE | 推得准但分叉选错：块被推到目标边缘后反复进出，缺最后一段精确导入 | [ablation_novae_best_effort.gif](../../assets/ablation_novae_best_effort.gif) |
| ACT 完整版（v3） | 少数 episode 最后 5% 覆盖率反复修正超时 | [act_v3_success.gif](../../assets/act_v3_success.gif)（成功例） |
| BC v2 | 能推进目标附近，单步误差累积后"绕着块转" | [bc_v2_best.gif](../../assets/bc_v2_best.gif)（成功例） |

一句话总结四种失效模式：**单步决策死于抖动，无 latent 死于分叉，完整 ACT 死于
最后 5%，BC 死于累积**——动作分块、CVAE、时间集成各自修补其中一段。
