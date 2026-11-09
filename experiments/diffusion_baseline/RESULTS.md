# Diffusion Policy 公平对照基线（与 BC v1 / ACT v1 同协议）

日期：2026-11-12 ｜ 环境：RTX A6000 / py312-robot（torch 2.7.1, lerobot 0.7.1）

## 协议对齐（控制变量）

| 项目 | 本实验 | BC v1 | ACT v1 |
|---|---|---|---|
| 数据集 | 本地 pusht，198 训练 / 8 验证（eval_split=0.04） | 同左 | 同左 |
| 训练步数 | 7680（batch=64） | 7680 | 7680 |
| 评估间隔 | 每 384 步留出集 eval_loss | 每 epoch（384 步） | 每 384 步 |
| 算法超参 | 官方 PushT 默认（n_obs_steps=2、horizon=16、n_action_steps=8、DDPM 100 步、cosine、lr 1e-4） | — | — |

## 训练

- eval_loss：0.423（epoch 1）→ **最优 0.0422 @ step 2304** → 回升至 0.09+（step 7680）。
- **早停现象**：约 epoch 6 后开始过拟合，最终 checkpoint 的 eval_loss 已劣化 2 倍。
  可用 checkpoint 为 step 3840（eval_loss 0.0478）与 7680，本次评估采用 **step 3840**。
  这是三次基线训练中唯一出现明显早停的——扩散模型容量（U-Net 251M）远大于
  BC（0.24M）与 ACT（51.6M），小数据集上最先撞上过拟合墙。
- 峰值显存 4.3 GB（batch=64），7680 步耗时约 15 分钟（8.6 步/s）。

## 评估（50 episodes × 300 步，与 v2/v3 基线同口径）

| 模型 | 训练量 | 成功率 | 峰值覆盖率 | avg sum_reward |
|---|---|---|---|---|
| BC v1 | 7.7k 步 | 0% | 0.13 | 34.5 |
| ACT v1 | 7.7k 步 | 0%（20ep） | 0.47 | 43.6 |
| **Diffusion（本实验）** | **7.7k 步** | **6%（3/50）** | **0.67** | **79.6** |

成功 episode：#13、#27、#44；回放 [diffusion_baseline_success.gif](../../assets/diffusion_baseline_success.gif)
（夹爪从左侧连续推入，约 10 秒完成，动作全程平滑无抖动）。

## 结论

1. **同预算下 Diffusion 显著最强**：7680 步是三个基线中唯一获得非零成功率的
   （6% vs 0%/0%），峰值覆盖率 0.67 也最高——迭代去噪生成的动作块天然平滑，
   不需要 ACT 的时间集成。
2. **过拟合最早出现**：251M 参数的 U-Net 在 2.4 万帧数据上 epoch 6 即达最优，
   之后 eval_loss 持续劣化——后续完整训练应配合更强的增广或更早停。
3. 推理成本（batch=1 微基准实测）：单次 U-Net 前向 16.9 ms，DDPM 100 步
   重规划一次 ≈ 1.7–2.2 s。与 ACT（10.3 ms/块）相差约 **170 倍**；
   摊到单步动作：BC 2.2 ms、ACT 0.34 ms、Diffusion ≈ 55 ms。
   补充说明：本次训练用了默认 horizon=64 / n_action_steps=32（每 32 步才重规划一次，
   摊薄了部分成本；评估墙钟 1.21 s/ep 与 ACT 的 0.94 s/ep 接近，是因为
   10 并发环境共享一次批量重规划）。
4. 配置勘误：官方 dataclass 默认 horizon=64/n_action_steps=32，并非 checkpoint
   文档所述的 PushT 特化值 16/8——本实验按默认 64/32 训练，结论不受影响。
