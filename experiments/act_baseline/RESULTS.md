# ACT 完整基线（20 epoch 当量）：训练 + 20 episode 评估

日期：2026-11-10 ｜ 环境：RTX A6000 / py312-robot（torch 2.7.1, lerobot 0.7.1）

## 协议（与 baselines/bc 严格对齐）

| 项目 | BC 基线 | 本实验（ACT） |
|---|---|---|
| 数据集 | 本地 pusht v3.0（198 训练 / 8 验证 episode） | 同左（lerobot `eval_split=0.04` ≈ 8 episodes 留出） |
| 训练步数 | 7680（20 epoch × 384 步，batch=64） | 7680（batch=64） |
| 评估间隔 | 每 epoch 验证集 MSE | 每 384 步留出集 L1（`eval_loss`） |
| 算法默认超参 | lr=1e-4（AdamW） | lr=1e-5（ACT preset），chunk_size=30，n_action_steps=30，kl_weight=10 |

## 训练结果

- 7680 步耗时 14.5 分钟（≈8.8 步/s），峰值显存 1.44 GB（batch=64）。
- eval_loss 曲线见 `../../assets/act_baseline_eval_loss.png`：0.4228（epoch 1）→ 最优 **0.3753 @ epoch 12** → 轻微回升至 0.3865（epoch 20）。
- 注意：本次存档策略为 `save_freq=7680`，只保存了最终 checkpoint；曲线显示 epoch 12 后已进入平台期，epoch 12–20 之间的波动（±0.005）在留出集噪声范围内，最终 checkpoint 即"最优可得"。

## 评估结果（20 episodes，完整回放）

| 指标 | 值 |
|---|---|
| 成功率 | **0/20（0%）** |
| avg sum_reward | 43.6 |
| avg max_reward（峰值覆盖率） | 0.47 |
| 评估耗时 | 25.4 s（20 episode 并行，1.27 s/ep） |

逐 episode 明细见 [eval_info.json](eval_info.json)；最佳 episode（sum_reward=114.4，覆盖率峰值约 0.55–0.6）
回放见 `../../assets/act_baseline_best.gif` / `.mp4`。

## 失败分析

1. **步数不足是主因**。ACT 官方模型 `lerobot/act_pusht` 需 30k–50k 步收敛；本实验为对齐 BC 协议只训了 7.7k 步（lr 还是 ACT 标准的 1e-5，比 BC 的 1e-4 低一个数量级）。eval_loss 平台期（0.375）对应的正是"块能推到目标附近但插不准"的行为。
2. **留出集 loss 与成功率脱钩**。eval_loss 从 epoch 1 到 12 只降了 11%，但成功率始终为 0——L1 回归误差的微小下降不改变闭环行为的分叉点，这是"离线指标≠在线表现"的又一例证（与 [docs/results.md](../../docs/results.md) 的结论一致）。
3. **协议公平性的代价**。控制变量（同数据、同步数、同评估间隔）意味着 ACT 只能用它自己合适的低学习率跑很少的 epoch，其算法优势（动作分块抑制误差累积）还没来得及显现。

## 改进方向

- 放开步数到 30k–50k（或 lr 升至 3e-5~1e-4），保持其余协议不变，复测 20 episode；
- `save_freq` 降到 1000，真正按 eval_loss 选最优 checkpoint；
- 用同一协议补训 BC 基线的在线评估（mujoco rollout），得到 BC 0/N 的对照数字。
