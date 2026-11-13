# ACT 跨本体迁移到 ALOHA 双臂插入（快速训练跑通）

> 把 ACT 策略迁移到 `lerobot/aloha_sim_insertion_human`，跑通训练与闭环评估全流程，
> 验证跨环境、跨本体（单臂 2 维 → 双臂 14 维）的迁移能力。本实验是
> [ALOHA 双臂数据集探查](aloha_dataset_probe.md) 之后的落地动作。
> 单实验记录见 [experiments/act_aloha/](../experiments/act_aloha/RESULTS.md)。

**一句话**：2000 步快速训练跑通，loss 前 900 步 10.73 → 2.07（-80.7%）；
闭环仿真评估也已跑通（5 集、0 成功，未收敛）。原先阻塞评估的
mujoco/dm_control 版本冲突已定位并修复（降级 mujoco 到 3.8.1，见第 6 节）。
**本实验证明的是跨本体流程可用，不是策略效果**——2000 步远未收敛，成功率数据不构成能力评价。

**日期**：2026-11-13 ｜ **环境**：lerobot 0.7.1 / py312-robot / A6000

---

## 1. 任务与动机
延续 ALOHA 探查：当时结论是「迁移的真正难点不是改维度，而是 50 Hz 预算与到位后保持」。
本次用 ACT（而非此前对比的 Diffusion/Flow）做真实训练，目的是给出一份
**跨本体迁移可跑通**的实证：同一套 lerobot 训练入口，从 PushT（单臂 2 维）切换到
ALOHA（双臂 14 维）只需改配置，无需重写代码；并进一步用独立评估入口验证闭环链路。

## 2. 环境与 CLI 适配（任务书为旧版写法）
任务书给的命令在 lerobot 0.7.1 下不能直接跑，实测需适配：

| 任务书写法 | 0.7.1 实测 |
|---|---|
| `pip install "lerobot[aloha]"` | 环境已具备（`aloha` extra 即 `gym-aloha>=0.1.5,<0.2.0`） |
| `python lerobot/scripts/train.py ...` | `python -m lerobot.scripts.lerobot_train`（直接 `python <文件>` 会 ImportError） |
| `--training.steps=2000` | `--steps=2000` |
| `--wandb.enable=false` | `--trackio.enable=false` |
| `--policy.device=cuda` / `--env.type=aloha` | 保留 |

额外两个 0.7.1 必填项（否则 `validate()` 抛错）：
- `--policy.repo_id=local/act_aloha_demo`（即便不推 hub）或 `--policy.push_to_hub=false`
- 不要预先创建 `--output_dir`：0.7.1 要求 output_dir 不存在且 `resume=False`，否则 `FileExistsError`

## 3. 训练结果
- 配置：2000 步、batch 8、ACT 52M 参数、A6000 单卡
- 耗时：约 **137 s**（数据集已缓存）
- 峰值显存：约 **2.1 GB**（A6000 49 GB，余量充足）
- 逐 step loss（前 900 步，每 100 步一条）：**10.73 → 2.07（-80.7%）**
  - l1 loss 0.61 → 0.29；kld loss 1.01 → 0.18
- checkpoint：001000 / 002000 均保存成功

![ACT 在 ALOHA 双臂插入任务的训练损失](../assets/act_aloha_loss.png)

| 指标 | step 100 | step 900 | 变化 |
|---|---|---|---|
| total loss | 10.728 | 2.073 | -80.7% |
| l1 loss | 0.610 | 0.292 | -52.1% |
| kld loss | 1.012 | 0.178 | -82.4% |

## 4. 跨本体迁移要点（呼应 ALOHA 探查）
- **维度**：PushT 动作 2 维 → ALOHA 14 维关节角（左右臂各 6 + 夹爪），策略输出头与归一化自动适配。
- **图像**：单 top 相机 480×640，与 PushT 的 96×96 不同，resize 由 lerobot 数据管线处理。
- **控制频率**：50 Hz（PushT 约 10 Hz）。本次用 `lerobot-eval` 做离线闭环 rollout，
  **不是 50 Hz 实时推理**；实时预算仍是后续重点（此前结论：只有 Flow 1–3 步能在 20 ms 内）。
- **双臂协同**：14 维两臂共享时间轴，无需拆策略，验证了 lerobot 数据管线对双臂格式的承载。

## 5. 闭环评估结果
用 2000 步 checkpoint 在 `AlohaInsertion-v0` 跑 5 集独立评估（`lerobot-eval`，
batch 1、seed 1000、单集最多 400 步、`MUJOCO_GL=egl`）。
数值由 [parse_eval.py](../experiments/act_aloha/scripts/parse_eval.py) 从 `eval_info.json` 自动生成，
完整摘要见 [eval_summary.md](../experiments/act_aloha/results/eval_summary.md)。

| 指标 | 值 |
|---|---|
| 集数 / 成功集数 | 5 / 0 |
| 成功率 | 0.0%（Wilson 95% CI **0.0%–43.4%**） |
| 平均累计奖励 | 11.4 |
| 平均峰值奖励 | 0.4 |
| 逐集累计奖励 | `[57.0, 0.0, 0.0, 0.0, 0.0]` |
| 逐集峰值奖励 | `[2.0, 0.0, 0.0, 0.0, 0.0]` |
| 评估耗时 | 82.3 s（16.5 s/集） |

代表性 rollout 视频：[ep0（唯一有非零奖励）](../assets/act_aloha_eval_ep0.mp4)、
[ep1（全零对照）](../assets/act_aloha_eval_ep1.mp4)。

**怎么读这个结果**：闭环链路（策略加载 → 双臂仿真 → 400 步 rollout → 指标/视频）完整跑通，
这是本次要验证的目标。5 集 0 成功是 2000 步训练远未收敛的预期表现，
**不能据此说 ACT 在 ALOHA 上无效**；反过来 Wilson CI 上限 43.4% 也说明 5 集样本太小，
同样不能声称"已经学会"。要评价策略能力需完整训练（通常数万步）后再评估。

## 6. 依赖冲突的定位与修复
首次评估时环境初始化报错：
`AttributeError: 'MjModel' object has no attribute 'flex_bandwidth'`。
调用栈落在 `dm_control/mujoco/index.py` 的 `struct_indexer` → `getattr(struct, field_name)`：
dm_control 的绑定表 `mjbindings/sizes.py` 硬编码了 `flex_bandwidth` 字段，而当前
`MjModel` 实例没有它。这不是策略或数据问题，是物理引擎与绑定的版本错配。

把三条版本约束摆在一起，答案是唯一的：

| 包 | 约束 | 来源 |
|---|---|---|
| dm_control 1.0.41 | `mujoco>=3.8.1` | 包元数据 `Requires-Dist` |
| gym-aloha 0.1.5 | `mujoco>=3.0.0,<3.9.0` | 包元数据 `Requires-Dist` |
| metaworld 3.0.0 | `mujoco>=3.0.0` | 包元数据 `Requires-Dist` |
| gym-pusht 0.1.8 | 不依赖 mujoco | 包元数据 |

原环境装的是 `mujoco 3.17.0`，实测 `hasattr(MjModel(), "flex_bandwidth")` 为 `False`
（新版已移除该字段），虽满足 dm_control 的 `>=3.8.1`，却违反 gym-aloha 的 `<3.9.0`，
且字段对不上。**两个约束的交集只有 `mujoco 3.8.1`**，修复只需一个命令：

```bash
pip install "mujoco==3.8.1"
```

修复后实测：`flex_bandwidth=True`；`gym_aloha/AlohaInsertion-v0` 创建与 `reset` 成功；
`gym_pusht` / `metaworld` / `lerobot` 回归导入全部正常（前两者不受降级影响）。
版本矩阵自动记录在 [eval_metrics.json](../experiments/act_aloha/results/eval_metrics.json) 的
`env_versions` 字段。若需回滚：`pip install "mujoco==3.17.0"`（会重现该报错）。

**关于「能不能直接上 mujoco 4」**：不行。PyPI 上 gym-aloha 最高版本就是 0.1.5，
它把 `mujoco<3.9` 写死；`dm_control 1.0.51` 是为 MuJoCo 4.0 升级的版本，
与 gym-aloha 0.1.5 的依赖闭包直接冲突。除非换用支持 mujoco 4 的新版 ALOHA 环境包，
否则 3.8.1 是当前版本组合下的唯一解。

> 更正说明：本报告前一版曾建议「mujoco 3.7.1 + dm_control 1.0.52」，实测后确认该建议错误：
> 3.7.1 低于 dm_control 1.0.41 的 `>=3.8.1` 下限，且 dm_control 并不存在 1.0.52 版本
> （最新为面向 MuJoCo 4 的 1.0.51）。此处以实跑验证的版本为准。

评估命令见 [run_eval.sh](../experiments/act_aloha/scripts/run_eval.sh)。

## 7. 复现
依赖版本矩阵见第 6 节；数据集已缓存到工作区 `datasets/aloha_sim_insertion_human`（约 91 MB）。

```bash
WZH_ROOT=<workspace-root> bash experiments/act_aloha/scripts/run_train.sh
WZH_ROOT=<workspace-root> python experiments/act_aloha/scripts/parse_train_log.py
# 闭环评估（需第 6 节的兼容版本组合）：
WZH_ROOT=<workspace-root> bash experiments/act_aloha/scripts/run_eval.sh
WZH_ROOT=<workspace-root> python experiments/act_aloha/scripts/parse_eval.py
```

## 8. 产物
```
docs/act_aloha_transfer.md                 本文
experiments/act_aloha/RESULTS.md           单实验记录
experiments/act_aloha/scripts/             run_train.sh / run_eval.sh / parse_train_log.py / parse_eval.py
experiments/act_aloha/results/             train_metrics.json / train_log_excerpt.txt
                                          eval_metrics.json / eval_summary.md
assets/act_aloha_loss.png                 训练损失曲线
assets/act_aloha_eval_ep0.mp4             代表性 rollout（唯一非零奖励）
assets/act_aloha_eval_ep1.mp4             全零对照 rollout
```
训练权重、原始训练日志与原始 `eval_info.json` 不入库（含工作区路径），
存于 `<workspace-root>/outputs/` 下对应实验目录。

## 9. 局限
1. 损失曲线覆盖前 45% 训练（lerobot 0.7.1 仅打印前 ~1000 步逐 step 日志）；完整 2000 步已完成。
2. 闭环评估已跑通，但仅 5 集、0 成功且策略未收敛；Wilson CI 很宽，
   该数据只用于证明链路可用，**不作为策略能力结论**。
3. 评估是 `lerobot-eval` 的离线 rollout，不是 50 Hz 实时控制；实时推理预算需另测。
4. 2000 步训练未做超参调优；要得到有意义的成功率，需完整训练后再评估。
