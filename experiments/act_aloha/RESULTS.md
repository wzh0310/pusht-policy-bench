# ACT 迁移到 ALOHA 双臂插入任务（快速训练 + 闭环评估）

## 目标
将 ACT 策略迁移到 `lerobot/aloha_sim_insertion_human`（双臂插销插入），
跑通训练与评估全流程，验证算法**跨环境、跨本体**（单臂 2 维 → 双臂 14 维）的迁移能力。
按任务书要求：不追求收敛，跑通即可，作为面试作品集的补充素材。

## 环境与入口
- lerobot 0.7.1（editable 安装，即工作区内的 `lerobot/scripts/lerobot_train.py`）
- conda `py312-robot` / torch 2.7.1+cu126 / A6000（49 GB）
- 启动方式：**必须**用 `python -m lerobot.scripts.lerobot_train`。
  直接 `python <文件路径>.py` 会因包内相对导入 `from .lerobot_eval import ...` 报 `ImportError`；
  `-m` 加载的正是工作区里那份脚本，与任务书意图一致。

## 相对任务书截图的关键适配（任务书为旧版 CLI）
| 任务书写法 | 0.7.1 实测 |
|---|---|
| `python lerobot/scripts/train.py` | `python -m lerobot.scripts.lerobot_train` |
| `--training.steps=2000` | `--steps=2000` |
| `--wandb.enable=false` | `--trackio.enable=false` |
| `--policy.device=cuda` | 保留（0.7.1 支持） |
| `--env.type=aloha` | 保留（0.7.1 支持） |

额外必须的修正（否则 `validate()` 直接抛错）：
- 必须给 `--policy.repo_id`（即便不推 hub）或显式 `--policy.push_to_hub=false`；
- **不要预先创建 `output_dir`**：0.7.1 要求 output_dir 不存在且 `resume=False`，否则 `FileExistsError`。

## 闭环评估的依赖版本矩阵（实测）
| 包 | 版本 | 对 mujoco 的约束 |
|---|---|---|
| dm-control | 1.0.41 | `mujoco>=3.8.1` |
| gym-aloha | 0.1.5 | `mujoco>=3.0.0,<3.9.0` |
| metaworld | 3.0.0 | `mujoco>=3.0.0` |
| gym-pusht | 0.1.8 | 不依赖 mujoco |

约束交集只有 `mujoco 3.8.1`。原环境 3.17.0 已移除 `flex_bandwidth` 字段，
会报 `'MjModel' object has no attribute 'flex_bandwidth'`。修复：
```bash
pip install "mujoco==3.8.1"
```
降级后 `gym_pusht` / `metaworld` / `lerobot` 回归导入全部正常。
完整推理、mujoco 4 为何不可行、以及前一版错误建议的更正见 `docs/act_aloha_transfer.md` 第 6 节。

## 复现
```bash
# 数据集已缓存到工作区 datasets/aloha_sim_insertion_human（约 91 MB）
WZH_ROOT=<workspace-root> bash scripts/run_train.sh
# 训练产物：<workspace-root>/outputs/act_aloha_demo/（checkpoints 001000 / 002000 + 训练日志）
WZH_ROOT=<workspace-root> python scripts/parse_train_log.py
#   → results/train_metrics.json、results/train_log_excerpt.txt、仓库 assets/act_aloha_loss.png
# 闭环评估（需上面的兼容版本组合）：
WZH_ROOT=<workspace-root> bash scripts/run_eval.sh
WZH_ROOT=<workspace-root> python scripts/parse_eval.py
#   → results/eval_metrics.json、results/eval_summary.md
```
可覆盖环境变量：`STEPS` `OUTPUT_DIR` `SAVE_CHECKPOINT` `SAVE_FREQ` `LOG_FREQ` `POLICY_REPO_ID` `CKPT` `N_EPISODES`。

## 结果
- 训练步数：**2000**（配置），实际耗时约 **137 s**；ACT 参数 **52 M**
- 逐 step loss（前 900 步，每 100 步一条）：**10.73 → 2.07（-80.7%）**
  - l1 loss：0.61 → 0.29；kld loss：1.01 → 0.18
- 峰值显存约 **2.1 GB**（batch 8，A6000 49 GB，余量充足）
- checkpoint：001000 / 002000 均保存成功
- **闭环评估（`AlohaInsertion-v0`，5 集，seed 1000，单集最多 400 步）**：
  - 成功 **0/5**，成功率 0.0%，Wilson 95% CI **0.0%–43.4%**
  - 平均累计奖励 **11.4**，平均峰值奖励 **0.4**
  - 逐集累计奖励 `[57.0, 0.0, 0.0, 0.0, 0.0]`；耗时 82.3 s（16.5 s/集）
  - 数值由 `scripts/parse_eval.py` 从 `eval_info.json` 自动生成
  - 代表性视频：`assets/act_aloha_eval_ep0.mp4`（唯一非零奖励）、`ep1`（全零对照）

## 说明与局限
1. **损失曲线覆盖前 45% 训练**：lerobot 0.7.1 仅对训练前 ~1000 步打印逐 step 损失日志
   （本实验捕获 step 100–900，共 9 点）。完整 2000 步训练已执行完毕，最终权重存于
   `checkpoints/002000`。若要完整曲线，可改为开启 trackio 记录或微调框架日志上限。
2. **闭环评估已跑通，但不能当能力评价**：2000 步远未收敛，5 集 0 成功是预期表现；
   Wilson CI 上限 43.4% 说明样本太小。该数据只证明「策略加载 → 双臂仿真 → rollout → 指标/视频」
   链路可用。要评价策略能力需完整训练（通常数万步）后再评估。
3. 评估是 `lerobot-eval` 的**离线** rollout，不是 50 Hz 实时控制；实时推理预算需另测。
4. 训练日志与原始 `eval_info.json` 不入库（含工作区绝对路径）；
   `results/train_log_excerpt.txt`、`results/eval_summary.md` 提供脱敏摘要。
