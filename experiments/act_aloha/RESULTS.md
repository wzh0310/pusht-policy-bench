# ACT 迁移到 ALOHA 双臂插入任务（快速训练跑通）

## 目标
将 ACT 策略迁移到 `lerobot/aloha_sim_insertion_human`（双臂插销插入），
跑通训练全流程，验证算法**跨环境、跨本体**（单臂 2 维 → 双臂 14 维）的迁移能力。
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

## 复现
```bash
# 数据集已缓存到工作区 datasets/aloha_sim_insertion_human（约 91 MB）
WZH_ROOT=<workspace-root> bash scripts/run_train.sh
# 训练产物：<workspace-root>/outputs/act_aloha_demo/（checkpoints 001000 / 002000 + 训练日志）
WZH_ROOT=<workspace-root> python scripts/parse_train_log.py
#   → results/train_metrics.json、results/train_log_excerpt.txt、仓库 assets/act_aloha_loss.png
# 闭环评估（需先解决环境依赖，见 docs/act_aloha_transfer.md）：
WZH_ROOT=<workspace-root> bash scripts/run_eval.sh
```
可覆盖环境变量：`STEPS` `OUTPUT_DIR` `SAVE_CHECKPOINT` `SAVE_FREQ` `LOG_FREQ` `POLICY_REPO_ID` `CKPT` `N_EPISODES`。

## 结果
- 训练步数：**2000**（配置），实际耗时约 **137 s**；ACT 参数 **52 M**
- 逐 step loss（前 900 步，每 100 步一条）：**10.73 → 2.07（-80.7%）**
  - l1 loss：0.61 → 0.29；kld loss：1.01 → 0.18
- 峰值显存约 **2.1 GB**（batch 8，A6000 49 GB，余量充足）
- checkpoint：001000 / 002000 均保存成功

## 说明与局限
1. **损失曲线覆盖前 45% 训练**：lerobot 0.7.1 仅对训练前 ~1000 步打印逐 step 损失日志
   （本实验捕获 step 100–900，共 9 点）。完整 2000 步训练已执行完毕，最终权重存于
   `checkpoints/002000`。若要完整曲线，可改为开启 trackio 记录或微调框架日志上限。
2. **闭环环境评估未在本机共享环境自动跑通**：lerobot-eval 触发
   `MjModel object has no attribute 'flex_bandwidth'`，根因是服务器上
   `dm_control` 与 `mujoco` 版本不兼容（ALOHA 仿真依赖 gym-aloha → mujoco）。
   这是底层依赖冲突，不是策略或数据问题。修复方向与命令见 `docs/act_aloha_transfer.md`。
3. 训练日志本身不入库（含工作区绝对路径）；`results/train_log_excerpt.txt` 提供脱敏的逐 step 摘要。
