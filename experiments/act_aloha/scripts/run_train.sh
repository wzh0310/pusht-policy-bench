#!/usr/bin/env bash
# ACT 迁移到 ALOHA 双臂插入任务：快速训练跑通全流程（不追求收敛）
# 环境：py312-robot / lerobot 0.7.1 / A6000
# 用法：WZH_ROOT=<workspace-root> bash scripts/run_train.sh
# 可选覆盖：STEPS / OUTPUT_DIR / SAVE_CHECKPOINT / SAVE_FREQ / LOG_FREQ / POLICY_REPO_ID
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py312-robot

# 禁用 tqdm 进度条，避免 \r 覆盖 step 日志导致只有前几步落盘
export TQDM_DISABLE=1

# 默认把工作区取为仓库的上级目录；也可用 WZH_ROOT 显式指定
WZH_ROOT="${WZH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATASET_ROOT="${WZH_ROOT}/datasets/aloha_sim_insertion_human"
OUTPUT_DIR="${OUTPUT_DIR:-${WZH_ROOT}/outputs/act_aloha_demo}"
STEPS="${STEPS:-2000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-100}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/act_aloha_demo}"

# 注意：不要预先创建 OUTPUT_DIR。lerobot 0.7.1 要求 output_dir 不存在
# 且 resume=False，否则 validate() 直接抛 FileExistsError。

# lerobot 是 editable 安装，-m 加载的正是工作区里 lerobot/scripts/lerobot_train.py；
# 直接 python <文件路径> 会因包内相对导入报 ImportError。
python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="${DATASET_ROOT}" \
  --env.type=aloha \
  --env_eval_freq=0 \
  --steps="${STEPS}" \
  --output_dir="${OUTPUT_DIR}" \
  --batch_size=8 \
  --num_workers=4 \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}" \
  --trackio.enable=false
