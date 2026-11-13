#!/usr/bin/env bash
# 用 2000 步 checkpoint 在 ALOHA 仿真环境做少量闭环评估（验证跨环境迁移）。
# 依赖版本矩阵（实测可跑）：mujoco 3.8.1 / dm-control 1.0.41 / gym-aloha 0.1.5
#   dm-control 1.0.41 要求 mujoco>=3.8.1，gym-aloha 0.1.5 要求 mujoco<3.9，
#   交集只有 3.8.1；装 3.17.0 会报 "MjModel object has no attribute 'flex_bandwidth'"。
#   定位与修复见 docs/act_aloha_transfer.md 第 6 节。
# 用法：WZH_ROOT=<workspace-root> bash scripts/run_eval.sh
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py312-robot

WZH_ROOT="${WZH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
CKPT="${CKPT:-${WZH_ROOT}/outputs/act_aloha_demo/checkpoints/002000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${WZH_ROOT}/outputs/act_aloha_demo_eval}"
N_EPISODES="${N_EPISODES:-5}"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
lerobot-eval \
  --policy.path="${CKPT}" \
  --env.type=aloha \
  --eval.n_episodes="${N_EPISODES}" \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --output_dir="${OUTPUT_DIR}" \
  --seed=1000
