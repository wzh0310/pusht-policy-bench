#!/usr/bin/env bash
# 评测 PushT 策略并导出逐 episode 视频
#
#   bash scripts/eval_pusht.sh
#   N_EPISODES=50 bash scripts/eval_pusht.sh
#   POLICY_PATH=./my_checkpoint bash scripts/eval_pusht.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON=${PYTHON:-python}
POLICY_PATH=${POLICY_PATH:-lerobot/diffusion_pusht}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/pusht}
N_EPISODES=${N_EPISODES:-10}

# 无显示器的服务器必须离屏渲染，否则 MuJoCo 报 GL 错误
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# 预检：CUDA 不可用时尽早失败，避免中途跑到一半才发现
"${PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 不可用：检查驱动 / torch 版本 / 设备节点"
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}")
PY

exec "${PYTHON}" -m lerobot.scripts.lerobot_eval \
  --policy.path="${POLICY_PATH}" \
  --env.type=pusht \
  --eval.n_episodes="${N_EPISODES}" \
  --output_dir="${OUTPUT_DIR}"

# 指标：${OUTPUT_DIR}/eval_info.json
# 视频：${OUTPUT_DIR}/videos/<task>_<id>/eval_episode_*.mp4
