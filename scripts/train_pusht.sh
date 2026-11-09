#!/usr/bin/env bash
# ACT 训练：默认读取 configs/act_pusht.yaml
#
#   bash scripts/train_pusht.sh              # 按配置训练
#   bash scripts/train_pusht.sh 100 8        # 冒烟测试：100 步，batch 8
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-${REPO_ROOT}/configs/act_pusht.yaml}
STEPS=${1:-}
BATCH=${2:-}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# 必须以模块方式启动，脚本内部使用相对导入
CMD=("${PYTHON}" -m lerobot.scripts.lerobot_train --config_path="${CONFIG}")
[[ -n "${STEPS}" ]] && CMD+=(--steps="${STEPS}")
[[ -n "${BATCH}" ]] && CMD+=(--batch_size="${BATCH}")

echo "运行: ${CMD[*]}"
exec "${CMD[@]}"

# 产物：<output_dir>/checkpoints/<step>/pretrained_model/
#   config.json · model.safetensors · policy_preprocessor.json · policy_postprocessor.json
# 续训：
#   --resume=true --config_path=<output_dir>/checkpoints/<step>/pretrained_model/train_config.json
