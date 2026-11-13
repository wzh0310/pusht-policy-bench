#!/usr/bin/env bash
# Flow Matching 推理步数消融：1 / 3 / 5 / 10 步 Euler 积分，同种子同协议
set -euo pipefail
# 环境：py312-robot（Python 3.12 / PyTorch 2.7），使用通用 conda 初始化
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py312-robot
cd "$(dirname "$0")/.."

for s in 1 3 5 10; do
    echo "===== Flow infer_steps=${s} ====="
    python scripts/eval_flow_steps.py \
        --infer-steps "${s}" \
        --episodes 50 \
        --max-steps 300 \
        --seed 1000 \
        --out "results/flow_steps${s}.json" \
        2>&1 | tee "results/flow_steps${s}_eval.txt"
done

echo "===== 全部步数消融完成 ====="
