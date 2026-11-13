"""训练稳定性量化对比。

两个基线都在每 384 步（每 epoch）记录一次留出集损失，采样频率完全一致：
  Flow      : experiments/flow_baseline/train_log.txt      -> "epoch N: train_loss X | val_loss Y"
  Diffusion : experiments/diffusion_baseline/train_log.txt -> "step N: eval_loss=Y"

指标：相邻验证点绝对波动、滑窗标准差、变异系数、最优点、最终点相对最优的劣化倍数。
"""

import json
import re
import statistics
from pathlib import Path

# --- 可移植路径：仓库内相对定位；外部产物可用 WZH_ROOT 覆盖 ---
import os
HERE = Path(__file__).resolve().parent            # experiments/flow_vs_diffusion/scripts
BASE = HERE.parent                                # experiments/flow_vs_diffusion
REPO = BASE.parent.parent                         # pusht-policy-bench 根目录
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))  # 默认：仓库上一级目录
OUT = BASE / "results"
ASSETS = REPO / "assets"

ROOT = REPO / "experiments"


def metrics(vals, label):
    diffs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    win = 4
    rolling = [statistics.pstdev(vals[i:i + win]) for i in range(len(vals) - win + 1)]
    best_i = min(range(len(vals)), key=lambda i: vals[i])
    downs = sum(1 for d in (vals[i] - vals[i - 1] for i in range(1, len(vals))) if d < 0)
    return {
        "label": label,
        "n_points": len(vals),
        "first": vals[0],
        "best": vals[best_i],
        "best_index": best_i,
        "final": vals[-1],
        "final_over_best": vals[-1] / vals[best_i],
        "mean_abs_step_change": sum(diffs) / len(diffs),
        "max_abs_step_change": max(diffs),
        "std": statistics.pstdev(vals),
        "cv": statistics.pstdev(vals) / statistics.mean(vals),
        "rolling4_std_mean": sum(rolling) / len(rolling),
        "rolling4_std_max": max(rolling),
        "down_step_fraction": downs / len(diffs),
        "series": vals,
    }


def main():
    flow_txt = (ROOT / "flow_baseline/train_log.txt").read_text(encoding="utf-8", errors="replace")
    diff_txt = (ROOT / "diffusion_baseline/train_log.txt").read_text(encoding="utf-8", errors="replace")

    flow_epochs = re.findall(r"epoch (\d+): train_loss ([\d.]+) \| val_loss ([\d.]+)", flow_txt)
    flow_val = [float(x[2]) for x in flow_epochs]
    flow_train = [float(x[1]) for x in flow_epochs]
    flow_step = [float(v) for _, v in re.findall(r"step (\d+) loss ([\d.]+)", flow_txt)]

    diff_eval = re.findall(r"step (\d+): eval_loss=([\d.]+)", diff_txt)
    diff_steps = [int(s) for s, _ in diff_eval]
    diff_val = [float(v) for _, v in diff_eval]

    data = {
        "protocol": "两者均为每 384 步（每 epoch）一次留出集验证，共 20 个点，采样频率一致",
        "flow": metrics(flow_val, "flow_matching val_loss"),
        "diffusion": metrics(diff_val, "diffusion_policy eval_loss"),
        "flow_train_loss_series": flow_train,
        "flow_batch_step_loss_series": flow_step,
        "diffusion_eval_steps": diff_steps,
        "notes": [
            "Flow 另有 batch 级 step loss 序列，可反映单 batch 梯度震荡；Diffusion 日志未逐 batch 打印 loss。",
            "val_loss 与 eval_loss 定义不同（MSE 速度场 vs 噪声预测 MSE），绝对值不可直接比，仅比较相对波动与早停位置。",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stability.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    show = {k: {kk: vv for kk, vv in v.items() if kk != "series"}
            for k, v in data.items() if k in ("flow", "diffusion")}
    print(json.dumps(show, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
