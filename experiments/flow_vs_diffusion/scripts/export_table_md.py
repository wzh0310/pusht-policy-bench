#!/usr/bin/env python3
"""把 results/ablation_table.csv 导出为可直接贴作品集的 Markdown 对比表。"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "ablation_table.csv"
DST = ROOT / "results" / "ablation_table.md"

COLS = [
    ("algo", "算法", "left"),
    ("infer_steps", "推理步数", "right"),
    ("success_pct", "成功率 %", "right"),
    ("success_ci95", "Wilson 95% CI", "center"),
    ("peak_coverage", "峰值覆盖率", "right"),
    ("avg_sum_reward", "avg sum_reward", "right"),
    ("replan_ms", "重规划 ms", "right"),
    ("per_step_ms", "单步 ms", "right"),
]

with SRC.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

out = [
    "# 推理步数消融量化对比表",
    "",
    "PushT-v0 ｜ 50 episodes × 300 步 ｜ 同种子 1000+ep ｜ RTX A6000 / torch 2.7.1+cu126",
    "",
    "| " + " | ".join(h for _, h, _ in COLS) + " |",
    "|" + "|".join({"left": ":---", "center": ":---:", "right": "---:"}[a] for _, _, a in COLS) + "|",
]
for r in rows:
    cells = []
    for key, _, _ in COLS:
        v = r[key]
        if key == "replan_ms":
            v = f"{float(v):.2f}"
        elif key == "per_step_ms":
            v = f"{float(v):.3f}"
        elif key == "success_pct":
            v = f"{float(v):.1f}"
        if r["algo"] == "Flow Matching" and r["infer_steps"] == "1":
            v = f"**{v}**"
        cells.append(v)
    out.append("| " + " | ".join(cells) + " |")

out += [
    "",
    "## 口径说明",
    "",
    "- **成功率**：50 episodes 的二值成功比例，括号内为 Wilson 95% 置信区间；各组 CI 互相重叠，不能用于排序。",
    "- **峰值覆盖率 / sum_reward**：连续量，方差更小，是本次跨算法比较的主要依据。",
    "- **重规划延迟**：Flow 为闭环评估内 `model.sample()` GPU 同步实测；Diffusion 为同机 batch=1 微基准 20 次重复均值；均为单次重规划口径，不含环境步进与渲染。",
    "- **来源**：Flow 四档为本次实跑；Diffusion 质量指标复用既有同协议步数消融结果（step-3840 早停 checkpoint）。",
    "",
    "详细分析见同目录 `REPORT.md`。",
]
DST.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote {DST}  ({len(rows)} rows)")
