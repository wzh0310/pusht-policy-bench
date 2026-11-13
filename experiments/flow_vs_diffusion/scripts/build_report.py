"""汇总所有实验产物，生成消融主表 CSV 与结构化 summary.json。"""

import csv
import json
import math
from pathlib import Path

# --- 可移植路径：仓库内相对定位；外部产物可用 WZH_ROOT 覆盖 ---
import os
HERE = Path(__file__).resolve().parent            # experiments/flow_vs_diffusion/scripts
BASE = HERE.parent                                # experiments/flow_vs_diffusion
REPO = BASE.parent.parent                         # pusht-policy-bench 根目录
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))  # 默认：仓库上一级目录
OUT = BASE / "results"
ASSETS = REPO / "assets"

R = BASE / "results"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    diff_q = {r["infer_steps"]: r for r in
              json.loads((R / "diffusion_steps.json").read_text(encoding="utf-8"))}
    lat_rows = {r["infer_steps"]: r for r in
                json.loads((R / "diffusion_latency.json").read_text(encoding="utf-8"))["rows"]}

    rows = []
    for s in (1, 3, 5, 10):
        d = json.loads((R / f"flow_steps{s}.json").read_text(encoding="utf-8"))["summary"]
        lo, hi = wilson(d["n_success"], d["n_episodes"])
        rows.append({
            "algo": "Flow Matching", "infer_steps": s,
            "n_episodes": d["n_episodes"], "n_success": d["n_success"],
            "success_pct": round(d["pc_success"] * 100, 1),
            "success_ci95": f"{lo*100:.1f}-{hi*100:.1f}",
            "peak_coverage": round(d["avg_max_reward"], 3),
            "avg_sum_reward": round(d["avg_sum_reward"], 1),
            "replan_ms": round(d["latency_ms_mean"], 2),
            "per_step_ms": round(d["latency_ms_per_euler_step"], 3),
            "latency_source": "闭环评估内 model.sample() GPU 同步实测",
            "quality_source": "本次消融 50ep x 300 步",
        })
    for s in (10, 20, 50):
        d = diff_q[s]
        l = lat_rows[s]
        lo, hi = wilson(d["n_success"], d["n_episodes"])
        rows.append({
            "algo": "Diffusion Policy", "infer_steps": s,
            "n_episodes": d["n_episodes"], "n_success": d["n_success"],
            "success_pct": round(d["pc_success"] * 100, 1),
            "success_ci95": f"{lo*100:.1f}-{hi*100:.1f}",
            "peak_coverage": round(d["avg_max_reward"], 3),
            "avg_sum_reward": round(d["avg_sum_reward"], 1),
            "replan_ms": round(l["latency_ms_mean"], 2),
            "per_step_ms": round(l["latency_ms_per_denoise_step"], 3),
            "latency_source": "同机 batch=1 微基准（20 次重复）",
            "quality_source": "既有步数消融 50ep x 300 步",
        })

    fields = list(rows[0].keys())
    with (R / "ablation_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    flow_lat = [(r["infer_steps"], r["replan_ms"])
                for r in rows if r["algo"] == "Flow Matching"]
    diff_lat = [(s, lat_rows[s]["latency_ms_mean"]) for s in (5, 10, 20, 50, 100)]

    def fit(pairs):
        n = len(pairs)
        sx = sum(x for x, _ in pairs)
        sy = sum(y for _, y in pairs)
        sxx = sum(x * x for x, _ in pairs)
        sxy = sum(x * y for x, y in pairs)
        k = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        b = (sy - k * sx) / n
        return {"slope_ms_per_step": k, "fixed_overhead_ms": b}

    summary = {
        "ablation_table": rows,
        "latency_fit": {"flow": fit(flow_lat), "diffusion": fit(diff_lat)},
        "diffusion_all_latency": diff_lat,
        "stability": json.loads((R / "stability.json").read_text(encoding="utf-8")),
        "smoothness": json.loads((R / "smoothness.json").read_text(encoding="utf-8")),
    }
    (R / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== 消融主表 ===")
    for r in rows:
        print(f"{r['algo']:17s} N={r['infer_steps']:2d}  succ={r['success_pct']:4.1f}% "
              f"CI[{r['success_ci95']}]  cov={r['peak_coverage']:.3f}  "
              f"sumR={r['avg_sum_reward']:6.1f}  ms={r['replan_ms']:7.2f}  "
              f"ms/step={r['per_step_ms']:.3f}")
    print("=== 延迟线性拟合 ===", json.dumps(summary["latency_fit"], indent=2))
    print("=== Diffusion 全档延迟 ===", diff_lat)


if __name__ == "__main__":
    main()
