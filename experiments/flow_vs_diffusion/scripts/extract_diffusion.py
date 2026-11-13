"""从既有 Diffusion 步数消融的 eval_info.json 提取同口径量化结果。

数据源：$WZH_ROOT/outputs/diffusion_eval_steps{5,10,20,50}/eval_info.json
口径：50 episodes x 300 步，与 Flow 消融一致（成功率 = successes 均值，
峰值覆盖率 = max_rewards 均值）。
"""

import json
from pathlib import Path

# --- 可移植路径：仓库内相对定位；外部产物可用 WZH_ROOT 覆盖 ---
import os
HERE = Path(__file__).resolve().parent            # experiments/flow_vs_diffusion/scripts
BASE = HERE.parent                                # experiments/flow_vs_diffusion
REPO = BASE.parent.parent                         # pusht-policy-bench 根目录
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))  # 默认：仓库上一级目录
OUT = BASE / "results"

def _sanitize(obj):
    """递归剥离工作区绝对路径，结果文件里只留相对路径。"""
    prefix = str(WZH.resolve()) + "/"
    if isinstance(obj, str):
        return obj.replace(prefix, "")
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj

def _rel_to_workspace(p):
    """结果文件里只存相对工作区的路径，不暴露机器绝对路径。"""
    q = Path(p)
    try:
        return str(q.resolve().relative_to(WZH.resolve()))
    except ValueError:
        return str(q)
ASSETS = REPO / "assets"

ROOT = WZH


def main():
    rows = []
    for s in (5, 10, 20, 50):
        p = ROOT / f"outputs/diffusion_eval_steps{s}" / "eval_info.json"
        if not p.exists():
            print(f"skip missing {p}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["per_task"][0]["metrics"]
        succ = [float(x) for x in m["successes"]]
        maxr = [float(x) for x in m["max_rewards"]]
        sumr = [float(x) for x in m["sum_rewards"]]
        rows.append({
            "algo": "diffusion_policy",
            "infer_steps": s,
            "n_episodes": len(succ),
            "n_success": int(sum(succ)),
            "pc_success": sum(succ) / len(succ),
            "avg_max_reward": sum(maxr) / len(maxr),
            "avg_sum_reward": sum(sumr) / len(sumr),
            "video_paths": [_rel_to_workspace(v) for v in m.get("video_paths", [])],
            "source": _rel_to_workspace(p),
            "overall": d.get("overall"),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diffusion_steps.json").write_text(
        json.dumps(_sanitize(rows), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_sanitize(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
