"""扩散策略失败模式：50ep 同种子四档配对统计 + 3 条代表性失败轨迹。

核心检验：「多步采样累积误差」是否真实存在——同一初始状态下，
把去噪步数从 10 加到 50，峰值覆盖率是否系统性退化。
并区分两类失败：步数敏感（换档可救）vs 状态困难（全档一致失败）。
视频仅存档前 10 条 episode，因此挑选范围限定 episode < 10。
"""
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))
OUT = BASE / "results"
ASSETS = REPO / "assets"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

from frame_metrics import compute_metrics, contact_sheet, read_video  # noqa: E402

STEPS = (5, 10, 20, 50)
N_VIDEO = 10


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(WZH.resolve()))
    except ValueError:
        return Path(p).name


def load(step):
    p = WZH / f"outputs/diffusion_eval_steps{step}" / "eval_info.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["per_task"][0]["metrics"]
    return {
        "success": np.asarray([bool(x) for x in m["successes"]]),
        "max_r": np.asarray([float(x) for x in m["max_rewards"]]),
        "sum_r": np.asarray([float(x) for x in m["sum_rewards"]]),
        "source": rel(p),
    }


data = {s: load(s) for s in STEPS}
n = len(data[10]["max_r"])


def sign_test(a, b, eps=0.02):
    """配对符号检验：忽略 |delta| <= eps 的平局，双侧二项 p 值。"""
    diff = b - a
    up = int((diff > eps).sum())
    down = int((diff < -eps).sum())
    m = up + down
    if m == 0:
        return {"n": 0, "improved": 0, "degraded": 0, "p_two_sided": 1.0}
    k = max(up, down)
    tail = sum(math.comb(m, i) for i in range(k, m + 1)) / (2 ** m)
    return {"n": m, "improved": up, "degraded": down,
            "p_two_sided": round(min(1.0, 2 * tail), 6)}


pairs = []
for a, b in ((5, 10), (10, 20), (20, 50), (10, 50)):
    da, db = data[a]["max_r"], data[b]["max_r"]
    pairs.append({
        "from_steps": a, "to_steps": b,
        "mean_delta": round(float(np.mean(db - da)), 4),
        "median_delta": round(float(np.median(db - da)), 4),
        "mean_abs_delta": round(float(np.mean(np.abs(db - da))), 4),
        "success_from": int(data[a]["success"].sum()),
        "success_to": int(data[b]["success"].sum()),
        **sign_test(da, db),
    })

# 每个 episode 的四档表现：区分「步数敏感」与「状态困难」
per_ep = []
for ep in range(n):
    r = {s: round(float(data[s]["max_r"][ep]), 4) for s in STEPS}
    suc = {s: bool(data[s]["success"][ep]) for s in STEPS}
    per_ep.append({
        "episode": ep,
        "max_r": r,
        "success": suc,
        "n_success": sum(suc.values()),
        "spread": round(max(r.values()) - min(r.values()), 4),
        "mean_r": round(float(np.mean(list(r.values()))), 4),
    })

all_fail = [e for e in per_ep if e["n_success"] == 0]
any_success = [e for e in per_ep if e["n_success"] > 0]
step_sensitive = [e for e in per_ep if e["spread"] > 0.2]

# 挑 3 条有视频的代表性失败轨迹（episode < 10）
pool = [e for e in per_ep if e["episode"] < N_VIDEO]


def pick(score_fn, used):
    cands = [e for e in pool if e["episode"] not in used]
    return max(cands, key=score_fn) if cands else None


used = []
selection = []

p1 = pick(lambda e: e["max_r"][10] - e["max_r"][50], used)
if p1:
    used.append(p1["episode"])
    selection.append({"mode": "high_step_regression", "label": "高步数退化",
                      "episode": p1["episode"], "steps": 50, "detail": p1})

p2 = pick(lambda e: -e["mean_r"], used)
if p2:
    used.append(p2["episode"])
    worst = min(STEPS, key=lambda s: p2["max_r"][s])
    selection.append({"mode": "all_step_failure", "label": "全档一致失败（疑似困难初态）",
                      "episode": p2["episode"], "steps": worst, "detail": p2})

p3 = pick(lambda e: e["max_r"][50] - e["max_r"][10], used)
if p3:
    used.append(p3["episode"])
    selection.append({"mode": "low_step_failure", "label": "低步数不足",
                      "episode": p3["episode"], "steps": 10, "detail": p3})

rows = []
for item in selection:
    ep, s = item["episode"], item["steps"]
    src = WZH / f"outputs/diffusion_eval_steps{s}/videos/PushT-v0_0/eval_episode_{ep}.mp4"
    if not src.exists():
        print("missing video:", rel(src), flush=True)
        continue
    frames = read_video(src)
    metrics, thumbs = compute_metrics(frames, chunk=32)
    dst = ASSETS / f"failure_diffusion_{item['mode']}_s{s}_ep{ep}.mp4"
    shutil.copy2(src, dst)
    item["video"] = f"assets/{dst.name}"
    item["source_video"] = rel(src)
    item["metrics"] = {k: v for k, v in metrics.items() if k != "per_frame"}
    item["per_frame"] = metrics["per_frame"]
    short = {"high_step_regression": "高步数退化",
             "all_step_failure": "全档位失败",
             "low_step_failure": "低步数不足"}[item["mode"]]
    rows.append({"label": f"扩散 {s}步 ep{ep}\n{short}", "thumbs": thumbs})
    print(f"selected {item['mode']}: steps={s} ep={ep} "
          f"max_r={item['detail']['max_r'][s]} frames={metrics['n_frames']}", flush=True)

sheet = ASSETS / "failure_diffusion_contact_sheet.jpg"
if rows:
    contact_sheet(rows, sheet)

report = {
    "algo": "diffusion_policy",
    "n_episodes": n,
    "steps": list(STEPS),
    "sources": {s: data[s]["source"] for s in STEPS},
    "per_step_summary": {str(s): {
        "n_success": int(data[s]["success"].sum()),
        "pc_success": round(float(data[s]["success"].mean()) * 100, 2),
        "avg_max_reward": round(float(data[s]["max_r"].mean()), 4),
    } for s in STEPS},
    "pairwise": pairs,
    "episode_level": {
        "n_all_step_failure": len(all_fail),
        "n_any_success": len(any_success),
        "n_step_sensitive_spread_gt_0_2": len(step_sensitive),
        "mean_spread": round(float(np.mean([e["spread"] for e in per_ep])), 4),
        "success_concentration": [e["episode"] for e in any_success][:20],
    },
    "selected_failures": selection,
    "per_episode": per_ep,
    "contact_sheet": f"assets/{sheet.name}" if rows else None,
}
(OUT / "diffusion_failure_analysis.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote results/diffusion_failure_analysis.json", flush=True)
print(json.dumps({"pairwise": pairs, "episode_level": report["episode_level"]},
                 ensure_ascii=False, indent=2), flush=True)
