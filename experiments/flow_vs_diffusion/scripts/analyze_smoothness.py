"""动作平滑度对比：从两种算法的轨迹视频提取运动能量曲线并量化顿挫。

统一口径：逐帧灰度化到 160x160，帧间平均绝对差 = 运动能量 d_t；
一阶差分 = 加速度 proxy，二阶差分 = jerk proxy（抖动/顿挫）。
另外按各自动作块长度（Flow K=8、Diffusion n_action_steps=32）比较块边界
与块内的 jerk，验证"开环块边界抖动"这一共性失效模式。
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
ASSETS = REPO / "assets"

import cv2
import numpy as np

OUT = BASE / "results"

VIDEOS = {
    "flow_matching_1step": {
        "path": ASSETS / "flow_1step_best.mp4",
        "chunk": 8,
    },
    "diffusion_policy_success": {
        "path": ASSETS / "diffusion_baseline_success.mp4",
        "chunk": 32,
    },
}


def read_gray(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2GRAY)
        out.append(g.astype(np.float32))
    cap.release()
    return out


def metrics(gray, chunk):
    d = np.asarray([np.abs(gray[i] - gray[i - 1]).mean() for i in range(1, len(gray))])
    accel = np.diff(d)
    jerk = np.abs(np.diff(accel))
    idx = np.arange(len(jerk))
    boundary = (idx + 2) % chunk == 0
    inner = ~boundary
    b_mean = float(jerk[boundary].mean()) if boundary.any() else None
    i_mean = float(jerk[inner].mean()) if inner.any() else None
    return {
        "n_frames": len(gray),
        "chunk_size": chunk,
        "motion_mean": float(d.mean()),
        "motion_std": float(d.std()),
        "accel_abs_mean": float(np.abs(accel).mean()),
        "jerk_abs_mean": float(jerk.mean()),
        "jerk_p95": float(np.percentile(jerk, 95)),
        "spike_ratio_gt3x_median": float((jerk > 3 * np.median(jerk)).mean()),
        "boundary_jerk_mean": b_mean,
        "inner_jerk_mean": i_mean,
        "boundary_over_inner": (b_mean / i_mean) if (b_mean and i_mean) else None,
    }


def contact_sheet(videos, out_path, n=6):
    rows = []
    for name, item in videos.items():
        gray = item["gray"]
        picks = np.linspace(0, len(gray) - 1, n).astype(int)
        cells = [cv2.cvtColor(np.clip(gray[i], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                 for i in picks]
        cells = [cv2.resize(c, (200, 200)) for c in cells]
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet)


def main():
    result = {}
    for name, item in VIDEOS.items():
        p = item["path"]
        if not p.exists():
            print(f"skip missing {p}")
            continue
        gray = read_gray(p)
        item["gray"] = gray
        result[name] = metrics(gray, item["chunk"])
        print(name, json.dumps(result[name], indent=2), flush=True)

    actions_path = OUT / "flow_best_actions.json"
    if actions_path.exists():
        a = np.asarray(json.loads(actions_path.read_text(encoding="utf-8"))["actions"])
        vel = np.diff(a, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        result["flow_action_space"] = {
            "n_actions": len(a),
            "action_vel_norm_mean": float(np.linalg.norm(vel, axis=1).mean()),
            "action_accel_norm_mean": float(np.linalg.norm(acc, axis=1).mean()),
            "action_jerk_norm_mean": float(np.linalg.norm(jerk, axis=1).mean()),
            "action_jerk_norm_p95": float(np.percentile(np.linalg.norm(jerk, axis=1), 95)),
        }

    if all("gray" in item for item in VIDEOS.values()):
        prefix = min(len(item["gray"]) for item in VIDEOS.values())
        aligned = {name: metrics(item["gray"][:prefix], item["chunk"])
                   for name, item in VIDEOS.items()}
        result["aligned_prefix"] = {
            "n_frames": prefix,
            "by_algo": aligned,
            "note": "两视频截取相同长度的前缀（轨迹起始段），消除片长与静止尾段差异",
        }
        print("aligned_prefix", json.dumps(result["aligned_prefix"], indent=2), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoothness.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(result) >= 2:
        contact_sheet(VIDEOS, ASSETS / "flow_vs_diffusion_smoothness.jpg")
        print("contact sheet:", ASSETS / "flow_vs_diffusion_smoothness.jpg", flush=True)


if __name__ == "__main__":
    main()
