"""Diffusion Policy 推理步数-延迟微基准（与 Flow 消融同机 A6000、同 batch=1 口径）。

加载既有 step-3840 基线 checkpoint，逐个设置 num_inference_steps，
测量一次 predict_action_chunk（= 一次重规划）的 GPU 同步墙钟耗时。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# --- 可移植路径：仓库内相对定位；外部产物可用 WZH_ROOT 覆盖 ---
import os
HERE = Path(__file__).resolve().parent            # experiments/flow_vs_diffusion/scripts
BASE = HERE.parent                                # experiments/flow_vs_diffusion
REPO = BASE.parent.parent                         # pusht-policy-bench 根目录
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))  # 默认：仓库上一级目录
OUT = BASE / "results"

def _rel_to_workspace(p):
    """结果文件里只存相对工作区的路径，不暴露机器绝对路径。"""
    q = Path(p)
    try:
        return str(q.resolve().relative_to(WZH.resolve()))
    except ValueError:
        return str(q)
ASSETS = REPO / "assets"

import numpy as np
import torch

sys.path.insert(0, str(WZH / "lerobot/src"))
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402

CKPT = str(WZH / "outputs/diffusion_baseline/checkpoints/003840/pretrained_model")


def make_batch(policy, device):
    """按 checkpoint 的 input_features 构造同形 batch。"""
    b, s = 1, policy.config.n_obs_steps
    batch = {}
    for key, feat in policy.config.input_features.items():
        shape = tuple(feat.shape)
        if key == "observation.image":
            batch[key] = torch.rand(b, s, *shape, device=device)
        else:
            batch[key] = torch.rand(b, s, *shape, device=device)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    args = ap.parse_args()

    policy = DiffusionPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
    print("config: n_obs_steps=%s horizon=%s n_action_steps=%s image_features=%s" % (
        policy.config.n_obs_steps, policy.config.horizon,
        policy.config.n_action_steps, policy.config.image_features), flush=True)
    batch = make_batch(policy, "cuda")
    print("batch:", {k: tuple(v.shape) for k, v in batch.items()}, flush=True)

    rows = []
    for n in args.steps:
        policy.diffusion.num_inference_steps = n
        with torch.no_grad():
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        lat = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(lat)
        row = {
            "algo": "diffusion_policy",
            "infer_steps": n,
            "n_repeats": args.repeats,
            "latency_ms_mean": float(arr.mean()),
            "latency_ms_p50": float(np.percentile(arr, 50)),
            "latency_ms_p95": float(np.percentile(arr, 95)),
            "latency_ms_per_denoise_step": float(arr.mean() / n),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diffusion_latency.json").write_text(
        json.dumps({"checkpoint": _rel_to_workspace(args.checkpoint), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
