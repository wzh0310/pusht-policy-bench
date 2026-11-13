"""Flow Matching 推理步数消融评估。

在既有 7680 步同协议 checkpoint 上，只改 Euler 积分步数（--infer-steps），
其余（模型权重、归一化、种子、50ep x 300 步、开环执行整块后重规划）全部固定。
同时统计每次重规划的采样延迟，用于与 Diffusion 的速度对比。
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

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(REPO / "baselines/flow"))
from flow_model import FlowPolicy  # noqa: E402

CKPT = str(WZH / "bc/outputs/flow/checkpoint.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-steps", type=int, required=True)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    norm = {k: v.cuda() for k, v in ckpt["normalization"].items()}
    kwargs = dict(ckpt["model_kwargs"])
    kwargs["infer_steps"] = args.infer_steps
    model = FlowPolicy(**kwargs).cuda().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    horizon = kwargs["horizon"]

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=args.max_steps,
    )

    dummy_img = torch.zeros(1, 3, 96, 96, device="cuda")
    dummy_state = torch.zeros(1, kwargs["state_dim"], device="cuda")
    with torch.no_grad():
        model.sample(dummy_img, dummy_state)
    torch.cuda.synchronize()

    latencies, results = [], []
    for ep in range(args.episodes):
        torch.manual_seed(args.seed + ep)
        np.random.seed(args.seed + ep)
        obs, _ = env.reset(seed=args.seed + ep)
        rewards = []
        step = 0
        info = {}
        while step < args.max_steps:
            img = torch.from_numpy(np.ascontiguousarray(obs["pixels"])).float()
            img = img.permute(2, 0, 1).cuda() / 255.0
            img = F.interpolate(img[None], size=(96, 96), mode="bilinear", align_corners=False)[0]
            img = (img - norm["img_mean"]) / norm["img_std"]
            state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).squeeze().cuda()
            state = (state - norm["state_mean"]) / norm["state_std"]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                chunk = model.sample(img[None], state[None])[0]
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            chunk = (chunk * norm["action_std"] + norm["action_mean"]).clamp(0, 512)
            for a in chunk:
                obs, r, term, trunc, info = env.step(a.cpu().numpy())
                rewards.append(float(r))
                step += 1
                if step >= args.max_steps:
                    break

        success = bool(info.get("is_success", False))
        results.append({
            "episode": ep,
            "sum_reward": float(np.sum(rewards)),
            "max_reward": float(np.max(rewards)),
            "success": success,
        })
        print(f"[steps={args.infer_steps}] ep {ep:02d} success={success} "
              f"max_r={np.max(rewards):.3f} sum={np.sum(rewards):.1f}", flush=True)

    lat = np.asarray(latencies)
    n_succ = sum(r["success"] for r in results)
    summary = {
        "algo": "flow_matching",
        "infer_steps": args.infer_steps,
        "checkpoint": _rel_to_workspace(args.checkpoint),
        "n_episodes": args.episodes,
        "seed": args.seed,
        "horizon": horizon,
        "n_success": n_succ,
        "pc_success": n_succ / args.episodes,
        "avg_max_reward": float(np.mean([r["max_reward"] for r in results])),
        "avg_sum_reward": float(np.mean([r["sum_reward"] for r in results])),
        "n_replans": int(lat.size),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_per_euler_step": float(lat.mean() / args.infer_steps),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_episode": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
