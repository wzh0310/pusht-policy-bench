"""录制 Flow Matching 最佳轨迹视频 + 动作序列，用于动作平滑度分析。

从步数消融结果里挑表现最好的 episode，用完全相同的策略重放一次，
逐帧渲染成 mp4，并保存每一步的动作向量。
"""

import argparse
import json
import sys
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
    ap.add_argument("--ablation-json", default=str(BASE / "results/flow_steps1.json"))
    ap.add_argument("--infer-steps", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out-video", default=str(ASSETS / "flow_1step_best.mp4"))
    ap.add_argument("--out-actions", default=str(BASE / "results/flow_best_actions.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.ablation_json).read_text(encoding="utf-8"))
    eps = data["per_episode"]
    best = max(eps, key=lambda e: (e["success"], e["max_reward"], e["sum_reward"]))
    ep = best["episode"]
    seed = args.seed_base + ep
    print(f"selected episode={ep} seed={seed} max_reward={best['max_reward']:.3f} "
          f"success={best['success']}", flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    norm = {k: v.cuda() for k, v in ckpt["normalization"].items()}
    kwargs = dict(ckpt["model_kwargs"])
    kwargs["infer_steps"] = args.infer_steps
    model = FlowPolicy(**kwargs).cuda().eval()
    model.load_state_dict(ckpt["model_state_dict"])

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", visualization_width=384,
                   visualization_height=384, max_episode_steps=args.max_steps)

    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    frames, actions = [], []
    step = 0
    while step < args.max_steps:
        img = torch.from_numpy(np.ascontiguousarray(obs["pixels"])).float()
        img = img.permute(2, 0, 1).cuda() / 255.0
        img = F.interpolate(img[None], size=(96, 96), mode="bilinear", align_corners=False)[0]
        img = (img - norm["img_mean"]) / norm["img_std"]
        state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).squeeze().cuda()
        state = (state - norm["state_mean"]) / norm["state_std"]
        with torch.no_grad():
            chunk = model.sample(img[None], state[None])[0]
        chunk = (chunk * norm["action_std"] + norm["action_mean"]).clamp(0, 512)
        for a in chunk:
            frame = env.render()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            actions.append([float(x) for x in a.cpu().numpy()])
            obs, _, _, _, _ = env.step(a.cpu().numpy())
            step += 1
            if step >= args.max_steps:
                break

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()

    Path(args.out_actions).write_text(json.dumps({
        "algo": "flow_matching", "infer_steps": args.infer_steps,
        "episode": ep, "seed": seed, "max_reward": best["max_reward"],
        "success": best["success"], "chunk_size": kwargs["horizon"],
        "actions": actions,
    }, indent=2), encoding="utf-8")
    print(f"video={args.out_video} frames={len(frames)} actions={len(actions)}", flush=True)


if __name__ == "__main__":
    main()
