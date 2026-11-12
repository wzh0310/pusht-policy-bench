"""Flow Policy 在线评估：gym_pusht PushT-v0 闭环 rollout（动作块开环执行）。"""

import argparse
import json

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from flow_model import FlowPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/flow/checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    n = {k: v.cuda() for k, v in ckpt["normalization"].items()}
    model = FlowPolicy(**ckpt["model_kwargs"]).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    K = ckpt["model_kwargs"]["horizon"]

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                   visualization_width=384, visualization_height=384, max_episode_steps=args.max_steps)

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        rewards = []
        step = 0
        while step < args.max_steps:
            img = torch.from_numpy(np.ascontiguousarray(obs["pixels"])).float().permute(2, 0, 1).cuda() / 255.0
            img = F.interpolate(img[None], size=(96, 96), mode="bilinear", align_corners=False)[0]
            img = (img - n["img_mean"]) / n["img_std"]
            state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).squeeze().cuda()
            state = (state - n["state_mean"]) / n["state_std"]
            chunk = model.sample(img[None], state[None])[0]  # (K, A)
            chunk = (chunk * n["action_std"] + n["action_mean"]).clamp(0, 512)
            for a in chunk:  # 开环执行整块后重规划
                obs, r, term, trunc, info = env.step(a.cpu().numpy())
                rewards.append(float(r))
                step += 1
                if step >= args.max_steps:
                    break
        success = bool(info.get("is_success", False))
        results.append({"episode": ep, "sum_reward": float(np.sum(rewards)),
                        "max_reward": float(np.max(rewards)), "success": success})
        print(f"ep {ep}: success={success} max_r={max(rewards):.3f} sum={np.sum(rewards):.1f}")

    n_succ = sum(r["success"] for r in results)
    summary = {"n_episodes": args.episodes, "n_success": n_succ,
               "pc_success": n_succ / args.episodes,
               "avg_sum_reward": float(np.mean([r["sum_reward"] for r in results])),
               "avg_max_reward": float(np.mean([r["max_reward"] for r in results]))}
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump({"summary": summary, "per_episode": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
