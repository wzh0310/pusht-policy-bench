"""BC 在线评估：gym_pusht PushT-v0 闭环 rollout。

BC 不是 lerobot 策略，无法走 lerobot-eval，这里直接:
  obs('pixels' + 'agent_pos') -> 与训练一致的归一化 -> BC 前向 -> 反归一化动作 -> env.step

用法:
    python eval_env.py --episodes 20 --out results.json --video best.mp4
"""

import argparse
import json

import gymnasium as gym
import gym_pusht  # noqa: F401  注册 PushT-v0
import numpy as np
import torch
import torch.nn.functional as F
from model import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--video", default=None, help="保存最佳 episode 的 mp4")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    norm = {k: v.cuda() for k, v in ckpt["normalization"].items()}
    model = BCPolicy(**ckpt["model_kwargs"]).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=args.max_steps,
    )

    results, best = [], None
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames, rewards = [], []
        for _ in range(args.max_steps):
            img = torch.from_numpy(np.ascontiguousarray(obs["pixels"])).float().permute(2, 0, 1) / 255.0
            img = F.interpolate(img[None], size=(96, 96), mode="bilinear", align_corners=False)[0].cuda()
            img = (img - norm["img_mean"]) / norm["img_std"]
            state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).squeeze().cuda()
            state = (state - norm["state_mean"]) / norm["state_std"]
            with torch.no_grad():
                action = model(img.cuda()[None], state.cuda()[None])[0]
            action = (action * norm["action_std"] + norm["action_mean"]).clamp(0, 512)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            frames.append(np.ascontiguousarray(obs["pixels"]))
            rewards.append(float(reward))
        success = bool(info.get("is_success", False))
        results.append({
            "episode": ep,
            "sum_reward": float(np.sum(rewards)),
            "max_reward": float(np.max(rewards)),
            "success": success,
        })
        print(f"ep {ep}: success={success} max_coverage={max(rewards):.3f} sum={np.sum(rewards):.1f}")
        score = np.sum(rewards)
        if best is None or score > best[0]:
            best = (score, frames, success)

    n_succ = sum(r["success"] for r in results)
    summary = {
        "n_episodes": args.episodes,
        "n_success": n_succ,
        "pc_success": n_succ / args.episodes,
        "avg_sum_reward": float(np.mean([r["sum_reward"] for r in results])),
        "avg_max_reward": float(np.mean([r["max_reward"] for r in results])),
        "checkpoint": args.checkpoint,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_episode": results}, f, indent=2)

    if args.video and best is not None:
        import av
        _, frames, _ = best
        container = av.open(args.video, "w")
        stream = container.add_stream("h264", rate=10)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        for frame in frames[::3]:  # 10Hz -> 3.3Hz，控制体积
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print(f"最佳 episode 视频已保存: {args.video}")


if __name__ == "__main__":
    main()
