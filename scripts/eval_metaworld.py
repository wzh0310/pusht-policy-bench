"""ACT 在 MetaWorld push-v3 上的自定义闭环评估。

背景：训练时数据集中有 39 维 observation.environment_state（keypoints），
lerobot 的 metaworld 环境封装不暴露该特征，lerobot-eval 无法通过校验。
本脚本手动完成归一化（直接读 checkpoint 里的 normalizer 统计量），
把环境渲染图 + agent_pos + 底层 metaworld 的 39 维观测一起喂给策略。

用法:
    python eval_metaworld.py --checkpoint outputs/act_metaworld/checkpoints/last/pretrained_model \
        --episodes 20 --out results.json --video best.mp4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.envs.configs import MetaworldEnvConfig
from lerobot.envs.factory import make_env
from lerobot.policies.act.modeling_act import ACTPolicy


def find_raw_env(vec_env):
    """本进程不直接访问 vec env 内部（worker 进程隔离）。
    改为创建一个同任务、同种子的 '孪生环境'，用相同的动作序列逐步推进，
    其 39 维观测即 observation.environment_state（MuJoCo 确定性保证一致）。"""
    raise RuntimeError("use twin env instead")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="metaworld-push-v3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = MetaworldEnvConfig(task=args.task, fps=80)
    envs = make_env(cfg)
    vec = list(envs[args.task].values())[0]

    # 孪生环境：只用于提供 39 维 environment_state
    import metaworld
    task_name = args.task.replace("metaworld-", "")
    ml1 = metaworld.ML1(task_name)
    twin = ml1.train_classes[task_name]()
    twin.set_task(list(ml1.test_tasks)[0])
    print("twin env:", type(twin).__name__, "obs dim:", twin.observation_space.shape)

    # 注意：必须先完成环境（EGL 初始化）再初始化 CUDA，否则 nvidia EGL 设备枚举失败
    policy = ACTPolicy.from_pretrained(str(ckpt), device="cuda")
    policy.eval()

    # 手动归一化：读 checkpoint 内 preprocessor 的统计量
    pre = load_file(str(ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    post = load_file(str(ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))

    def norm(key, x):
        mean = pre[f"{key}.mean"].cuda().float()
        std = pre[f"{key}.std"].cuda().float()
        return (x - mean) / (std + 1e-8)  # 与 lerobot NormalizeProcessor 一致，防止 0/0

    action_mean = post["action.mean"].cuda().float()
    action_std = post["action.std"].cuda().float()

    cfg = MetaworldEnvConfig(task=args.task, fps=80)
    envs = make_env(cfg)
    vec = list(envs[args.task].values())[0]

    # 孪生环境：只用于提供 39 维 environment_state
    import metaworld
    task_name = args.task.replace("metaworld-", "")
    ml1 = metaworld.ML1(task_name)
    twin = ml1.train_classes[task_name]()
    twin.set_task(list(ml1.test_tasks)[0])
    print("twin env:", type(twin).__name__, "obs dim:", twin.observation_space.shape)

    results, best = [], None
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = vec.reset(seed=seed)
        twin_obs, _ = twin.reset(seed=seed)
        policy.reset()
        frames, rewards, successes = [], [], []
        for _ in range(args.max_steps):
            o = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    v = v[0] if v.shape[0] == 1 else v
                elif hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == 1:
                    v = v[0]
                o[k] = v
            image = torch.from_numpy(np.ascontiguousarray(o["observation.image"])).float().permute(2, 0, 1).cuda() / 255.0
            image = norm("observation.image", image)
            state = torch.from_numpy(np.asarray(o["observation.state"], dtype=np.float32).squeeze()).cuda()
            state = norm("observation.state", state)
            env_state = torch.from_numpy(np.asarray(twin_obs, dtype=np.float32).squeeze()).cuda()
            env_state = norm("observation.environment_state", env_state)
            batch = {
                "observation.image": image[None],
                "observation.state": state[None],
                "observation.environment_state": env_state[None],
            }
            with torch.no_grad():
                action = policy.select_action(batch)
            action = (action * action_std + action_mean).clamp(-1, 1)
            action_np = action.cpu().numpy().reshape(-1)
            obs, reward, terminated, truncated, info = vec.step(action_np[None])
            twin_obs, twin_r, twin_term, twin_trunc, twin_info = twin.step(action_np)
            r = float(np.asarray(reward).reshape(-1)[0])
            rewards.append(r)
            successes.append(bool(twin_info.get("success", False)))
            frames.append(np.ascontiguousarray(o["observation.image"]))
        success = any(successes)
        results.append({"episode": ep, "sum_reward": float(np.sum(rewards)),
                        "max_reward": float(np.max(rewards)), "success": success})
        print(f"ep {ep}: success={success} max_r={max(rewards):.3f} sum={np.sum(rewards):.1f}")
        score = np.sum(rewards)
        if best is None or score > best[0]:
            best = (score, frames, success)

    n_succ = sum(r["success"] for r in results)
    summary = {
        "n_episodes": args.episodes, "n_success": n_succ,
        "pc_success": n_succ / args.episodes,
        "avg_sum_reward": float(np.mean([r["sum_reward"] for r in results])),
        "avg_max_reward": float(np.mean([r["max_reward"] for r in results])),
        "checkpoint": str(ckpt),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump({"summary": summary, "per_episode": results}, open(args.out, "w"), indent=2)

    if args.video and best is not None:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, best[1][::4], fps=10)
        print(f"最佳 episode 视频已保存: {args.video}")


if __name__ == "__main__":
    main()
