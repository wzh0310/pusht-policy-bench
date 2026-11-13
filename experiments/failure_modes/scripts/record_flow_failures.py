"""流匹配失败轨迹录制：3 条代表性失败 + 同种子 1/10 步对照。

用既有 7680 步同协议 checkpoint，完全复刻消融评估的 seed 与开环执行协议，
逐帧渲染并保存每步动作/奖励。用于检验两个假设：
H3 单步推理细节精度下降；H4 高速动态场景适配性不足。
"""
import json
import os
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))
OUT = BASE / "results"
ASSETS = REPO / "assets"
ABL = REPO / "experiments/flow_vs_diffusion/results"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "baselines/flow"))
from flow_model import FlowPolicy  # noqa: E402

from frame_metrics import compute_metrics, contact_sheet  # noqa: E402

CKPT = str(WZH / "bc/outputs/flow/checkpoint.pt")
MAX_STEPS = 300
BASE_SEED = 1000


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(WZH.resolve()))
    except ValueError:
        return Path(p).name


abl = {s: json.loads((ABL / f"flow_steps{s}.json").read_text(encoding="utf-8"))
       for s in (1, 3, 5, 10)}
per_ep = {s: {r["episode"]: r for r in abl[s]["per_episode"]} for s in (1, 3, 5, 10)}
n_ep = len(per_ep[1])

# 三个代表性失败初态
cands = []
for ep in range(n_ep):
    r1 = per_ep[1][ep]["max_reward"]
    r10 = per_ep[10][ep]["max_reward"]
    mean4 = float(np.mean([per_ep[s][ep]["max_reward"] for s in (1, 3, 5, 10)]))
    cands.append({"episode": ep, "r1": r1, "r10": r10, "mean4": mean4,
                  "success1": per_ep[1][ep]["success"]})

used = []


def pick(key, reverse=False):
    pool = [c for c in cands if c["episode"] not in used]
    chosen = (min if reverse else max)(pool, key=key)
    used.append(chosen["episode"])
    return chosen


near_miss = pick(lambda c: c["r1"])
step_gap = pick(lambda c: c["r10"] - c["r1"])
hard_init = pick(lambda c: c["mean4"], reverse=True)

selection = [
    {"mode": "near_miss", "label": "接近成功但未达标", "episode": near_miss["episode"]},
    {"mode": "step_sensitive", "label": "10 步明显优于 1 步", "episode": step_gap["episode"]},
    {"mode": "hard_init", "label": "全档位低覆盖（困难初态）", "episode": hard_init["episode"]},
]
print("selected:", [(x["mode"], x["episode"]) for x in selection], flush=True)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
norm = {k: v.cuda() for k, v in ckpt["normalization"].items()}
actions_all, results, rows = [], [], []

for steps in (1, 10):
    kwargs = dict(ckpt["model_kwargs"])
    kwargs["infer_steps"] = steps
    model = FlowPolicy(**kwargs).cuda().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    chunk_size = kwargs["horizon"]

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", visualization_width=384,
                   visualization_height=384, max_episode_steps=MAX_STEPS)

    for item in selection:
        ep, seed = item["episode"], BASE_SEED + item["episode"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs, _ = env.reset(seed=seed)
        initial_state = [float(x) for x in np.asarray(obs["agent_pos"]).reshape(-1)]
        frames, actions, rewards = [], [], []
        step = 0
        while step < MAX_STEPS:
            img = torch.from_numpy(np.ascontiguousarray(obs["pixels"])).float()
            img = img.permute(2, 0, 1).cuda() / 255.0
            img = F.interpolate(img[None], size=(96, 96), mode="bilinear",
                                align_corners=False)[0]
            img = (img - norm["img_mean"]) / norm["img_std"]
            state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).squeeze().cuda()
            state = (state - norm["state_mean"]) / norm["state_std"]
            with torch.no_grad():
                chunk = model.sample(img[None], state[None])[0]
            chunk = (chunk * norm["action_std"] + norm["action_mean"]).clamp(0, 512)
            for a in chunk:
                frames.append(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
                actions.append([float(x) for x in a.cpu().numpy()])
                obs, r, term, trunc, info = env.step(a.cpu().numpy())
                rewards.append(float(r))
                step += 1
                if step >= MAX_STEPS:
                    break

        a = np.asarray(actions, dtype=np.float64)
        vel = np.linalg.norm(np.diff(a, axis=0), axis=1)
        acc = np.diff(vel)
        jerk = np.diff(acc)
        p90 = float(np.percentile(vel, 90))
        high = vel > p90
        r = np.asarray(rewards)
        boundary = [abs(x) for i, x in enumerate(jerk) if (i + 3) % chunk_size == 0]
        inner = [abs(x) for i, x in enumerate(jerk) if (i + 3) % chunk_size != 0]

        video = ASSETS / (f"failure_flow_{steps}step_ep{ep}.mp4" if steps == 1
                          else f"control_flow_{steps}step_ep{ep}.mp4")
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()

        metrics, thumbs = compute_metrics(frames, chunk=chunk_size)
        rec = {
            "mode": item["mode"], "label": item["label"], "episode": ep, "seed": seed,
            "infer_steps": steps, "chunk_size": chunk_size,
            "success": bool(info.get("is_success", False)),
            "max_reward": round(float(r.max()), 4),
            "sum_reward": round(float(r.sum()), 2),
            "peak_step": int(np.argmax(r)) + 1,
            "final_reward": round(float(r[-1]), 4),
            "post_peak_drop": round(float(r.max() - r[-1]), 4),
            "initial_state": initial_state,
            "video": f"assets/{video.name}",
            "checkpoint": rel(CKPT),
            "action_metrics": {
                "vel_mean": round(float(vel.mean()), 3),
                "vel_p95": round(float(np.percentile(vel, 95)), 3),
                "acc_mean": round(float(np.abs(acc).mean()), 3),
                "jerk_mean": round(float(np.abs(jerk).mean()), 3),
                "jerk_p95": round(float(np.percentile(np.abs(jerk), 95)), 3),
                "high_speed_ratio": round(float(high.mean()), 4),
                "delta_reward_high_speed": round(float(np.diff(r)[high].mean()), 5)
                if high.any() else None,
                "delta_reward_low_speed": round(float(np.diff(r)[~high].mean()), 5),
                "boundary_jerk_mean": round(float(np.mean(boundary)), 3) if boundary else None,
                "inner_jerk_mean": round(float(np.mean(inner)), 3) if inner else None,
                "stall_ratio": round(float((vel < 1.0).mean()), 4),
            },
            "frame_metrics": {k: v for k, v in metrics.items() if k != "per_frame"},
            "per_frame": metrics["per_frame"],
        }
        results.append(rec)
        actions_all.append({"mode": item["mode"], "episode": ep,
                            "infer_steps": steps, "actions": actions,
                            "rewards": [round(float(x), 5) for x in rewards]})
        short = {"near_miss": "近成功未达标",
                 "step_sensitive": "步数敏感",
                 "hard_init": "困难初态"}[item["mode"]]
        rows.append({"label": f"Flow {steps}步 ep{ep}\n{short}", "thumbs": thumbs})
        print(f"steps={steps} ep={ep} max_r={rec['max_reward']:.3f} "
              f"success={rec['success']} frames={len(frames)}", flush=True)
    env.close()

contact_sheet(rows, ASSETS / "failure_flow_contact_sheet.jpg")
(OUT / "flow_failure_analysis.json").write_text(
    json.dumps({"algo": "flow_matching", "base_seed": BASE_SEED,
                "max_steps": MAX_STEPS, "selection": selection,
                "trajectories": results,
                "contact_sheet": "assets/failure_flow_contact_sheet.jpg"},
               ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "flow_failure_actions.json").write_text(
    json.dumps(actions_all, ensure_ascii=False), encoding="utf-8")
print("wrote results/flow_failure_analysis.json + flow_failure_actions.json", flush=True)
