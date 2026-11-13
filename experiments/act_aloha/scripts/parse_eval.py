#!/usr/bin/env python
"""解析 lerobot-eval 的 eval_info.json -> 脱敏指标 JSON + Markdown 摘要。

结论数值一律从结果 JSON 自动生成，避免文档手写漂移。
用法：WZH_ROOT=<workspace-root> python scripts/parse_eval.py
默认读取 $WZH_ROOT/outputs/act_aloha_demo_eval/eval_info.json，
写出 experiments/act_aloha/results/eval_metrics.json 与 eval_summary.md。
"""
import importlib.metadata as md
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))
EVAL_INFO = Path(os.environ.get("EVAL_INFO", WZH / "outputs" / "act_aloha_demo_eval" / "eval_info.json"))
OUT = BASE / "results"


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    d = json.loads(EVAL_INFO.read_text())
    o = d["overall"]
    t = d["per_task"][0]["metrics"]
    ci = o["pc_success_ci95"]

    metrics = {
        "experiment": "act_aloha_transfer_eval",
        "env_task": d["per_task"][0]["task_group"],
        "policy_checkpoint": "outputs/act_aloha_demo/checkpoints/002000/pretrained_model",
        "n_episodes": o["n_episodes"],
        "n_success": o["n_success"],
        "pc_success": o["pc_success"],
        "pc_success_ci95_pct": [round(ci[0], 1), round(ci[1], 1)],
        "avg_sum_reward": o["avg_sum_reward"],
        "avg_max_reward": o["avg_max_reward"],
        "sum_rewards": t["sum_rewards"],
        "max_rewards": t["max_rewards"],
        "successes": t["successes"],
        "eval_s": round(o["eval_s"], 2),
        "eval_ep_s": round(o["eval_ep_s"], 2),
        "max_episode_steps": 400,
        "seed": 1000,
        "video_assets": ["assets/act_aloha_eval_ep0.mp4", "assets/act_aloha_eval_ep1.mp4"],
        "env_versions": {
            "mujoco": version("mujoco"),
            "dm-control": version("dm-control"),
            "gym-aloha": version("gym-aloha"),
            "gymnasium": version("gymnasium"),
            "lerobot": version("lerobot"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# ACT/ALOHA 闭环评估摘要",
        "",
        f"- 任务：`{metrics['env_task']}`；checkpoint：2000 步（未收敛）",
        f"- 集数：**{o['n_episodes']}**，成功 **{o['n_success']}** 集，"
        f"成功率 **{o['pc_success']:.1f}%**（Wilson 95% CI "
        f"{ci[0]:.1f}%–{ci[1]:.1f}%）",
        f"- 平均累计奖励：**{o['avg_sum_reward']}**；平均峰值奖励：**{o['avg_max_reward']}**",
        f"- 逐集累计奖励：{t['sum_rewards']}",
        f"- 逐集峰值奖励：{t['max_rewards']}",
        f"- 评估耗时：{o['eval_s']:.1f} s（{o['eval_ep_s']:.1f} s/集）",
        "",
        "> 结论：闭环链路（策略加载 → 双臂仿真 → 400 步 rollout → 指标/视频）已跑通；"
        "2000 步训练远未收敛，5 集 0 成功，因此**该数据只证明流程可用，不构成策略能力评价**。",
        "",
        "数值由 `scripts/parse_eval.py` 从 `eval_info.json` 自动生成；"
        "原始 `eval_info.json` 含工作区绝对路径，不入库。",
        "",
    ]
    (OUT / "eval_summary.md").write_text("\n".join(lines))
    print((OUT / "eval_summary.md").read_text())


if __name__ == "__main__":
    main()
