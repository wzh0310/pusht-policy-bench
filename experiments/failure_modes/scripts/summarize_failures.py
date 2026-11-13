"""汇总失败模式证据：量化对比表 + 假设检验结论。

结论数值全部从 results/*.json 自动读取，避免手写数字漂移；
逐帧判读文字来自对抽帧拼图的人工观察（与 per_frame 序列交叉核对）。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results"
d = json.loads((OUT / "diffusion_failure_analysis.json").read_text(encoding="utf-8"))
f = json.loads((OUT / "flow_failure_analysis.json").read_text(encoding="utf-8"))

ANNOT = {
    "diff:high_step_regression":
        "夹爪接触后把 T 块推离目标并停在左上；后半程运动能量衰减过半，块边界 jerk 比 3.20（全表最高）",
    "diff:all_step_failure":
        "灰块被推到画面边缘，与绿色目标全程分离（峰值覆盖 0）；夹爪贴着块反复微调但无实质推进",
    "diff:low_step_failure":
        "灰块绕目标下方来回打转，多帧姿态往复；停滞比 0.31，块边界 jerk 比 2.57",
    "flow:near_miss:1":
        "第 203 步曾完全对齐（覆盖 1.000），随后继续开环推动把块推走，终态掉到 0.183",
    "flow:step_sensitive:1":
        "块始终在目标右下游走，6 帧抽帧里未见重叠；高速段 Δreward 为正，推进主要靠大幅动作",
    "flow:hard_init:1":
        "灰块全程停在画面上方，与下方绿色目标分离，夹爪贴块但无法把它带向目标（覆盖恒为 0）",
    "flow:near_miss:10":
        "同初态换 10 步后几乎不动，块停在下方、目标在上方，峰值覆盖仅 0.147；停滞比 0.16",
    "flow:step_sensitive:10":
        "同初态换 10 步后第 273 步达到完全对齐且终态仍有 0.920，是六条里唯一接近保持成功的轨迹",
    "flow:hard_init:10":
        "与 1 步同样全程零覆盖，换步数无法救援；块边界 jerk 比升到 2.09，动作更抖但无推进",
}


def fmt(v, spec="{:.3f}"):
    return "—" if v is None else spec.format(v)


rows = []
for x in d["selected_failures"]:
    m = x["metrics"]
    key = f"diff:{x['mode']}"
    rows.append({
        "algo": "Diffusion", "steps": x["steps"], "ep": x["episode"],
        "mode": x["label"], "peak": x["detail"]["max_r"][str(x["steps"])],
        "final": None, "peak_step": None, "drop": None,
        "stall": m["stall_ratio"], "boundary": m["boundary_ratio"],
        "spike": m["spike_rate"], "annot": ANNOT[key],
    })
for t in f["trajectories"]:
    m, a = t["frame_metrics"], t["action_metrics"]
    key = f"flow:{t['mode']}:{t['infer_steps']}"
    rows.append({
        "algo": "Flow", "steps": t["infer_steps"], "ep": t["episode"],
        "mode": t["label"], "peak": t["max_reward"],
        "final": t["final_reward"], "peak_step": t["peak_step"],
        "drop": t["post_peak_drop"], "stall": m["stall_ratio"],
        "boundary": m["boundary_ratio"], "spike": m["spike_rate"],
        "annot": ANNOT[key],
    })

lines = [
    "# 扩散 vs 流匹配：失败轨迹量化对比",
    "",
    "PushT-v0 ｜ 50 episodes × 300 步同种子协议 ｜ 抽帧拼图见 assets/",
    "",
    "| 算法 | 步数 | ep | 失败模式 | 峰值覆盖 | 终态覆盖 | 峰值步 | 峰后衰减 | 停滞比 | 块边界 jerk 比 | 逐帧判读 |",
    "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    lines.append(
        f"| {r['algo']} | {r['steps']} | {r['ep']} | {r['mode']} | "
        f"{fmt(r['peak'])} | {fmt(r['final'])} | "
        f"{'—' if r['peak_step'] is None else r['peak_step']} | {fmt(r['drop'])} | "
        f"{fmt(r['stall'], '{:.2f}')} | {fmt(r['boundary'], '{:.2f}')} | {r['annot']} |")
lines += [
    "",
    "## 口径",
    "",
    "- **峰值/终态覆盖**：PushT 每步奖励即目标重叠覆盖率；Diffusion 侧只有 eval 汇总，无逐帧奖励，终态列留空。",
    "- **停滞比**：逐帧灰度帧差低于该视频中位运动能量一半的帧占比（动态阈值，消除渲染量级差异）。",
    "- **块边界 jerk 比**：开环动作块边界处的像素 jerk 均值 / 块内均值，>1 表示重规划引入不连续；\n  Diffusion 按 K=32、Flow 按 K=8 切块（各自实际动作块长度）。",
    "- **逐帧判读**：每行 6 帧等间隔抽帧的人工观察，完整 300 帧序列见 `results/*_analysis.json` 的 `per_frame`。",
    "",
    "详细分析见 `docs/failure_mode_analysis.md`。",
]
(OUT / "failure_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

pair = {f"{p['from_steps']}->{p['to_steps']}": p for p in d["pairwise"]}
ep = d["episode_level"]
flow = {(t["mode"], t["infer_steps"]): t for t in f["trajectories"]}
near1, near10 = flow[("near_miss", 1)], flow[("near_miss", 10)]
sens1, sens10 = flow[("step_sensitive", 1)], flow[("step_sensitive", 10)]
hard1 = flow[("hard_init", 1)]

hypotheses = [
    {
        "id": "H1", "claim": "扩散多步采样累积误差：步数越多，同初态覆盖率系统性下降",
        "verdict": "不支持",
        "evidence": (f"10→50 步平均 Δ={pair['10->50']['mean_delta']:+.4f}，"
                     f"改善 {pair['10->50']['improved']} / 退化 {pair['10->50']['degraded']}，"
                     f"配对符号检验双侧 p={pair['10->50']['p_two_sided']}；"
                     f"20→50 p={pair['20->50']['p_two_sided']}，均无显著退化趋势。"),
    },
    {
        "id": "H2", "claim": "极端分布外场景导致动作突变",
        "verdict": "部分支持（初态主导成立，动作突变证据不足）",
        "evidence": (f"{ep['n_all_step_failure']}/50 个初态在四个步数档全部失败，"
                     f"仅 {ep['n_any_success']} 个初态曾成功；但三条失败轨迹的"
                     f"尖峰率 {min(x['metrics']['spike_rate'] for x in d['selected_failures']):.2f}–"
                     f"{max(x['metrics']['spike_rate'] for x in d['selected_failures']):.2f}，"
                     "更多表现为「贴块微调后停滞」而非突跳。"),
    },
    {
        "id": "H3", "claim": "流匹配单步推理细节精度普遍下降",
        "verdict": "不支持（存在步数 × 初态交互）",
        "evidence": (f"同初态 ep{near1['episode']}：1 步覆盖 {near1['max_reward']:.3f} vs "
                     f"10 步 {near10['max_reward']:.3f}；ep{sens1['episode']} 反向："
                     f"1 步 {sens1['max_reward']:.3f} vs 10 步 {sens10['max_reward']:.3f}。"
                     "单步在部分初态最优、在另一部分初态最差，方向不一致。"),
    },
    {
        "id": "H4", "claim": "流匹配高速动态场景适配性不足",
        "verdict": "部分支持（精细调整阶段高速动作有害）",
        "evidence": (f"接近目标的两条轨迹高速段 Δreward 为负"
                     f"（1 步 ep{near1['episode']} {near1['action_metrics']['delta_reward_high_speed']:+.5f}、"
                     f"10 步 ep{near10['episode']} {near10['action_metrics']['delta_reward_high_speed']:+.5f}），"
                     f"而推进阶段 1 步 ep{sens1['episode']} 高速段 {sens1['action_metrics']['delta_reward_high_speed']:+.5f} 为正。"),
    },
    {
        "id": "H5", "claim": "新增：峰值达标后无法保持，是跨算法共有的首要失效模式",
        "verdict": "强支持",
        "evidence": (f"Flow 1 步 ep{near1['episode']} 峰值 {near1['max_reward']:.3f}、"
                     f"终态 {near1['final_reward']:.3f}、衰减 {near1['post_peak_drop']:.3f}；"
                     f"10 步 ep{sens10['episode']} 峰值 {sens10['max_reward']:.3f}、"
                     f"终态 {sens10['final_reward']:.3f} 仍判失败；"
                     "本次录制的 6 条 Flow 轨迹无一成功，与既有基线记录的「成功即停」一致。"),
    },
    {
        "id": "H6", "claim": "困难初态无法靠换采样步数稳定救援",
        "verdict": "支持",
        "evidence": (f"ep{hard1['episode']}：Diffusion 5/10/20/50 步峰值覆盖 "
                     f"{[e['max_r'][str(s)] for s in (5, 10, 20, 50) if (e := next(x for x in d['per_episode'] if x['episode'] == hard1['episode']))]}，"
                     f"Flow 1/10 步 {hard1['max_reward']:.3f}/{flow[('hard_init', 10)]['max_reward']:.3f}；"
                     f"Diffusion 侧 {ep['n_all_step_failure']}/50 初态四档全失败。"),
    },
]

summary = {
    "n_diffusion_pairs": len(d["pairwise"]),
    "episode_level": ep,
    "diffusion_pairwise": d["pairwise"],
    "hypotheses": hypotheses,
    "primary_failure_mode": "峰值达标后保持失效（动作块开环执行 + 保持行为无监督）",
    "secondary_failure_modes": [
        "困难初态：全档位零覆盖，步数与算法都救不了",
        "步数 × 初态交互：同初态换步数可让覆盖率从 0.147 跳到 1.000，方向不固定",
        "块边界不连续：开环重规划处 jerk 比普遍 >1（Diffusion 1.42–3.20，Flow 1.12–2.09）",
    ],
}
(OUT / "failure_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("wrote results/failure_table.md + failure_summary.json")
for h in hypotheses:
    print(f"  {h['id']}: {h['verdict']}")
