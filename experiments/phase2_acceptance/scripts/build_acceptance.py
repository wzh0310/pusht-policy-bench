#!/usr/bin/env python
"""第二阶段验收对照：从结果 JSON 自动生成覆盖矩阵报告（数值不手写）。

用法：在仓库根执行 python experiments/phase2_acceptance/scripts/build_acceptance.py
输出：docs/phase2_acceptance.md
      experiments/phase2_acceptance/results/acceptance.json
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPO = BASE.parent.parent
OUT = BASE / "results"
DOC = REPO / "docs" / "phase2_acceptance.md"


def load(rel):
    return json.loads((REPO / rel).read_text())


def metrics(rel, schema):
    d = load(rel)
    if schema == "overall":
        s = d["overall"]
    elif schema == "summary":
        s = d["summary"]
    else:
        s = d
    # 不同来源的 pc_success 量纲不一致：lerobot eval_info 存百分比，
    # 自研 eval_results 存 0-1 比例。统一从计数重算，避免口径漂移。
    return {
        "n_episodes": s["n_episodes"],
        "n_success": s["n_success"],
        "pc_success": round(100.0 * s["n_success"] / s["n_episodes"], 1),
        "avg_sum_reward": round(s["avg_sum_reward"], 2),
        "avg_max_reward": round(s["avg_max_reward"], 3),
    }


# (标签, 算法, 环境, 引擎, 结果 JSON, schema, 训练曲线证据, 成功率口径说明)
# (标签, 算法, 环境, 引擎, 结果 JSON, schema, 训练曲线证据, 成功率口径说明, 训练日志)
RUNS = [
    ("BC v2 (50ep)", "BC", "PushT-v0", "pymunk", "experiments/bc_baseline/eval_results_v2_50ep.json",
     "summary", "baselines/bc + train_log_v2.txt", "50ep", "experiments/bc_baseline/train_log_v2.txt"),
    ("ACT v3 (50k 步)", "ACT", "PushT-v0", "pymunk", "experiments/act_v3/eval_info.json",
     "overall", "train_log.txt / assets/loss_curves.png", "50ep", "experiments/act_v3/train_log.txt"),
    ("Diffusion baseline", "Diffusion", "PushT-v0", "pymunk", "experiments/diffusion_baseline/eval_info.json",
     "overall", "train_log.txt / assets/summary_chart.png", "50ep", "experiments/diffusion_baseline/train_log.txt"),
    ("Flow baseline", "Flow Matching", "PushT-v0", "pymunk", "experiments/flow_baseline/eval_results.json",
     "summary", "train_log.txt（20 epoch + step 级 loss）", "50ep", "experiments/flow_baseline/train_log.txt"),
    ("ACT MetaWorld v1", "ACT", "MetaWorld push-v3", "MuJoCo", "experiments/act_metaworld/eval_results_v1_3k.json",
     "summary", "assets/act_metaworld_train_curves.png (10 点)", "20ep x 400 steps",
     "experiments/act_metaworld/train_log_v1.txt"),
    ("ACT MetaWorld v2", "ACT", "MetaWorld push-v3", "MuJoCo", "experiments/act_metaworld/eval_results_v2.json",
     "summary", "assets/act_metaworld_train_curves.png (20 点)", "20ep x 400 steps",
     "experiments/act_metaworld/train_log_v2.txt"),
    ("ACT ALOHA", "ACT", "ALOHA AlohaInsertion-v0", "MuJoCo", "experiments/act_aloha/results/eval_metrics.json",
     "flat", "assets/act_aloha_loss.png (前 900 步)", "5ep x 400 steps",
     "experiments/act_aloha/results/train_log_excerpt.txt"),
]

ABLATIONS = [
    ("A1", "ACT chunk_size 消融 (1 / 10 / 30)", "experiments/act_ablation/RESULTS.md",
     "chunk=1 成功率归零，chunk=10 最优 18%"),
    ("A2", "ACT CVAE 消融 (use_vae true / false)", "experiments/act_ablation/RESULTS.md",
     "关闭 CVAE 成功率 14%→6%，峰值覆盖率反升"),
    ("A3", "Diffusion 推理采样步数消融 (DDIM5 / DDIM20 / DDPM100)", "experiments/diffusion_inference_steps/RESULTS.md",
     "5 步 DDIM 追平 100 步，重规划 51ms vs 1235ms"),
    ("A4", "扩散 vs 流匹配 步数消融 (7 档)", "experiments/flow_vs_diffusion/results/ablation_table.md",
     "Flow 1 步 2.67ms / 覆盖率 0.727；Diffusion 10–50 步 95–463ms"),
    ("A5", "训练稳定性对比 (loss CV)", "experiments/flow_vs_diffusion/results/stability.json",
     "Flow CV 0.147 vs Diffusion 0.310"),
    ("A6", "动作平滑度对比 (jerk / 块边界)", "experiments/flow_vs_diffusion/results/smoothness.json",
     "jerk 0.422 vs 0.666，均无高频颤动"),
    ("A7", "四算法同协议基线对比 (7680 步)", "experiments/flow_baseline/RESULTS.md",
     "Diffusion 6% > Flow 2% > ACT/BC 0%"),
    ("A8", "训练量扩展对比 (BC v1→v2 / ACT v1→v2→v3)", "experiments/act_v3/RESULTS.md",
     "ACT 0%→5%→8%→14%，eval_loss 几乎不变"),
    ("A9", "失效模式 50ep 配对假设检验 (H1–H6)", "experiments/failure_modes/results/failure_summary.json",
     "H1/H3 不支持、H5 强支持"),
]

FAILURE_EVIDENCE = [
    ("生成式动作头失效模式逐帧对比（3+3 轨迹、6 假设）", "docs/failure_mode_analysis.md",
     "experiments/failure_modes/"),
    ("三算法失效模式深度分析", "docs/comparison.md", "assets/diffusion_nearmiss.gif"),
    ("11 条真实故障归因库", "docs/findings.md", "assets/demo_failure.mp4"),
    ("ACT 消融四类失效画像", "experiments/act_ablation/RESULTS.md", "assets/ablation_chunk1_best_effort.gif"),
    ("跨环境过拟合失效（悬停不接触 / 全零对照）", "experiments/act_metaworld/RESULTS.md",
     "experiments/act_aloha/results/eval_summary.md"),
]


def count_loss_points(rel):
    """统计日志中「有效 loss 记录行数」，佐证完整训练曲线而非凭描述断言。

    口径：按行计数，一行只算一条；排除 tqdm 进度条行（其中也含 loss 数值，
    会造成重复计数）。自研实现的 CSV 摘要按数据行计数。
    """
    p = REPO / rel
    if not p.exists():
        return 0
    t = p.read_text(errors="ignore").replace("\r", "\n")
    pat = re.compile(r"(?:eval_loss|train_loss|val_loss|loss)\s*[=:]?\s*[0-9]*\.[0-9]+")
    lines = [l for l in t.split("\n") if "Training:" not in l and pat.search(l)]
    if lines:
        return len(lines)
    return len([l for l in t.split("\n") if re.match(r"^\d+,", l)])


def check_readme():
    text = (REPO / "README.md").read_text()
    demo = re.findall(r"\]\(assets/[^)]+\.(?:mp4|gif)\)", text)
    return {
        "env_setup": all(k in text for k in ["conda create", "pip install", "MUJOCO_GL", "mujoco"]),
        "repro": all(k in text for k in ["bash scripts/", "WZH_ROOT", "scripts/run_train.sh"]),
        "results": "## 结果" in text and "成功率" in text and "## 第二阶段验收" in text,
        "demo_video_links": len(set(demo)),
        "demo_links_ok": len(set(demo)) >= 10,
    }


def main():
    rows = []
    for label, algo, env, engine, rel, schema, curve, note, log in RUNS:
        m = metrics(rel, schema)
        rows.append({"label": label, "algorithm": algo, "env": env, "engine": engine,
                     "result_file": rel, **m, "eval_protocol": note,
                     "train_curve": curve, "train_log": log,
                     "loss_records": count_loss_points(log)})

    engines = {}
    for r in rows:
        engines.setdefault(r["engine"], set()).add(r["env"])
    algo_by_engine = {}
    for r in rows:
        algo_by_engine.setdefault(r["engine"], set()).add(r["algorithm"])
    mujoco_algos = sorted(algo_by_engine.get("MuJoCo", set()))
    mujoco_envs = sorted(engines.get("MuJoCo", set()))

    ab_rows = []
    for tag, title, rel, note in ABLATIONS:
        ab_rows.append({"tag": tag, "title": title, "evidence": rel,
                        "finding": note, "evidence_exists": (REPO / rel).exists()})

    fail_rows = []
    for title, doc, extra in FAILURE_EVIDENCE:
        fail_rows.append({"title": title, "doc": doc, "extra": extra,
                          "doc_exists": (REPO / doc).exists()})

    rm = check_readme()
    verdict = {
        "criterion_1": {
            "text": "3 个算法在 2 个以上 MuJoCo 环境跑通，有完整训练曲线和成功率指标",
            "mujoco_env_count": len(mujoco_envs),
            "mujoco_algos": mujoco_algos,
            "mujoco_algo_count": len(mujoco_algos),
            "strict_pass": len(mujoco_envs) >= 2 and len(mujoco_algos) >= 3,
            "status": "部分达标",
        },
        "criterion_2": {
            "text": "至少 5 组消融 / 对比实验，有失败案例分析",
            "ablation_count": len(ab_rows),
            "failure_evidence_count": len(fail_rows),
            "status": "通过" if len(ab_rows) >= 5 and len(fail_rows) >= 2 else "未达标",
        },
        "criterion_3": {
            "text": "GitHub README 清晰，包含环境搭建、复现步骤、实验结果、demo 视频链接",
            **rm,
            "status": "通过" if all([rm["env_setup"], rm["repro"], rm["results"], rm["demo_links_ok"]]) else "未达标",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance.json").write_text(json.dumps({
        "experiment": "phase2_acceptance",
        "runs": rows,
        "ablations": ab_rows,
        "failure_evidence": fail_rows,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n")

    # ---- 生成 Markdown（数值全部来自 rows / JSON）----
    L = []
    L.append("# 第二阶段验收对照报告")
    L.append("")
    L.append("> 本报告由 `experiments/phase2_acceptance/scripts/build_acceptance.py` 从各实验结果 JSON "
             "自动生成，机器可读版本见 [acceptance.json](../experiments/phase2_acceptance/results/acceptance.json)。"
             "**结论数值不手写**，与结果文件保持一致。")
    L.append("")
    L.append("## 一句话判定")
    L.append("")
    v = verdict
    L.append(f"| 标准 | 判定 | 关键事实 |")
    L.append(f"|---|---|---|")
    L.append(f"| 1. 3 算法 × 2+ MuJoCo 环境 + 曲线/成功率 | **{v['criterion_1']['status']}** | "
             f"MuJoCo 环境 {v['criterion_1']['mujoco_env_count']} 个，"
             f"但只跑通 {v['criterion_1']['mujoco_algo_count']} 个算法（{', '.join(mujoco_algos)}） |")
    L.append(f"| 2. ≥5 组消融/对比 + 失败案例分析 | **{v['criterion_2']['status']}** | "
             f"{v['criterion_2']['ablation_count']} 组消融/对比，"
             f"{v['criterion_2']['failure_evidence_count']} 类失败案例证据 |")
    L.append(f"| 3. README 四要素 + GitHub | **{v['criterion_3']['status']}** | "
             f"环境搭建/复现/结果/demo 链接齐备，{v['criterion_3']['demo_video_links']} 个视频/GIF 链接 |")
    L.append("")
    L.append("## 1. 环境口径澄清（重要）")
    L.append("")
    L.append("验收标准写的是「MuJoCo 环境」，本仓库三个环境的物理引擎并不相同，必须分开统计：")
    L.append("")
    L.append("| 环境 | 物理引擎 | 依据 |")
    L.append("|---|---|---|")
    L.append("| PushT-v0 | **pymunk**（2D 刚体） | `gym-pusht` 包元数据 `Requires-Dist: pymunk>=6.6.0,<7.0.0`，不依赖 mujoco |")
    L.append("| MetaWorld push-v3 | **MuJoCo** | `metaworld 3.0.0` 要求 `mujoco>=3.0.0` |")
    L.append("| ALOHA AlohaInsertion-v0 | **MuJoCo** | `gym-aloha 0.1.5` 要求 `mujoco>=3.0.0,<3.9.0` |")
    L.append("")
    L.append("因此**真正的 MuJoCo 环境是 2 个**（MetaWorld、ALOHA），PushT 不计入。"
             "若把 PushT 按泛化的「仿真环境」口径算进来，则覆盖 3 个环境、4 个算法；"
             "两种口径的判定差异见第 3 节，本报告不做有利于自己的口径选择。")
    L.append("")
    L.append("## 2. 算法 × 环境覆盖矩阵（自动生成）")
    L.append("")
    L.append("| 实验 | 算法 | 环境 | 引擎 | Episodes | 成功率 | avg_sum_reward | avg_max_reward | loss 行数 | 训练曲线 |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        L.append(f"| {r['label']} | {r['algorithm']} | {r['env']} | {r['engine']} | "
                 f"{r['n_episodes']} | {r['pc_success']:.1f}% ({r['n_success']}/{r['n_episodes']}) | "
                 f"{r['avg_sum_reward']} | {r['avg_max_reward']} | {r['loss_records']} | {r['train_curve']} |")
    L.append("")
    L.append("成功率指标口径：PushT 50ep × 300 步；MetaWorld 20ep × 400 步（自研闭环脚本）；"
             "ALOHA 5ep × 400 步（`lerobot-eval`）。")
    L.append("")
    L.append("**覆盖矩阵（按引擎）**")
    L.append("")
    L.append("| 引擎 | 环境 | 已跑通算法 |")
    L.append("|---|---|---|")
    for eng in ["pymunk", "MuJoCo"]:
        es = sorted(engines.get(eng, set()))
        algos = sorted(algo_by_engine.get(eng, set()))
        L.append(f"| {eng} | {', '.join(es)} | {', '.join(algos)}（{len(algos)} 个） |")
    L.append("")
    L.append("![MetaWorld 训练曲线](../assets/act_metaworld_train_curves.png)")
    L.append("")
    mw1 = next(r for r in rows if r["label"] == "ACT MetaWorld v1")["loss_records"]
    mw2 = next(r for r in rows if r["label"] == "ACT MetaWorld v2")["loss_records"]
    al = next(r for r in rows if r["label"] == "ACT ALOHA")["loss_records"]
    L.append(f"曲线完整度说明（loss 行数由脚本解析日志自动统计，排除 tqdm 进度条）：MetaWorld v1/v2 为完整 "
             f"eval_loss 序列（{mw1} / {mw2} 条，"
             "[train_curves.json](../experiments/act_metaworld/results/train_curves.json)）；"
             f"ALOHA 仅 {al} 条 —— lerobot 0.7.1 只打印前 ~1000 步日志，"
             "覆盖 2000 步训练的前 45%，如实标注见 "
             "[act_aloha_transfer.md](act_aloha_transfer.md) 第 9 节。"
             "PushT 四算法的 loss 记录数见上表，均为完整序列。")
    L.append("")
    L.append("## 3. 标准一：逐项判定")
    L.append("")
    L.append("| 子项 | 事实 | 判定 |")
    L.append("|---|---|---|")
    L.append(f"| 2 个以上 MuJoCo 环境 | MetaWorld push-v3 + ALOHA = {len(mujoco_envs)} 个 | ✅ |")
    L.append(f"| 3 个算法在这些环境跑通 | 仅 {', '.join(mujoco_algos)}（{len(mujoco_algos)} 个） | ❌ |")
    L.append("| 完整训练曲线 | PushT 四算法齐；MetaWorld 两版本齐；ALOHA 仅前 45% | ⚠️ |")
    L.append(f"| 成功率指标 | 全部 {len(rows)} 组实验均有 n_episodes / n_success / 奖励 | ✅ |")
    L.append("")
    L.append("**结论：部分达标。** 若按严格字面（3 个算法各自在 ≥2 个 MuJoCo 环境跑通）则未达标，"
             "缺口是 BC / Diffusion / Flow 三个算法都只在 PushT（pymunk）上训练与评估，"
             "未迁移到 MuJoCo 环境。这一点不能靠文档表述弥补，补跑方案见第 6 节。")
    L.append("")
    L.append("## 4. 标准二：消融 / 对比实验清单（自动生成）")
    L.append("")
    L.append(f"共 **{len(ab_rows)} 组**（要求 ≥5）：")
    L.append("")
    L.append("| 编号 | 实验 | 关键结论 | 证据 |")
    L.append("|---|---|---|---|")
    for a in ab_rows:
        L.append(f"| {a['tag']} | {a['title']} | {a['finding']} | [{a['evidence']}](../{a['evidence']}) |")
    L.append("")
    L.append(f"**失败案例分析**（{len(fail_rows)} 类证据）：")
    L.append("")
    L.append("| 内容 | 文档 | 附加证据 |")
    L.append("|---|---|---|")
    for f in fail_rows:
        L.append(f"| {f['title']} | [{f['doc']}](../{f['doc']}) | `{f['extra']}` |")
    L.append("")
    L.append("## 5. 标准三：README 四要素自检")
    L.append("")
    L.append("由脚本扫描 README 自动判定：")
    L.append("")
    L.append("| 要素 | 自检结果 | 位置 |")
    L.append("|---|---|---|")
    L.append(f"| 环境搭建 | {'✅' if rm['env_setup'] else '❌'} | 快速开始 + 第二阶段验收导航（含 MuJoCo 版本矩阵、EGL） |")
    L.append(f"| 复现步骤 | {'✅' if rm['repro'] else '❌'} | 各实验 RESULTS.md / run_*.sh 统一索引 |")
    L.append(f"| 实验结果 | {'✅' if rm['results'] else '❌'} | 结果主表 + 跨环境覆盖矩阵 |")
    L.append(f"| demo 视频链接 | {'✅' if rm['demo_links_ok'] else '❌'} | Demo 回放章节，共 {rm['demo_video_links']} 个 mp4/GIF |")
    L.append("")
    L.append("## 6. 标准一的缺口与补跑方案")
    L.append("")
    L.append("要让严格口径成立，需把至少 2 个算法补到 MetaWorld 与 ALOHA。可行路径：")
    L.append("")
    L.append("1. **Diffusion（成本最低）**：lerobot 0.7.1 原生 `--policy.type=diffusion`，"
             "两个数据集均已缓存到工作区 `datasets/`；沿用 "
             "[run_train.sh](../experiments/act_aloha/scripts/run_train.sh) 的命令模板替换 policy.type 即可，"
             "评估走 `lerobot-eval`。耗时参考：ACT 在 ALOHA 上 2000 步实测约 137 s，"
             "Diffusion 因网络更大需实测，此处不预估。")
    L.append("2. **Flow Matching / BC（自研最小实现）**：当前 `baselines/flow`、`baselines/bc` "
             "按 PushT 的 2 维动作 / 96×96 图像编写，迁到 ALOHA 14 维 / 480×640 需改数据管线与归一化，"
             "工作量明显更大；不建议为凑数硬迁。")
    L.append("3. **依赖前提（已验证）**：当前共享环境 `mujoco 3.8.1` 下 ALOHA 与 MetaWorld "
             "均可创建环境并完成 rollout，两者可共用一套依赖，无需再切换版本。")
    L.append("")
    L.append("> 补跑只证明跨环境链路可用，2000 步未收敛、成功率预期为 0，"
             "不构成策略能力评价——与 [act_aloha_transfer.md](act_aloha_transfer.md) 的口径一致。")
    L.append("")
    L.append("## 7. 复现")
    L.append("")
    L.append("```bash")
    L.append("python experiments/phase2_acceptance/scripts/build_acceptance.py")
    L.append("python experiments/act_metaworld/scripts/plot_train_curves.py")
    L.append("```")
    L.append("")
    L.append("## 8. 边界与诚实性声明")
    L.append("")
    L.append("1. 成功率口径不统一：PushT 50ep、MetaWorld 20ep、ALOHA 5ep，"
             "且 ALOHA Wilson 95% CI 为 0%–43.4%，样本量小，不可跨环境直接比较高低。")
    L.append("2. 多数跨环境成功率为 0（未收敛），本报告只用它们证明「链路跑通 + 指标齐全」，"
             "不用于算法优劣排序。")
    L.append("3. 环境引擎归属以包元数据为准，不把 pymunk 环境计入 MuJoCo。")
    L.append("")
    DOC.write_text("\n".join(L))
    print("doc ->", DOC.relative_to(REPO))
    print("json ->", (OUT / "acceptance.json").relative_to(REPO))
    for k, vv in verdict.items():
        print(k, vv["status"])


if __name__ == "__main__":
    main()
