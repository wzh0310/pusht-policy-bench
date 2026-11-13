# 扩散 vs 流匹配：失败模式对比（2026-11-13）

**完整报告**：[docs/failure_mode_analysis.md](../../docs/failure_mode_analysis.md)
**量化主表**：[failure_table.md](results/failure_table.md) ｜ [假设检验汇总](results/failure_summary.json)
**抽帧图**：[扩散 3 条](../../assets/failure_diffusion_contact_sheet.jpg) ｜ [流匹配 6 条（含同种子对照）](../../assets/failure_flow_contact_sheet.jpg)

## 一句话结论

首要失效不是「扩散多步累积误差」，也不是「流匹配单步精度下降」，而是**峰值达标后无法保持**：
两条 Flow 轨迹覆盖率到过 1.000，随后继续开环推动又把 T 块推走（其中一条终态掉到 0.183）。
步数是与初态交互的旋钮，不是误差源；42/50 个初态在扩散四档全部失败，说明主因是初态覆盖不足。

## 假设检验

| 编号 | 说法 | 判定 |
|---|---|---|
| H1 | 扩散多步采样累积误差 | **不支持**（10→50 步 p=0.461，四档覆盖率非单调） |
| H2 | 极端分布外场景导致动作突变 | **部分支持**（初态主导成立，突变证据不足） |
| H3 | 流匹配单步细节精度普遍下降 | **不支持**（步数 × 初态交互，方向不一致） |
| H4 | 流匹配高速动态适配性不足 | **部分支持**（精细调整阶段高速动作有害） |
| H5 | 峰值达标后无法保持是主要失效 | **强支持** |
| H6 | 困难初态无法靠换步数稳定救援 | **支持** |

逐条证据与自动生成的数值见 [failure_summary.json](results/failure_summary.json)。

## 3 + 3 条失败轨迹

| 算法 | 代表轨迹 | 峰值覆盖 | 关键现象 |
|---|---|---:|---|
| Diffusion | 50 步 ep9 | 0.475 | 推离目标后停滞，块边界 jerk 比 3.20 |
| Diffusion | 10 步 ep5 | 0.000 | 四档皆 0，灰块被推到画面边缘 |
| Diffusion | 10 步 ep2 | 0.436 | 绕目标打转，停滞比 0.31 |
| Flow | 1 步 ep6 | **1.000** | 第 203 步完全对齐，终态 0.183 |
| Flow | 1 步 ep5 | 0.380 | 同初态换 10 步可达 1.000 |
| Flow | 1 步 ep28 | 0.000 | 两档恒 0，困难初态 |

每条扩散失败轨迹配 1 条 mp4；每条 Flow 失败轨迹配 1 步 / 10 步两条 mp4（后者为同种子对照）。
9 个视频均在仓库 `assets/`，完整 300 帧逐帧序列在 `results/*_analysis.json` 的 `per_frame`。

## 复现

```bash
cd pusht-policy-bench/experiments/failure_modes
conda activate py312-robot
# 若 shell 未初始化 conda，先执行 eval "$(conda shell.bash hook)"
# 外部产物默认位于仓库上一级目录；换位置则 export WZH_ROOT=<workspace-root>

python3 scripts/analyze_diffusion_failures.py   # 四档配对 + 3 条扩散失败轨迹
python  scripts/record_flow_failures.py         # 3 初态 x 2 档录制
python3 scripts/summarize_failures.py           # 对比表 + 假设检验
```

依赖 `cv2` / `Pillow` / `gym_pusht` / `torch`。外部数据：
`outputs/diffusion_eval_steps{5,10,20,50}/eval_info.json` 与其前 10 条回放视频、
`bc/outputs/flow/checkpoint.pt`；仓库内还需 `experiments/flow_vs_diffusion/results/` 的消融结果。
脚本用 `Path(__file__)` 相对定位，不改动既有训练/评估代码，结果里的路径均为相对工作区路径。

## 目录内容

```
scripts/frame_metrics.py                 逐帧指标 + 抽帧拼图（共用模块）
scripts/analyze_diffusion_failures.py    四档配对符号检验 + 失败轨迹选取
scripts/record_flow_failures.py          Flow 失败轨迹与同种子对照录制
scripts/summarize_failures.py            对比表 + 假设检验结论
results/diffusion_failure_analysis.json 配对统计、3 条轨迹指标、50ep 逐集数据
results/flow_failure_analysis.json      6 条轨迹的动作/逐帧指标
results/flow_failure_actions.json       6 x 300 步动作与奖励
results/failure_table.md / failure_summary.json
```

## 验收

| 验收项 | 状态 |
|---|---|
| 扩散与流匹配各 3 条失败轨迹 + 逐帧分析 | ✅ 9 条 mp4 + 抽帧图 + 逐帧序列 |
| 失败模式对比文字分析 | ✅ 完整报告第 1–4 节 |
| 什么场景优先选扩散 / 流匹配 | ✅ 完整报告第 5 节，8 个场景 + 按失效边界选型 |

## 局限

扩散视频只存档前 10 条 episode；jerk/停滞为像素代理指标；OOD 未用训练集分布量化；
保持失效在扩散侧缺逐步奖励验证；重录与消融评估可能有浮点级差异。详见完整报告第 8 节。
