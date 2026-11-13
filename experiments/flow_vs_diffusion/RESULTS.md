# 扩散 vs 流匹配：核心维度定向对比（2026-11-13）

**完整报告**：[docs/flow_vs_diffusion.md](../../docs/flow_vs_diffusion.md)
**展示图**：[步数消融双面板](../../assets/flow_vs_diffusion_ablation.png) ｜ [轨迹抽帧对比](../../assets/flow_vs_diffusion_smoothness.jpg) ｜ [Flow 回放 GIF](../../assets/flow_1step_best.gif)

## 一句话结论

固定数据集、训练预算与评估协议，只对比两个核心维度：**流匹配单步成本是扩散的 1/16，1 步 Euler 就拿到全表最高峰值覆盖率 0.727 与 2.67 ms 重规划延迟；扩散靠步数线性换精度，10–50 步需 95–463 ms**。实时控制选 Flow，极限质量与多模态采样选 Diffusion。

## 量化主表（50 episodes × 300 步）

| 算法 | 步数 | 成功率 | Wilson 95% CI | 峰值覆盖率 | sumR | 重规划 ms | 单步 ms |
|---|---:|---:|---|---:|---:|---:|---:|
| **Flow** | **1** | **4.0%** | 1.1–13.5% | **0.727** | **92.6** | **2.67** | 2.665 |
| Flow | 3 | 0.0% | 0.0–7.1% | 0.471 | 63.3 | 3.78 | 1.261 |
| Flow | 5 | 0.0% | 0.0–7.1% | 0.531 | 72.3 | 5.10 | 1.020 |
| Flow | 10 | 0.0% | 0.0–7.1% | 0.549 | 71.9 | 7.76 | 0.776 |
| Diffusion | 10 | 2.0% | 0.4–10.5% | 0.682 | 73.9 | 94.54 | 9.454 |
| Diffusion | 20 | 4.0% | 1.1–13.5% | 0.622 | 74.0 | 186.11 | 9.305 |
| Diffusion | 50 | 6.0% | 2.1–16.2% | 0.661 | 79.9 | 462.63 | 9.253 |

机器可读：[ablation_table.csv](results/ablation_table.csv) ｜ [ablation_table.md](results/ablation_table.md) ｜ [summary.json](results/summary.json)（含逐档 Wilson CI 与延迟来源标注）。

## 三个关键发现

1. **Flow 步数不是越多越准**：1 步最优，10 步覆盖率掉到 0.549、延迟涨 2.9 倍。直线插值下速度场理论为常数，单步 Euler 即精确解；多步积分在中间时刻反复查询 0.3M 小网络，误差沿积分链累积。
2. **延迟严格线性**：`Flow ≈ 2.13 + 0.567N` ms、`Diffusion ≈ 3.01 + 9.169N` ms，单步成本差 16.2×；Flow 1 步比 Diffusion 10 步快 35.4×、比 50 步快 173×。
3. **训练稳定性**：Flow CV 0.147 / 劣化 1.42× / 最优 index 10；Diffusion CV 0.310 / 劣化 2.15× / 最优 index 5。Flow 更稳但既有基线质量更低——**稳定 ≠ 更强**。

**统计诚实性**：50ep 下 0%–6% 的 Wilson CI 全部重叠，成功率**不能用于排序**；跨算法判断依据覆盖率、sumR、延迟这三个方差更小的连续量。

## 训练稳定性与动作平滑度

- 稳定性 20 点同频验证序列与全部指标：[stability.json](results/stability.json)
- 平滑度指标（等长 118 帧前缀、jerk 代理、块边界效应）：[smoothness.json](results/smoothness.json)
- 两者均无高频颤动；Diffusion 是「整体更粗」（jerk 0.666、尖峰率 0%），Flow 是「整体更平滑但有零星尖峰」（jerk 0.422、尖峰率 0.87%）。块边界效应均 >1，Flow 因重规划频率高 4 倍反而更明显。
- Flow 真实动作序列（300 步）：[flow_best_actions.json](results/flow_best_actions.json)

## 复现

```bash
cd pusht-policy-bench/experiments/flow_vs_diffusion
conda activate py312-robot
# 外部产物默认位于仓库上一级目录；换位置则 export WZH_ROOT=<workspace-root>

bash scripts/run_ablation.sh                     # Flow 1/3/5/10（约 3.5 min）
python3 scripts/extract_diffusion.py             # 既有 Diffusion 步数结果
python  scripts/bench_diffusion_latency.py --repeats 20 --steps 5 10 20 50 100
python3 scripts/analyze_stability.py
python  scripts/record_flow_rollout.py           # 视频与动作序列 → assets/ + results/
python  scripts/analyze_smoothness.py            # 平滑度 + 抽帧图 → assets/
python3 scripts/build_report.py                  # CSV + summary.json
python3 scripts/export_table_md.py               # Markdown 主表
python3 scripts/make_figures.py                  # 展示图 + GIF
```

环境为 `py312-robot`（Python 3.12 / PyTorch 2.7）；若 shell 未初始化 conda，先执行 `eval "$(conda shell.bash hook)"`。

脚本用 `Path(__file__)` 相对定位仓库，外部 checkpoint / lerobot 源码 / 既有评估结果由 `WZH_ROOT` 指向；不改动仓库既有训练与评估代码。需 `cv2`、`Pillow`、`gym_pusht`、`torch`（`py312-robot` 环境），A6000 实测。

**外部依赖**：`bc/outputs/flow/checkpoint.pt`、`outputs/diffusion_baseline/checkpoints/003840/pretrained_model`、`outputs/diffusion_eval_steps{5,10,20,50}/eval_info.json`、`lerobot/src`。仓库内已有 `experiments/{flow,diffusion}_baseline/train_log.txt` 与本目录 `results/` 全量数据，因此**不重训也能复算表格与图表**。

## 目录内容

```
scripts/   10 个脚本：消融评估、结果提取、延迟微基准、稳定性、录屏、平滑度、汇总、出图
results/   flow_steps{1,3,5,10}.json + *_eval.txt；diffusion_steps.json；diffusion_latency.json；
           stability.json；smoothness.json；flow_best_actions.json；
           ablation_table.csv / .md；summary.json
assets/    （仓库根）flow_vs_diffusion_ablation.png / _smoothness.jpg / flow_1step_best.mp4 / .gif
```

## 验收

| 验收项 | 状态 |
|---|---|
| 推理步数消融（Flow 1/3/5/10；Diffusion 10/20/50） | ✅ |
| 量化对比表格（成功率 + CI + 覆盖率 + sumR + 延迟 + 单步成本） | ✅ |
| 训练稳定性量化 + 动作平滑度定性/定量 | ✅ |
| 总结两种算法各自适用场景 | ✅ 见完整报告第 4 节 |

## 局限

成功率样本量不足、两者 checkpoint 早停口径不同、延迟测量分属闭环与微基准、平滑度为像素代理指标、未覆盖 DDIM / 高阶积分器。详见完整报告第 8 节。
