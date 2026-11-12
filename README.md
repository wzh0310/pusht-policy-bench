# PushT Policy Bench

![Python](https://img.shields.io/badge/python-3.12-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.7.1-orange) ![LeRobot](https://img.shields.io/badge/lerobot-0.7.1-8A2BE2) ![License](https://img.shields.io/badge/license-MIT-green) ![PushT Success Rate](https://img.shields.io/badge/PushT-90%25-success)

模仿学习三主流方案——**BC / ACT / Diffusion Policy**——的可复现对比基准。当前覆盖
PushT 平面推块任务与 MetaWorld 单臂推球，对 LeRobot 生态中的策略给出统一环境下的
成功率、奖励、推理耗时与显存占用，并附完整消融与失效模式分析。

**算法一句话**：BC 把模仿当监督学习直接回归动作（下限基线）；ACT 加动作分块 +
CVAE（质量/延迟均衡解）；Diffusion Policy 用迭代去噪生成动作块（质量上限，推理成本
随采样步数线性增长，需 DDIM 裁剪）。

![PushT 成功案例](assets/demo_success.gif)

*`lerobot/diffusion_pusht` 在 PushT-v0 上的评测回放：将 T 形蓝块推入目标区域，覆盖率达标即判定成功。*

## 任务介绍：PushT 是什么

PushT 是视觉运动模仿学习最常用的基准任务之一：用圆形夹爪把随机放置的 **T 形蓝色块**推进虚线标出的**绿色目标轮廓**，重叠覆盖率（coverage）达标即成功。状态空间为夹爪二维位置（`observation.state`，2 维），动作空间为夹爪下一步的二维目标位置（`action`，2 维），第三人称俯视相机提供 96×96 RGB 图像，控制频率 10 Hz。官方演示数据 206 条轨迹、25650 帧（本仓库使用本地 v3.0 格式副本）。单episode最长 5 秒（50 步），对误差累积敏感，适合暴露行为克隆的分布偏移问题。

## 结果

| 策略 | 任务 | Episodes | 成功率 | avg sum_reward | avg max_reward | 平均耗时 |
|---|---|---|---|---|---|---|
| `lerobot/diffusion_pusht`（官方权重） | PushT-v0 | 10 | **90.0%** | 71.91 | 0.963 | 5.08 s/ep |
| `bc_v2`（自训，ResNet18+增广，19k 步） | PushT-v0 | **50** | **10.0%** | 95.24 | 0.542 | ≈0.3 s/ep |
| `act_v2`（自训 30k 步） | PushT-v0 | **50** | **8.0%** | 61.09 | 0.594 | 1.49 s/ep |
| `act_v3`（自训 50k 步） | PushT-v0 | **50** | **14.0%** | 61.67 | 0.613 | 1.49 s/ep |
| `act_chunk10`（消融：chunk=10） | PushT-v0 | **50** | **18.0%** | — | 0.631 | 1.49 s/ep |
| `act_chunk1`（消融：单步） | PushT-v0 | **50** | 0.0% | — | 0.083 | 1.49 s/ep |
| `act_novae`（消融：关 CVAE） | PushT-v0 | **50** | 6.0% | — | 0.680 | 1.49 s/ep |
| `bc_baseline` v1（自训最小 BC） | PushT-v0 | 20 | 0.0% | 34.54 | 0.132 | ≈0.3 s/ep |
| `diffusion_baseline`（自训 7.7k 步，公平对照） | PushT-v0 | **50** | **6.0%** | 79.55 | 0.670 | ≈1.6 s/ep |
| `flow_baseline`（自训流匹配，7.7k 步公平对照） | PushT-v0 | **50** | 2.0% | 65.30 | 0.534 | ≈1.0 s/ep |
| `act_pusht_baseline` v1（自训 7680 步） | PushT-v0 | 20 | 0.0% | 43.62 | 0.471 | 1.27 s/ep |

自训基线详情（协议、曲线、失败归因）：[experiments/bc_baseline/](experiments/bc_baseline/)（v1 + v2）、[experiments/act_baseline/](experiments/act_baseline/)（v1）、[experiments/act_v2/](experiments/act_v2/)（v2）。
两个 v2 模型均获得非零成功率；扩到 50 episode 复测后 BC 10%（5/50）、ACT v2 8%（4/50）；ACT 继续加训至 50k 步（v3）后达 14%（7/50）反超 BC。三次实验一致表明：eval_loss 进入平台后成功率仍在随训练量上升。消融实验（同协议 50k 步）：chunk=1 成功率归零（动作分块的因果证据），chunk=10 最优 18%，关闭 CVAE 腰斩至 6%。四种失效模式对比：单步决策死于抖动、无 latent 死于分叉、完整 ACT 死于最后 5%、BC 死于累积。关键路径：BC v1 → v2 靠 ResNet18 + 增广（覆盖率 0.13→0.60）；ACT v1 → v2 靠步数进收敛区间（0.47→0.62）。
严格同协议对照（7680 步）：Diffusion 6%（3/50）> ACT 0% ≈ BC 0%，同预算下迭代去噪的动作块质量最高，但最先过拟合（epoch 6 后 eval_loss 回升）；代价是推理成本约为 ACT 的 170 倍（微基准），详见 [diffusion_baseline](experiments/diffusion_baseline/RESULTS.md)。

成功率 95% 置信区间 59.6%–98.2%（10 个 episode 样本量小，区间较宽，扩到 50 个 episode 会明显收窄）。

测试环境：NVIDIA RTX A6000 / driver 550.107.02 / CUDA 12.6 / PyTorch 2.7.1 / LeRobot 0.7.1 / Python 3.12。

失败样例见 [assets/demo_failure.mp4](assets/demo_failure.mp4)：该 episode 末帧覆盖率停在 0.63，夹爪在目标区边缘来回修正后超时。

## 从零实现的 BC 基线

[baselines/bc/](baselines/bc/) 是一个约 200 行、端到端可运行的**单步行为克隆**基线（小 CNN 视觉编码器 + MLP 状态编码器 + 融合 trunk + 线性动作头），不依赖 lerobot 训练框架，便于逐行精读。ACT / Diffusion Policy 都共享这套"编码 + 融合"骨架，区别只在动作头。

**实验设置**：本地 pusht 数据集（198 条训练 / 8 条验证 episode）；图像 96×96；归一化（mean/std）后在归一化动作空间计算损失；batch_size 8（与 ACT demo 对齐）。

**结果对照**（batch_size=8，A6000，详细日志见 [experiments/](experiments/)）：

| | 参数量 | 峰值显存 | 损失函数 | 训练吞吐 |
|---|---|---|---|---|
| BC（本仓库） | 0.24M | **32 MiB** | MSE | 7680 步 / 188 s（≈41 步/s，batch=64） |
| ACT（lerobot 官方） | 51.6M | **≈950 MiB** | L1 + 10×KL（CVAE） | 1000 步 / 46 s（≈22 步/s，batch=8） |

![BC 与 ACT 损失曲线](assets/loss_curves.png)

*左：BC 前 7 个 epoch（机器负载高导致复跑中断，7 epoch 已呈收敛趋势；train 0.097→0.017，val 0.043→0.034，归一化动作空间 MSE）。右：ACT 预热 1000 步，loss 由 L1 回归项与 KL 散度项（`kl_weight=10`）相加而成，KLD 初期冲高后被压至 0.15 以下；BC 只有纯粹的回归损失。ACT 显存约为 BC 的 30 倍，来源是 ResNet18 视觉编码器、Transformer 结构以及默认 `chunk_size=100` 的动作序列标签。*

## 失败分析与改进方向

BC 在此任务上的已知缺陷与对应改进路线（完整推导见 [docs/theory.md](docs/theory.md)）：

1. **误差累积**：单步模仿误差 $\epsilon$ 在 T 步闭环中放大为 $O(T^2\epsilon)$（Ross et al. 2011）。失败样例中夹爪在目标区边缘来回修正即此表现 → 动作分块（ACT）、在线纠偏（DAgger）。
2. **多模态动作模糊**：MSE 回归在多种合理动作间取平均，输出"糊糊的动作" → CVAE latent（ACT）、扩散生成（Diffusion Policy）。
3. **训练/部署分布偏移**：部署时策略看到自己造成的画面，落在训练分布外 → DAgger 聚合标注、数据覆盖度增强。

4. **离线 loss 与在线成功率脱钩**（ACT 基线实测）：自训 ACT 在留出集 L1 上从 0.423 降到 0.375（epoch 12 后平台期），但 20 episode 在线评估成功率为 0——回归误差的微小改善不改变闭环行为的分叉点。评估策略必须看环境回滚，不能只看验证集 loss。

5. **归一化量纲错误在监督指标上是隐形的**（BC 基线实测）：v3.0 数据集图像是 `[0,1]` float 且 stats 同量纲，按旧版 0-255 约定二次除 255 后，训练/验证 loss 完全正常（train/test 同步缩放），但部署时正确预处理的输入与训练分布差 255 倍，rollout 覆盖率掉到 0.008。修复后重训，覆盖率恢复到 0.13，但 0.24M 小 CNN + 单步动作的 BC 仍 0/20——工程 bug 修掉后，算法瓶颈（容量、误差累积）才暴露出来。详见 [experiments/bc_baseline/RESULTS.md](experiments/bc_baseline/RESULTS.md)。

## Demo 回放（按算法分类）

**BC**
- [bc_v2_best.gif](assets/bc_v2_best.gif)：自训 BC v2（ResNet18+增广）成功推入
- [bc_v1_failure.gif](assets/bc_v1_failure.gif)：典型失效——分布外跑偏，夹爪停角落

**ACT**
- [act_v3_success.gif](assets/act_v3_success.gif)：自训 ACT 50k 步成功推入
- [act_v2_success.gif](assets/act_v2_success.gif)：ACT 30k 步成功
- [ablation_chunk1_best_effort.gif](assets/ablation_chunk1_best_effort.gif)：chunk=1 失效——停在 0.932 未过线
- [ablation_novae_best_effort.gif](assets/ablation_novae_best_effort.gif)：无 CVAE 失效——分叉选错，块反复进出

**Diffusion Policy**
- [diffusion_baseline_success.gif](assets/diffusion_baseline_success.gif)：自训基线成功（同协议 7.7k 步）
- [diffusion_nearmiss.gif](assets/diffusion_nearmiss.gif)：典型失效——覆盖率 0.9995 到位后保不住
- [demo_success.gif](assets/demo_success.gif)：官方权重 90% 成功率参考
- [demo_failure.mp4](assets/demo_failure.mp4)：官方权重失败样例

**跨环境迁移（MetaWorld push-v3）**
- [metaworld_v2_best.gif](assets/metaworld_v2_best.gif)：ACT 迁移——"悬停不接触"过拟合失效

## 技术报告

完整结论汇总（基线对比、消融量化、失效模式分析）见 [docs/REPORT.md](docs/REPORT.md)；
三算法全维度对比与失效模式深度分析见 [docs/comparison.md](docs/comparison.md)；
选型建议速查见 REPORT 第 8 节；常见问答见 [docs/faq.md](docs/faq.md)。

## 这个仓库解决什么

机器人策略工作里最耗时的不是训练，而是**让环境跑起来、让数字可信**。本项目聚焦三件事：

1. **环境可复现** —— 两种搭建路径的取舍边界，附可自检的环境核对清单（[docs/environment.md](docs/environment.md)）
2. **指标可复现** —— 固定 episode 数、固定策略版本、逐 episode 导出录像，任何数字都能复算（[docs/results.md](docs/results.md)）
3. **失败可归因** —— 收集 11 条真实故障，每条给出"现象 / 根因 / 解法"，而不只是"改了什么让它能跑"（[docs/findings.md](docs/findings.md)）

## 快速开始

```bash
# 1. 环境（源码模式，便于改配置与调试）
conda create -n lerobot python=3.12 -y && conda activate lerobot
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[training]"

# 2. 对齐 GPU 版本（wheel 自带的 CUDA runtime 需与驱动匹配）
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126

# 3. 核对
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 4. 评测（自动下载策略与数据集，导出视频）
bash scripts/eval_pusht.sh
```

无显示器的服务器需要离屏渲染，脚本已内置：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 目录结构

```
.
├── README.md
├── configs/
│   └── act_pusht.yaml      # ACT 训练配置，每个超参标注调参依据
├── baselines/
│   ├── bc/                   # 最小单步 BC 基线（代码 + 配置 + 说明）
│   └── flow/                 # 最小 Flow Matching 动作头实现（对照 Diffusion）
├── experiments/
│   ├── bc_baseline/          # BC 20 epoch 训练日志与复现脚本
│   └── act_warmup/           # ACT 1000 步预热训练日志与复现脚本
├── docs/
│   ├── environment.md        # 环境方案对比与环境自检
│   ├── results.md            # 评测方法、逐 episode 明细、耗时分解
│   ├── findings.md           # 概念辨析与故障归因库
│   ├── theory.md             # BC 缺陷的数学解释、DAgger、ACT 解读、BC vs 离线 RL
│   ├── REPORT.md             # 项目技术报告：原理、基线对比、消融与失效模式分析
│   ├── comparison.md         # 三算法全维度对比 + 失效模式深度分析（含回放）
│   ├── diffusion_notes.md    # Diffusion Policy 源码拆解（架构图/形状实测/超参调参方向）
│   ├── flow_notes.md         # Flow Policy（流匹配）与 Diffusion 的结构差异 + 实测
│   └── faq.md                # 高频问答（全部答案带实测数据）
├── scripts/
│   ├── eval_pusht.sh       # 评测流水线（含 GPU 预检）
│   └── train_pusht.sh      # 训练流水线（支持冒烟参数）
└── assets/                 # 回放录像与 GIF
```

## 关键发现

**1. ACT / Diffusion Policy 是模仿学习，不是强化学习**

这两类策略的损失函数是动作回归（L1 + KL 正则），`next.reward` 不参与梯度计算。它们缺少强化学习的三个要件：奖励参与优化、试错探索、自举估计。归类上属于"监督学习中的行为克隆"，与 RL 的交集仅在于解决的序贯决策问题相同。

**2. 训练与推理是两套流程**

训练时输入来自固定数据集、有标准答案、可完全并行；推理时输入来自环境实时渲染、无答案、必须串行闭环。策略一旦偏离演示分布，看到的画面就落在训练分布之外，误差会累积放大（covariate shift）。这是行为克隆区别于普通监督学习的核心难点。

**3. 权重迁移后要检查 `device` 字段**

旧版 checkpoint 转换为新版 processor 格式时，若转换发生在无 GPU 环境，会把 `"device": "cpu"` 写入 `config.json`。策略会尊重该字段，不会自动改，表现为装了 GPU 版 torch 仍是 CPU 速度——实测相差两个数量级。

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 最小 BC 基线从零实现，跑通训练 / 验证 / 推理闭环 | 完成 |
| P1 | 环境搭建、官方权重评测、训练链路验证 | 完成 |
| P2 | ACT 从零训练至收敛，记录 loss / 成功率曲线 | 完成（30k 步 5% 成功率） |
| P3 | ACT / Diffusion / VQ-BeT 横向对比（成功率 + 推理延迟） | 计划 |
| P4 | 超参消融：`chunk_size` / `n_action_steps` / `kl_weight` / 学习率 | 部分完成（chunk_size、CVAE 见 [experiments/act_ablation/](experiments/act_ablation/)） |

## 跨环境迁移实验（MetaWorld push-v3）

SO-101 仿真已退出 lerobot 生态（无环境注册/无 extra/官方数据集下架），改用同为
MuJoCo 底层的 MetaWorld push-v3 验证 ACT 跨环境能力：同一套配方（chunk 30、lr 2e-5、CVAE）
零改动在新环境训出有目的行为的策略（v1 3000 步峰值奖励 0.51，冒烟出现成功 episode；
v2 20000 步过拟合出"悬停不接触"）。环境修复手册（mujoco 3.1.6 版本锚定、免 root EGL
libglvnd 方案、孪生环境补 39 维 keypoints、CUDA/EGL 初始化顺序）见
[experiments/act_metaworld/RESULTS.md](experiments/act_metaworld/RESULTS.md)。

## 参考

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot)
- [gym-pusht](https://github.com/huggingface/gym-pusht)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, RSS 2023)
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (RSS 2023)

## License

MIT，见 [LICENSE](LICENSE)。
