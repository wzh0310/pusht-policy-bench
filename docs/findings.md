# 概念辨析与故障归因

本文记录在 PushT 上跑通 LeRobot 后得到的认知结论，以及实际遇到的 11 条故障。每条故障按"现象 / 根因 / 解法"组织，便于检索。

---

## 一、概念：模仿学习不是强化学习

常见的误解是"策略学习 = 强化学习"。对 ACT / Diffusion Policy 这类方法，这个说法不成立。

### 判定依据

强化学习有四个特征，二者逐条对照：

| 特征 | ACT / Diffusion Policy |
|---|---|
| 奖励参与参数优化 | 否。loss 只有动作 L1 回归（+ KL 正则） |
| 试错探索 | 否。不采样动作、不与环境交互 |
| 自举估计（Bellman 更新） | 否。没有 critic、没有 Q 值 |
| 时间维度信用分配 | 否。每帧独立计算误差 |

因此它们属于**监督学习中的行为克隆（behavior cloning）**，训练目标是最小化预测动作与人类动作的距离：

```
L = E_{(o,a)~D} [ || pi(o) - a ||_1 ] + kl_weight * KL
```

其中 `D` 是固定数据集，`a` 是人类演示动作。它与 RL 的交集仅在于解决的**问题**（序贯决策）相同，方法完全属于监督学习。

### 数据样本的输入与标签

一条训练样本对应一个时间步：

| 项 | 字段 | 形状 | 含义 |
|---|---|---|---|
| 输入 | `observation.image` | 3×96×96 | 俯视 RGB，归一化到 [-1,1] |
| 输入 | `observation.state` | 2 | 夹爪在 512×512 画布上的像素坐标 |
| 标签 | `action` | 100×2 | **未来 100 步**的目标坐标（动作块） |
| 评估用 | `next.reward` | 1 | 覆盖率，**不参与训练** |

两个容易忽略的细节：

- `action` 是**绝对目标坐标**，不是速度增量。环境每步只把夹爪朝目标拖约 6~7 像素，而目标点平均在前方约 15 像素处，所以人类始终在"提前瞄准"。
- 标签是 100 步的动作块而非单步动作。这样能吸收人类演示中的停顿与抖动，也避免单步预测的误差累积。

### 训练与推理的差异

| | 训练 | 推理 |
|---|---|---|
| 输入来源 | 固定数据集 | 环境实时渲染 |
| 标签 | 有（人类动作） | 无 |
| 执行方式 | 可完全并行 | 串行闭环 |
| 优化目标 | 最小化动作误差 | 最大化 episode 成功率 |

推理时策略看到的画面由自己的动作产生，一旦偏离演示分布，输入就落在训练分布之外，误差持续放大（covariate shift）。这是行为克隆的核心难点，也是离线 loss 与最终成功率不直接对应的原因。

---

## 二、故障库

### 1. CUDA 驱动与 torch 版本不匹配

**现象**：`The NVIDIA driver on your system is too old (found version 13000)`

**根因**：安装了 `cu132` 版 torch，需要 r580+ 驱动，而机器驱动为 550.107.02，上限是 CUDA 12.4。

**解法**：按驱动选择 wheel，`torch.version.cuda` 主版本不超过驱动上限。cu126 / cu128 在 550 驱动上均可运行。

### 2. GPU 完全不可见

**现象**：`nvidia-smi` 报无法与驱动通信，`/dev/nvidia*` 不存在。

**根因**：设备节点未暴露给当前进程（容器缺少 `--gpus`，或进程运行在屏蔽 `/dev` 的沙箱中）。

**解法**：容器加 `--gpus all`；宿主机检查驱动加载状态。验证标准是 `nvidia-smi` 有输出 **且** `torch.cuda.is_available()` 为 `True`，两者缺一不可。

### 3. torchcodec 破坏 CUDA 初始化

**现象**：导入 torch 后 CUDA 不可用，日志出现 `libnvrtc.so.13` 加载失败。

**根因**：`torchcodec` 链接了 CUDA 13 的 runtime，与 cu126 版 torch 冲突，导入即污染 CUDA 上下文。

**解法**：`pip uninstall -y torchcodec`，LeRobot 自动回退 PyAV 解码。

### 4. 迁移后的 checkpoint 静默使用 CPU

**现象**：GPU 可用，但训练速度只有约 15.9 s/step。

**根因**：旧版权重转换时在无 GPU 环境执行，把 `"device": "cpu"` 写入 `config.json`。策略会尊重配置字段，不会自动切换。

**解法**：检查并修改 `config.json` 中的 `device` 字段。修复后单 episode 评测从分钟级降到 5.08 s。

### 5. 脚本路径与参数随版本变更失效

**现象**：`can't open file '.../eval_policy.py'`；`--env-id`、`--video-path` 报未知参数。

**根因**：0.7.x 将脚本迁入 `src/lerobot/scripts/`，参数改为 draccus 点号嵌套风格，视频路径由 `output_dir` 派生。

**解法**：使用 `python -m lerobot.scripts.lerobot_eval`，参数写作 `--env.type=pusht`、`--eval.n_episodes=10`。

### 6. 相对导入要求模块方式启动

**现象**：`ImportError: attempted relative import with no known parent package`

**根因**：训练脚本内部使用 `from .lerobot_eval import ...`。

**解法**：统一用 `python -m lerobot.scripts.lerobot_train`。

### 7. 数据集体积异常小

**现象**：`lerobot/pusht` 只有 7.4 MB，疑似下载不全。

**根因**：v3.0 格式把图像编码为 AV1 视频流，不再逐帧存图。

**解法**：核对 `meta/info.json` 的 `total_episodes` / `total_frames`（206 / 25650）是否匹配。

### 8. `next.success` 全为 False

**现象**：206 条演示中该字段无一条为真，而末帧覆盖率已达 0.94。

**根因**：成功阈值为 0.95，这批演示普遍停在 0.89–0.949。

**解法**：不要用该字段做筛选或分类标签，改用末帧 reward 排序挑选高质量轨迹。

### 9. 输出目录已存在导致退出

**现象**：`FileExistsError: Output directory ... already exists and resume is False`

**根因**：设计上防止覆盖已有实验结果。

**解法**：更换目录名，或使用 `--resume=true` 并指向旧 checkpoint 的 `train_config.json`。

### 10. 官方模型 ID 不存在

**现象**：`401 Client Error ... Repository Not Found`。

**根因**：教程中的仓库名已失效。PushT 可用的预训练权重为 `lerobot/diffusion_pusht`、`lerobot/vqbet_pusht`、`lerobot/diffusion_pusht_keypoints`。

**解法**：先在 Hub 确认仓库存在，再写入命令。

### 11. 旧版 processor 格式不兼容

**现象**：加载旧 checkpoint 时报缺失 `policy_preprocessor.json` 等文件。

**根因**：新版把归一化统计从 checkpoint 内部拆分到独立的 processor 配置文件。

**解法**：使用官方迁移脚本转换 checkpoint，转换后检查 `device` 字段（见第 4 条）。
