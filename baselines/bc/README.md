# BC（行为克隆）最小实现

本目录是一个**可运行的单步 BC 基线**，用于精读对照：ACT 和 Diffusion Policy 都共享这套
"视觉编码 + 状态编码 + 融合"骨架，区别只在动作头。

## 网络结构

```
相机图像 (3,96,96)                      机器人状态 (2,)
      │                                      │
  ConvEncoder（4×Conv+GroupNorm+ReLU    MLP(2→128→128)
   + Flatten + Linear，输出 128 维）           │
      │  img_feat (128)                      │  state_feat (128)
      └────────────── concat (256) ──────────┘
                          │
              trunk MLP(256→128→128)
                          │
              Linear(128→2)  ← 动作头：纯线性回归，输出 a_t
```

## 训练目标与损失

监督学习：输入当前帧图像 + 状态，回归演示数据中同一时刻的动作。

```
L = MSE(归一化后的预测动作, 归一化后的演示动作)
```

在归一化（mean/std）后的动作空间上计算 MSE，量纲统一、数值稳定。
权重保存在 `outputs/checkpoint.pt`（含模型权重、归一化统计量、模型配置）。

## 对照 lerobot 中的策略

| 组件 | 本实现（BC） | lerobot `diffusion` | lerobot `act` |
|---|---|---|---|
| 视觉编码器 | 小 CNN（本目录 `model.py`） | ResNet18（`modeling_diffusion.py`） | ResNet18（`modeling_act.py`） |
| 动作头 | 线性回归单步动作 | 扩散去噪（损失仍是 MSE，预测噪声） | CVAE + 动作分块 + 时间集成 |
| 推理代价 | 1 次前向 | 多步去噪 | 1 次前向 + 集成平滑 |

## 使用方法

```bash
# 环境：py312-robot（lerobot 0.7.1 + torch 2.7.1 + CUDA）
conda activate py312-robot

# 训练（默认 20 epoch，A6000 约 10 秒/epoch）
python train.py --config config.yaml

# 冒烟测试（每 epoch 只跑 30 步）
python train.py --max-steps 30
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | BC 策略网络 + MSE 损失（重点精读文件） |
| `dataset.py` | 用 lerobot 的 `LeRobotDataset` 读取本地 pusht 数据集并做归一化（注意 v3.0 图像为 [0,1] float 且 stats 同量纲，勿二次除 255） |
| `train.py` | 训练循环：AdamW + MSE，按验证集保存最优权重 |
| `eval_env.py` | 在线评估：gym_pusht 闭环 rollout，预处理与训练严格一致（离线 loss 正常 ≠ 在线能跑，案例见 experiments/bc_baseline/RESULTS.md） |
| `config.yaml` | 超参数：lr、batch_size、hidden_dim 等 |
| `outputs/checkpoint.pt` | 训练好的权重与归一化统计量 |

## 推断动作（加载 checkpoint）

```python
import torch
ckpt = torch.load("outputs/checkpoint.pt", weights_only=False)
model = BCPolicy(**ckpt["model_kwargs"]).cuda()
model.load_state_dict(ckpt["model_state_dict"])
# image: (1,3,96,96) float [0,1]；state: (1,2) float 原始量纲
image = (image / 255 - ckpt["normalization"]["img_mean"]) / ckpt["normalization"]["img_std"]
state = (state - ckpt["normalization"]["state_mean"]) / ckpt["normalization"]["state_std"]
action_norm = model(image.cuda(), state.cuda())
action = action_norm * ckpt["normalization"]["action_std"] + ckpt["normalization"]["action_mean"]
```
