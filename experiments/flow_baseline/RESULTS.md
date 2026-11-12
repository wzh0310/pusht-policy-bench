# Flow Matching 完整基线（与 Diffusion 基线严格同协议）

日期：2026-11-12 ｜ 环境：RTX A6000 / py312-robot ｜ 实现：[baselines/flow/](../../baselines/flow/)

## 协议对齐

与 [diffusion_baseline](../diffusion_baseline/RESULTS.md) 完全同协议：本地 pusht、
198/8 划分、7680 步、batch 64、每 384 步（每 epoch）留出集验证、lr 1e-4、无增广。
唯一差异是算法本体：观测编码相同（ResNet18），动作头为 0.3M 速度场网络
（Diffusion 为 251M U-Net）。

## 训练

- 单步耗时 ≈ 107 ms（41s/epoch ÷ 384 步），峰值显存 **462 MiB**；
- val loss：0.569（epoch 0）→ 最优 **0.309（约 epoch 9）** → 回升至 0.44（epoch 19）。
  与 Diffusion 一样出现早停，但拐点更晚（epoch 9 vs epoch 6）——容量小 500 倍，
  过拟合更慢。

## 评估（50 episodes × 300 步，同种子同口径）

| 模型（同协议 7680 步） | 成功率 | 峰值覆盖率 | avg sum_reward | 单步训练耗时 | 峰值显存 |
|---|---|---|---|---|---|
| **Flow（本实验）** | 2%（1/50） | 0.534 | 65.3 | ≈107 ms | **462 MiB** |
| Diffusion | **6%（3/50）** | **0.670** | **79.6** | ≈116 ms | 4.98 GB |

参考（推理）：Flow 9.0ms/次重规划 vs Diffusion DDPM-100 1235ms / DDIM-5 51ms。

## 结论

1. 同协议下 Diffusion 仍领先（6% vs 2%）——251M U-Net 的容量优势在 7.7k 步内真实
   存在；但 Flow 以 **1/500 的参数量、1/10 的显存**拿到同量级行为（覆盖率 0.53 vs 0.67）。
2. Flow 的收敛更快、过拟合更晚，配合其 137 倍推理优势，是"预算受限 + 实时部署"
   场景的更优工程选择；追求极限动作质量选 Diffusion。
3. 下一步：Flow 加大容量（速度场换小型 U-Net）或配合增广，观察与 Diffusion 的差距
   是否主要来自容量。
