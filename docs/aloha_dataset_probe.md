# ALOHA 双臂数据集探查（下周迁移验证准备）

> 成功加载 `lerobot/aloha_sim_insertion_human`，摸清双臂动作空间、观测设置与任务类型，
> 并给出从 PushT 迁移到双臂任务的改动清单。原始数据见
> [aloha_probe.json](../experiments/aloha_probe/results/aloha_probe.json)，
> 单实验记录与脚本见 [experiments/aloha_probe/](../experiments/aloha_probe/RESULTS.md)；
> 失效模式结论见 [failure_mode_analysis.md](failure_mode_analysis.md)。

**一句话结论**：这是一个 **50 集 / 25000 帧 / 50 fps** 的双臂插销仿真数据集，
动作与状态同为 **14 维关节角**（左右臂各 6 关节 + 1 夹爪）；
观测**只有一路 top 相机**（480×640），并非多视角。
迁移到它的真正难点不是把动作维度从 2 改成 14，而是 **50 Hz 控制预算** 与
**插销任务必需的「到位后保持」能力**——后者恰好是上一份失败分析里的主要失效模式。

**日期**：2026-11-13 ｜ **环境**：lerobot 0.7.1 / `py312-robot`

---

## 0. 数据规格速览

| 项目 | 实测值 |
|---|---|
| 数据集 | `lerobot/aloha_sim_insertion_human` |
| 轨迹数 | **50 episodes** |
| 总帧数 | **25000 frames**（每集固定 500 帧） |
| 采样率 | **50 fps** → 每集约 10 秒 |
| 任务 | `Insert the peg into the socket.`（双臂协同插销入孔） |
| 动作 | `action` shape **[14]**, float32，与 `observation.state` 同构 |
| 状态 | `observation.state` shape **[14]**, float32 |
| 图像 | `observation.images.top` shape **[480, 640, 3]**，样本返回 `[3, 480, 640]` float32 |
| 相机数 | **1**（仅 top） |
| 其它键 | `timestamp`、`frame_index`、`episode_index`、`index`、`task_index`、`task`、`next.done` |

---

## 1. 加载与 API 差异（用户示例代码需要适配）

任务书给的示例基于旧版 lerobot，`observation_keys` 在 **0.7.1 已移除**：

```python
# 旧示例（0.7.1 会在 observation_keys 处报错）
from lerobot.datasets import LeRobotDataset
dataset = LeRobotDataset('lerobot/aloha_sim_insertion_human')
print('观测键:', dataset.observation_keys)
print('动作形状:', dataset[0]['action'].shape)
print('轨迹数:', dataset.num_episodes)
```

实测可用的 0.7.1 写法：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset('lerobot/aloha_sim_insertion_human',
                    root='datasets/aloha_sim_insertion_human',
                    download_videos=True)
print('轨迹数:', ds.num_episodes)              # 50
print('帧数 / fps:', ds.num_frames, ds.fps)   # 25000 / 50
print('观测键:', sorted(k for k in ds.features if k.startswith('observation')))
print('动作形状:', ds[0]['action'].shape)     # torch.Size([14])
print('任务:', ds[0]['task'])                 # Insert the peg into the socket.
```

| 旧写法 | 0.7.1 状态 | 替代 |
|---|---|---|
| `dataset.observation_keys` | 已移除 | `dataset.features` 或 `dataset.meta.info['features']` |
| `dataset.num_episodes` | 仍可用 | — |
| `dataset[0]['action'].shape` | 仍可用 | 返回 `torch.Tensor`，不是 numpy |
| `LeRobotDataset(repo_id)` 不指定 root | 可用，但落到默认 HF 缓存 | 显式 `root=` 便于复现与清理 |
| 直接读图像 | 0.7.1 用视频后端（pyav / torchcodec）解码 | 需 `download_videos=True`，否则样本里没有像素 |

> 首次加载约 91 MB（其中 top 相机视频 88 MB），已缓存到工作区 `datasets/` 下，不入库。

---

## 2. 双臂动作空间：14 维从哪来

`action` 与 `observation.state` 的 `names.motors` 完全一致，按**左臂 7 + 右臂 7** 排列：

| 侧 | 关节 |
|---|---|
| 左 | `left_waist`, `left_shoulder`, `left_elbow`, `left_forearm_roll`, `left_wrist_angle`, `left_wrist_rotate`, `left_gripper` |
| 右 | `right_waist`, `right_shoulder`, `right_elbow`, `right_forearm_roll`, `right_wrist_angle`, `right_wrist_rotate`, `right_gripper` |

**含义**：这是**关节角位置控制**（6 个关节角 + 1 个夹爪开合），
不是末端执行器位姿（SE(3)），也不是 PushT 那样的 2 维目标点。
因此归一化统计量、损失维度、动作裁剪范围都要按 14 维**逐维**统计；
两臂还共享同一个时间轴，需要协同，不能拆成两个独立策略直接拼接。

---

## 3. 观测的多视角设置：实测是单视角

`camera_keys = ['observation.images.top']`，`n_cameras = 1`。
为避免「只是这个数据集恰好单视角」，我额外只读了 ALOHA 仿真系列另外三个数据集的
`meta/info.json`（不下载视频）：`aloha_sim_insertion_scripted`、
`aloha_sim_transfer_cube_human`、`aloha_sim_transfer_cube_scripted`——
**四个数据集的相机键都只有 `observation.images.top`，动作与状态均为 14 维，50 集 / 50 fps**。

对比真机 ALOHA 常见配置（top + 左右腕部共 3 路），lerobot 的这批仿真数据只导出俯视单目。
所以下周若要做**多视角融合验证**，需要：
- 另选带腕部相机的数据集或自行录制；
- 改策略的图像编码入口（当前 `flow_model.py` 只接单张 96×96 图 + 2 维 state）；
- 多相机按 `observation.images.*` 键动态枚举并拼接特征。

**遮挡风险**：单 top 相机在插销接触阶段会被夹爪遮挡，这正是任务最需要精细反馈的时刻，
也是迁移验证要重点观察的失效来源。

---

## 4. 任务类型与难度

`Insert the peg into the socket.` 属于**双臂协同 + 富接触 + 精修 + 保持**任务：
一侧持销、一侧持孔（或双臂共同对准），需要毫米级对准后稳定保持再插入。

与 PushT「单点推块到目标区域」相比，多了三件事：
1. **双侧约束**：两个 7 维子空间必须协同，误差来源翻倍；
2. **接触阶段**：接触后动作幅度必须骤降，与失败分析 H4「精细调整阶段高速动作有害」直接对应；
3. **保持要求**：对准后不能继续推，正好命中 H5「峰值达标后无法保持」这一主要失效模式。

**结论**：这个数据集会把当前两种策略最薄弱的环节暴露出来。
迁移时不能只改维度，必须补终止判断 / 保持监督 / 闭环冻结，否则会复现「覆盖率到 1.0 又被推走」。

---

## 5. PushT → ALOHA 迁移清单

| 维度 | PushT 现状 | ALOHA 目标 | 需要改动 |
|---|---|---|---|
| 动作维度 | 2 维 `(x, y)` | **14 维关节角** | 策略输出头、动作归一化均值/方差、loss 维度、动作裁剪 |
| 状态维度 | 2 维 `agent_pos` | **14 维关节角** | 观测编码输入层、state 归一化统计量 |
| 图像 | 96×96×3（渲染后 resize） | **480×640×3** | resize 到 96×96 或换更强 encoder；注意拉伸比例 |
| 相机数 | 1 | 1（本数据集） | 保持单路即可；多视角验证需另选数据 |
| 控制频率 | ~10 Hz | **50 Hz** | 一次重规划预算从 100 ms 压到 **20 ms** |
| 重规划延迟预算 | Flow 1 步 2.67 ms / Diffusion 10 步 94.5 ms | 20 ms | **只有 Flow 1–3 步可用**；Diffusion 10 步超预算 4.7 倍，只能做离线对照 |
| 动作块长度 | Flow K=8 | 建议 K≈50（1 秒）或更短 | 重规划频率与块长度需按 50 Hz 重设；短块可缓解保持失效 |
| Episode 长度 | 300 步 | **500 步** | 评估循环、录像长度、逐帧序列长度 |
| 数据量 | 198 训练轨迹 | **50 集 / 25000 帧** | 数据更少，过拟合风险高 → 早停、增广、更强正则（参见稳定性分析） |
| 任务结构 | 单臂推动、覆盖率为奖励 | 双臂接触 + 精修 + 保持 | 需要新指标：峰值/终态对准误差、保持时长、接触阶段动作幅度 |
| 终止信号 | 无 | 需要 | 加终止头或闭环判定，避免「到位后继续推」 |

---

## 6. 下周迁移验证建议（按优先级）

1. **先打通数据管线**：14 维逐维归一化、图像 resize、`delta_timestamps` 与动作块对齐。
   用 `ds[0]` 的键名直接改 `flow_model.py` 的输入构造，不要复用 PushT 的硬编码维度。
2. **模型选 Flow 1 步起步**：50 Hz 下它是唯一在预算内的方案（2.67 ms）；
   Diffusion 只作为离线质量上限对照，不进实时闭环。
3. **训练加早停与增广**：50 集样本量小，参照既有稳定性分析（Diffusion 在 2.4 万帧上 epoch 5 就过拟合）。
4. **评估补三个指标**：峰值对准误差、终态保持时长、接触阶段动作幅度；
   直接复用 `frame_metrics.py` 的块边界 jerk 与停滞比。
5. **重点观察**：单视角遮挡、夹爪开合的离散性、以及是否复现「峰值达标后被推走」。

---

## 7. 复现

```bash
cd pusht-policy-bench/experiments/aloha_probe
conda activate py312-robot
# 若 shell 未初始化 conda，先执行 eval "$(conda shell.bash hook)"
# 数据默认缓存到工作区 datasets/；换位置则 export WZH_ROOT=<workspace-root>

python scripts/probe_aloha.py
```

脚本用 `Path(__file__)` 相对定位仓库，结果写入 `results/aloha_probe.json`；
首次运行会从 Hugging Face 拉取约 91 MB 数据（需要网络），之后走本地缓存。
只读 `meta/info.json` 对比同系列多视角配置的命令已在本实验记录里说明。

---

## 8. 产物与局限

```
docs/aloha_dataset_probe.md              本文
experiments/aloha_probe/RESULTS.md       单实验记录（速览 + 复现）
experiments/aloha_probe/scripts/probe_aloha.py   加载与格式探查
experiments/aloha_probe/results/aloha_probe.json 完整 features / 样本形状 / 任务描述
```

1. **只探查了 human 演示版**，未下载 scripted 版的视频（仅读元数据做多视角对比）。
2. **未做策略迁移训练**，所有迁移建议基于格式差异与既有 PushT 实验结论，需下周实测验证。
3. **延迟数字来自 PushT 的 A6000 实测**，14 维动作头与更大图像会让实际延迟上升，
   50 Hz 预算结论方向上成立，但精确值需在 ALOHA 上重新测。
4. **单视角结论限于 lerobot 仿真系列**，不代表 ALOHA 真机数据集。
