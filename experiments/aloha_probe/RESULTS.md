# ALOHA 双臂数据集探查（2026-11-13）

**完整笔记**：[docs/aloha_dataset_probe.md](../../docs/aloha_dataset_probe.md)
**原始数据**：[aloha_probe.json](results/aloha_probe.json)（features、样本形状、任务描述、API 差异）

## 一句话结论

`lerobot/aloha_sim_insertion_human` 加载成功：**50 集 / 25000 帧 / 50 fps**，
动作与状态同为 **14 维关节角**（左右臂各 6 关节 + 1 夹爪）；
观测**只有一路 top 相机** 480×640，并非多视角；任务是双臂协同插销入孔。
迁移难点是 **50 Hz 控制预算**（20 ms/次重规划，只有 Flow 1–3 步可用）与
**插销任务必需的「到位后保持」能力**。

## 实测规格

| 项目 | 值 |
|---|---|
| 轨迹数 / 总帧数 | 50 episodes / 25000 frames（每集 500 帧） |
| fps | 50（每集约 10 秒） |
| `action` / `observation.state` | `[14]` float32，关节角，两臂各 7 维 |
| 图像 | `observation.images.top`，`[480, 640, 3]`，样本返回 `[3, 480, 640]` |
| 相机数 | 1（top） |
| 任务 | `Insert the peg into the socket.` |
| 样本键 | `action`, `observation.state`, `observation.images.top`, `task`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `next.done` |

**14 维拆解**：左臂 `left_waist / shoulder / elbow / forearm_roll / wrist_angle / wrist_rotate / gripper`，
右臂同构。属关节角位置控制，不是末端位姿；归一化需逐维统计。

**多视角核对**：额外只读了 ALOHA 仿真系列另外三个数据集的 `meta/info.json`
（`aloha_sim_insertion_scripted`、`aloha_sim_transfer_cube_human`、`aloha_sim_transfer_cube_scripted`），
四个数据集的相机键都只有 `observation.images.top`，动作/状态均为 14 维、50 集 / 50 fps。
真机 ALOHA 常见的 top + 左右腕部三路配置，在这批仿真数据里没有导出。

## API 差异（任务书示例基于旧版 lerobot）

| 旧写法 | 0.7.1 状态 | 替代 |
|---|---|---|
| `dataset.observation_keys` | 已移除，会报错 | `dataset.features` / `dataset.meta.info['features']` |
| `dataset.num_episodes` | 可用 | — |
| `dataset[0]['action'].shape` | 可用 | 返回 `torch.Size([14])` |
| 不指定 `root` | 落到默认 HF 缓存 | 显式 `root=` 便于复现 |
| 直接读图像 | 需视频解码 | `download_videos=True` |

## 复现

```bash
cd pusht-policy-bench/experiments/aloha_probe
conda activate py312-robot
# 若 shell 未初始化 conda，先执行 eval "$(conda shell.bash hook)"
# 数据默认缓存到工作区 datasets/；换位置则 export WZH_ROOT=<workspace-root>

python scripts/probe_aloha.py
```

首次运行从 Hugging Face 拉取约 91 MB（top 相机视频 88 MB），之后走本地缓存。
脚本用 `Path(__file__)` 相对定位仓库，只写 `results/aloha_probe.json`，不改动原项目代码。

多视角对比用的只读元信息查询（不下载视频）：
```bash
# 读取同系列数据集的 meta/info.json，筛 observation.images.* 键
python3 -c "import json,urllib.request as u; d=json.load(u.urlopen(\
\"https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/main/meta/info.json\")); \
print([k for k in d[\"features\"] if k.startswith(\"observation.images\")])"
```

## 与 PushT 的迁移清单（摘要）

| 维度 | PushT | ALOHA |
|---|---|---|
| action / state | 2 / 2 | **14 / 14** |
| 图像 | 96×96 | 480×640 |
| 控制频率 | ~10 Hz | **50 Hz** |
| 动作块 | K=8 | 建议按 50 Hz 重设 |
| episode | 300 步 | 500 步 |
| 数据量 | 198 训练轨迹 | 50 集 / 25000 帧 |
| 任务 | 单臂推块 | 双臂接触 + 精修 + 保持 |

详见完整笔记第 5–6 节：输出头/归一化/图像编码器改动、50 Hz 延迟预算（Flow 1 步 2.67 ms 可用，
Diffusion 10 步 94.5 ms 超预算）、早停与增广、以及「终止/保持监督」的必要性。

## 验收

| 验收项 | 状态 |
|---|---|
| 成功加载 ALOHA 数据集 | ✅ 50 集 / 25000 帧，样本可索引 |
| 理解双臂动作空间维度 | ✅ 14 维关节角，左右各 7，state 与 action 同构 |
| 理解观测多视角设置 | ✅ 实测单 top 相机，并核对同系列 4 个数据集 |
| 理解任务类型 | ✅ 插销入孔，双臂协同 + 接触 + 保持 |
| 为下周迁移验证做准备 | ✅ 11 项迁移清单 + 5 条优先级建议 |

## 局限

只探查 human 演示版；未做策略迁移训练；延迟数字来自 PushT 的 A6000 实测，需在 ALOHA 上重测；
单视角结论限于 lerobot 仿真系列，不代表真机数据集。
