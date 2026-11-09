# 环境搭建：两条路径与自检清单

## 方案 A：pip 安装

```bash
conda create -n lerobot python=3.12 -y && conda activate lerobot

# 顺序很重要：先装与驱动匹配的 torch，再装 lerobot
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install lerobot
```

| 优点 | 缺点 |
|---|---|
| 一条命令到位，依赖版本由包维护者锁定 | 无法修改源码：改默认超参、改 loss、打断点都做不到 |
| 升级简单 | 包版本可能落后于官方仓库，文档示例对不上 |
| 冲突概率低 | 遇到 bug 只能等新版，无法自行打补丁 |

**适用**：只想跑通官方 demo，不改算法。

## 方案 B：源码安装

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[training]"

# 手动对齐 torch（驱动不支持新 CUDA 时必须显式指定）
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

以模块方式运行（脚本内部使用相对导入，直接执行 `.py` 会报 `attempted relative import with no known parent package`）：

```bash
python -m lerobot.scripts.lerobot_eval --help
python -m lerobot.scripts.lerobot_train --help
```

| 优点 | 缺点 |
|---|---|
| 可改源码、可新增策略、可调试 | 需自行对齐 torch / torchvision / CUDA |
| 与官方仓库版本一致 | 必须用 `python -m` 启动 |
| 便于追踪配置如何流入训练循环 | 未 `pip install -e` 时需手动设 `PYTHONPATH` |

**适用**：需要二次开发、理解内部机制、迁移旧版权重。

## 选择建议

| 场景 | 选择 |
|---|---|
| 只验证模型能否跑通 | A |
| 要改超参默认值 / 改损失 / 加新策略 | B |
| 两者都要 | 直接用 B，一个环境足够 |

本仓库采用 B：需要迁移旧 checkpoint、修正配置中的 `device` 字段、并验证训练-保存-评测闭环。

## 环境自检

```bash
python - <<'PY'
import sys, torch, torchvision, lerobot
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__, "| cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("lerobot    ", lerobot.__version__)
print("gpu        ", torch.cuda.is_available())
PY

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

三项必须同时满足：

1. `torch.version.cuda` 的主版本 **不超过** 驱动支持的最高 CUDA 版本
2. `torch.cuda.is_available()` 为 `True`
3. `nvidia-smi` 能正常输出（连不上驱动说明设备节点未暴露给当前进程）

## CUDA 版本对应

| torch wheel | 需要驱动 | 说明 |
|---|---|---|
| cu126 | >= 525 | 兼容性最好，推荐 |
| cu128 | >= 525 | 可用 |
| cu129 及以上 | >= 580 | 旧驱动报 "driver too old" |

判断方法：`nvidia-smi` 右上角的 `CUDA Version` 是驱动支持的**上限**，torch 自带的 runtime 不能超过它。

## 容器与共享机器

- Docker 需显式暴露 GPU：`docker run --gpus all ...`
- 沙箱环境常屏蔽 `/dev/nvidia*`，此时 `nvidia-smi` 与 `torch.cuda.is_available()` 同时失败，需在真实终端验证
- 共享机器请用仓库级 `git config --local`，避免覆盖他人的全局身份

## 可选依赖取舍

`torchcodec` 编译时可能链接较新的 CUDA runtime，与旧驱动冲突，表现为导入后 CUDA 直接不可用。若出现：

```bash
pip uninstall -y torchcodec
```

LeRobot 会自动回退到 PyAV 解码，功能不受影响，仅解码略慢。
