"""数据加载与归一化。

复用 lerobot 的 LeRobotDataset 读取本地 v3.0 格式数据集（parquet + mp4 解码它都处理好了），
本文件只做两件事：
1. 指定用哪些特征（图像 / 状态 / 动作）；
2. 按 meta/stats.json 的统计量做归一化（BC 在归一化后的动作空间上算 MSE）。
"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class BCDataset(torch.utils.data.Dataset):
    """单帧 (图像, 状态) -> 当前动作。单步 BC 不用历史帧、不用动作分块。"""

    def __init__(
        self,
        root: str,
        image_key: str = "observation.image",
        state_key: str = "observation.state",
        action_key: str = "action",
        episodes: list[int] | None = None,
        augment: bool = False,  # 仅训练集开启：随机平移 + 亮度/对比度抖动
        max_shift: int = 8,
    ):
        # repo_id 仅作标识，真正的数据目录由 root 指定
        self.ds = LeRobotDataset(repo_id="local/bc", root=root, episodes=episodes)
        self.image_key = image_key
        self.state_key = state_key
        self.action_key = action_key
        self.augment = augment
        self.max_shift = max_shift

        stats = self.ds.meta.stats
        # 图像：lerobot v3.0 返回的已是 [0,1] float 帧，stats 也是 [0,1] 量纲；
        # 对 uint8（0-255）输入做兼容判断，避免二次缩放
        self.img_mean = torch.tensor(stats[image_key]["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.img_std = torch.tensor(stats[image_key]["std"], dtype=torch.float32).view(-1, 1, 1)
        if self.img_mean.max() > 1.5:  # stats 基于 0-255 的旧格式数据集
            self.img_mean, self.img_std = self.img_mean / 255.0, self.img_std / 255.0
        self.state_mean = torch.tensor(stats[state_key]["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats[state_key]["std"], dtype=torch.float32)
        self.action_mean = torch.tensor(stats[action_key]["mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats[action_key]["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        item = self.ds[idx]
        image = item[self.image_key].float()
        if image.max() > 1.5:  # uint8 帧（0-255）才除 255，float 帧已是 [0,1]
            image = image / 255.0
        if image.shape[-1] == 3:  # 防御：若是 (H,W,C) 则转成 (C,H,W)
            image = image.permute(2, 0, 1)
        state = item[self.state_key].float()
        action = item[self.action_key].float().squeeze()
        if self.augment:
            # 图像与状态/动作联动平移：内容右移 dx 像素 <=> agent 的世界坐标 x 增大
            image, dx, dy = self._shift(image)
            scale = 512.0 / image.shape[-1]  # 96px 图像对应 512 世界坐标
            # 图像 y 向下、世界 y 向上：内容下移 dy 像素 <=> 世界 y 减小
            state = state + torch.tensor([dx * scale, -dy * scale])
            action = action + torch.tensor([dx * scale, -dy * scale])
        image = (image - self.img_mean) / self.img_std
        state = (state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        return {"image": image, "state": state, "action": action}

    def _shift(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """随机平移（reflect 填充后裁剪），返回 (图像, dx, dy)：
        内容在图像中右移 dx、下移 dy 像素（图像坐标 y 向下）。"""
        p = self.max_shift
        padded = torch.nn.functional.pad(image[None], (p, p, p, p), mode="reflect")[0]
        top = torch.randint(0, 2 * p + 1, (1,)).item()
        left = torch.randint(0, 2 * p + 1, (1,)).item()
        out = padded[:, top:top + image.shape[1], left:left + image.shape[2]]
        return out, p - left, p - top
