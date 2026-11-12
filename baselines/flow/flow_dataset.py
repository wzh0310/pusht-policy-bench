"""Flow Policy 数据集：在 BC 基础上把动作标签换成未来 K 步窗口。

LeRobotDataset 的 delta_timestamps 按 fps 自动对齐历史/未来帧，
item['action'] 形状变为 (K, action_dim)。
"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class FlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        horizon: int,
        fps: int = 10,
        image_key: str = "observation.image",
        state_key: str = "observation.state",
        action_key: str = "action",
        episodes: list[int] | None = None,
        augment: bool = False,
        max_shift: int = 8,
    ):
        self.ds = LeRobotDataset(
            repo_id="local/flow",
            root=root,
            episodes=episodes,
            delta_timestamps={action_key: [i / fps for i in range(horizon)]},
        )
        self.image_key = image_key
        self.state_key = state_key
        self.action_key = action_key
        self.augment = augment
        self.max_shift = max_shift
        self.horizon = horizon

        stats = self.ds.meta.stats
        self.img_mean = torch.tensor(stats[image_key]["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.img_std = torch.tensor(stats[image_key]["std"], dtype=torch.float32).view(-1, 1, 1)
        if self.img_mean.max() > 1.5:
            self.img_mean, self.img_std = self.img_mean / 255.0, self.img_std / 255.0
        self.state_mean = torch.tensor(stats[state_key]["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats[state_key]["std"], dtype=torch.float32)
        self.action_mean = torch.tensor(stats[action_key]["mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats[action_key]["std"], dtype=torch.float32)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item[self.image_key].float()
        if image.max() > 1.5:
            image = image / 255.0
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        state = item[self.state_key].float()
        action = item[self.action_key].float()  # (K, A)
        if self.augment:
            image, dx, dy = self._shift(image)
            scale = 512.0 / image.shape[-1]
            state = state + torch.tensor([dx * scale, -dy * scale])
            action = action + torch.tensor([dx * scale, -dy * scale])
        image = (image - self.img_mean) / self.img_std
        state = (state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        return {"image": image, "state": state, "action": action}

    def _shift(self, image):
        p = self.max_shift
        padded = torch.nn.functional.pad(image[None], (p, p, p, p), mode="reflect")[0]
        top = torch.randint(0, 2 * p + 1, (1,)).item()
        left = torch.randint(0, 2 * p + 1, (1,)).item()
        out = padded[:, top:top + image.shape[1], left:left + image.shape[2]]
        return out, p - left, p - top
