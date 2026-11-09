"""最小化 BC（行为克隆）策略：CNN 视觉编码器 + MLP 状态编码器 + 融合 MLP + 回归动作头。

网络结构（对照 README.md 中的结构图）：

    图像 (3,96,96)                状态 (state_dim,)
        │                              │
    ConvEncoder(4×Conv+Linear)    MLP(state_dim→hidden)
        │  img_feat (hidden,)          │  state_feat (hidden,)
        └──────── concat (2*hidden) ───┘
                     │
              MLP(2*hidden→hidden→hidden)
                     │
              Linear → 动作 (action_dim,)   ← 单步回归，BC 的动作头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv_encoder(out_dim: int) -> nn.Sequential:
    """小 CNN：4 层 stride=2 卷积把 96x96 特征图逐层减半，最后展平投影到 out_dim。"""
    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.GroupNorm(1, cout),  # 等价于 LayerNorm，小 batch 下比 BatchNorm 稳定
            nn.ReLU(inplace=True),
        )

    encoder = nn.Sequential(
        block(3, 32),    # 96 -> 48
        block(32, 32),   # 48 -> 24
        block(32, 32),   # 24 -> 12
        block(32, 32),   # 12 -> 6
        nn.Flatten(),
        nn.Linear(32 * 6 * 6, out_dim),
        nn.ReLU(inplace=True),
    )
    # kaiming 初始化
    for m in encoder.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return encoder


def make_resnet18_encoder(out_dim: int) -> nn.Module:
    """ResNet18（ImageNet 预训练）视觉编码器，输出 out_dim 维特征。"""
    from torchvision.models import resnet18, ResNet18_Weights

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_feats = net.fc.in_features  # 512
    net.fc = nn.Sequential(nn.Linear(in_feats, out_dim), nn.ReLU(inplace=True))
    return net


def make_mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for cin, cout in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(cin, cout), nn.ReLU(inplace=True)]
    mlp = nn.Sequential(*layers)
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return mlp


class BCPolicy(nn.Module):
    """BC 策略：输入当前图像 + 状态，直接回归下一动作。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        backbone: str = "cnn",  # "cnn" | "resnet18"
    ):
        super().__init__()
        if backbone == "resnet18":
            self.vision_encoder = make_resnet18_encoder(hidden_dim)
        else:
            self.vision_encoder = make_conv_encoder(hidden_dim)
        self.state_encoder = make_mlp([state_dim, hidden_dim, hidden_dim])
        self.trunk = make_mlp([2 * hidden_dim, hidden_dim, hidden_dim])
        self.dropout = nn.Dropout(dropout)
        # 动作头：纯线性回归。BC 与 ACT/Diffusion 的全部区别就在这一层。
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        image: torch.Tensor,  # (B, 3, H, W)，值域 [0, 1]
        state: torch.Tensor,  # (B, state_dim)，已归一化
    ) -> torch.Tensor:  # (B, action_dim)，归一化动作空间
        img_feat = self.vision_encoder(image)
        state_feat = self.state_encoder(state)
        feat = self.trunk(torch.cat([img_feat, state_feat], dim=-1))
        return self.action_head(self.dropout(feat))

    def loss(self, batch: dict) -> torch.Tensor:
        """BC 的训练目标：预测动作与演示动作的 MSE（在归一化动作空间上计算）。"""
        pred = self.forward(batch["image"], batch["state"])
        return F.mse_loss(pred, batch["action"])
